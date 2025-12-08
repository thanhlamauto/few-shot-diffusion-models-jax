"""
Train a JAX/Flax diffusion model (DiT backbone) with pmap on multi-device (e.g., TPU v5e-8).

This script mirrors the structure of main.py but targets vfsddpm_jax.
"""

import argparse
import dataclasses
import jax
import numpy as np
import flax
import flax.serialization as serialization
import orbax.checkpoint as ocp
import os
import wandb
from tqdm import tqdm

from dataset import create_loader
from model import select_model  # keeps existing namespace for non-JAX
from model.select_model_jax import select_model_jax
from model.vfsddpm_jax import vfsddpm_loss, leave_one_out_c
from model.set_diffusion import logger
from metrics import fid_jax
from model.set_diffusion.train_util_jax import (
    create_train_state_pmap,
    shard_batch,
    train_step_pmap,
    sample_ema,
)
from model.set_diffusion.script_util_jax import (
    add_dict_to_argparser as add_dict_to_argparser_jax,
    args_to_dict as args_to_dict_jax,
    model_and_diffusion_defaults as model_and_diffusion_defaults_jax,
)
from utils.path import set_folder
from utils.util import set_seed


DIR = set_folder()


def numpy_from_torch(batch):
    # Assume batch is a torch Tensor on CPU; values in [0,1] -> scale to [-1,1]
    arr = batch.detach().cpu().numpy().astype(np.float32)
    if arr.max() <= 1.01 and arr.min() >= -0.01:
        arr = arr * 2.0 - 1.0
    return arr


def eval_loop(p_state, modules, cfg, loader, n_devices, num_batches):
    """
    Simple eval: run vfsddpm_loss (train=False) on a few batches (host-side).
    Uses EMA params for evaluation.
    """
    if num_batches <= 0:
        return 0.0
    params_eval = flax.jax_utils.unreplicate(p_state.ema_params)
    losses = []
    it = iter(loader)
    for _ in tqdm(range(num_batches), desc="Eval", unit="batch"):
        try:
            batch = next(it)
        except StopIteration:
            break
        batch_np = numpy_from_torch(batch)
        # use full batch on host (single device) for eval
        loss_dict = vfsddpm_loss(jax.random.PRNGKey(
            0), params_eval, modules, batch_np, cfg, train=False)
        losses.append(np.array(loss_dict["loss"]))
    if not losses:
        return 0.0
    return float(np.mean(losses))


def sample_loop(p_state, modules, cfg, loader, num_batches, rng, use_ddim, eta):
    """
    Minimal sampling: use first batches from loader, build conditioning, and run diffusion.
    Saves .npz files with samples/cond and returns samples for logging.
    """
    if num_batches <= 0:
        return None, None
    ema_params = flax.jax_utils.unreplicate(p_state.ema_params)
    diffusion = modules["diffusion"]
    dit = modules["dit"]
    encoder = modules["encoder"]
    posterior = modules.get("posterior")

    it = iter(loader)
    all_samples = []
    all_support = []

    for i in tqdm(range(num_batches), desc="Sampling", unit="batch"):
        try:
            batch = next(it)
        except StopIteration:
            break
        batch_np = numpy_from_torch(batch)
        b, ns, c, h, w = batch_np.shape
        rng, sub = jax.random.split(rng)
        # build conditioning c (host-side, deterministic)
        c_cond, _ = leave_one_out_c(sub, {"encoder": ema_params["encoder"], "posterior": ema_params.get(
            "posterior")}, modules, batch_np, cfg, train=False)
        # sampling shape
        shape = (b * ns, c, h, w)
        # model_apply using ema_params["dit"]
        model_apply = lambda params, x, t, c=None, **kw: dit.apply(
            params, x, t, c=c, **kw)
        samples = sample_ema(sub, ema_params["dit"], diffusion, model_apply,
                             shape, conditioning=c_cond, use_ddim=use_ddim, eta=eta)
        # save npz
        out_path = os.path.join(DIR, f"samples_{i:03d}.npz")
        np.savez(out_path, samples=np.array(samples), cond=batch_np)

        # Collect for wandb logging
        all_samples.append(np.array(samples))
        all_support.append(batch_np)

    return np.concatenate(all_samples, axis=0) if all_samples else None, \
        np.concatenate(all_support, axis=0) if all_support else None


def compute_fid_4096(p_state, modules, cfg, val_loader, n_samples, rng, use_ddim, eta, inception_fn, samples_per_support_set=50):
    """
    Compute FID score by generating n_samples images from different support sets.

    Args:
        p_state: Parallel training state with EMA parameters
        modules: Dict of model modules
        cfg: Model configuration
        val_loader: Validation data loader
        n_samples: Number of samples to generate for FID (e.g., 4096)
        rng: JAX random key
        use_ddim: Whether to use DDIM sampling
        eta: DDIM eta parameter
        inception_fn: InceptionV3 apply function for FID
        samples_per_support_set: Number of samples to generate from each support set (default=50)

    Returns:
        fid_score: FID score (lower is better)
    """
    print(f"\n=== Computing FID with {n_samples} samples ===")

    from model.vfsddpm_jax import leave_one_out_c
    from model.set_diffusion.train_util_jax import sample_ema

    all_generated = []
    all_real = []

    it = iter(val_loader)
    n_batches = 0
    total_generated = 0

    pbar = tqdm(total=n_samples, desc="FID Generation", unit="samples")

    while total_generated < n_samples:
        try:
            batch = next(it)
        except StopIteration:
            # Restart iterator if we run out of data
            it = iter(val_loader)
            batch = next(it)

        # batch shape: (bs, ns, C, H, W)
        batch_np = np.array(batch)
        bs, ns, C, H, W = batch_np.shape

        # Unreplicate EMA params from pmap
        ema_params = flax.jax_utils.unreplicate(p_state.ema_params)

        # Get conditioning via leave-one-out (COMPUTE ONCE per support set)
        sub = {"encoder": ema_params["encoder"],
               "posterior": ema_params.get("posterior")}
        rng, cond_rng = jax.random.split(rng)
        c_cond, _ = leave_one_out_c(
            cond_rng, sub, modules, batch_np, cfg, train=False)

        # Generate MULTIPLE samples from the SAME support set (BATCHED)
        # Instead of 20 calls with 6 samples each, make 1 call with 120 samples
        n_repeats = min(samples_per_support_set, (n_samples - total_generated + bs*ns - 1) // (bs*ns))
        if n_repeats == 0:
            break

        total_samples_this_batch = bs * ns * n_repeats  # e.g., 1 * 6 * 20 = 120

        # Replicate conditioning: (6, hdim) -> (120, hdim)
        # Each conditioning vector is repeated n_repeats times
        c_replicated = jax.numpy.repeat(c_cond, n_repeats, axis=0)

        # Sample (SINGLE BATCHED CALL - 15-18x faster than loop)
        diffusion = modules["diffusion"]
        model_apply = modules["dit"].apply
        shape_batched = (total_samples_this_batch, C, H, W)

        rng, sample_rng = jax.random.split(rng)
        # Force DDIM for FID (much faster: 50-100 steps vs 1000)
        samples = sample_ema(
            sample_rng, ema_params["dit"], diffusion, model_apply,
            shape_batched, conditioning=c_replicated, use_ddim=True, eta=0.0
        )

        # Convert samples from (120, C, H, W) to (120, H, W, C) for InceptionV3
        samples_np = np.array(samples)
        samples_hwc = samples_np.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        # Replicate real images to match: (6, C, H, W) -> (120, C, H, W)
        real_np = batch_np.reshape(-1, C, H, W)  # (6, C, H, W)
        real_replicated = np.tile(real_np, (n_repeats, 1, 1, 1))  # (120, C, H, W)
        real_hwc = real_replicated.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        all_generated.append(samples_hwc)
        all_real.append(real_hwc)

        total_generated += samples_hwc.shape[0]
        n_batches += 1

        pbar.update(samples_hwc.shape[0])
        pbar.set_postfix({"support_sets": n_batches})

        if total_generated >= n_samples:
            break

    pbar.close()

    # Concatenate all samples
    generated_images = np.concatenate(all_generated, axis=0)[:n_samples]
    real_images = np.concatenate(all_real, axis=0)[:n_samples]

    print(f"Generated images shape: {generated_images.shape}")
    print(f"Real images shape: {real_images.shape}")
    print(
        f"Image value range: [{generated_images.min():.2f}, {generated_images.max():.2f}]")

    # Compute FID
    try:
        fid_score = fid_jax.compute_fid(
            real_images,
            generated_images,
            inception_fn=inception_fn,
            batch_size=256  # Increased from 64 for 2-4x faster Inception extraction
        )
        print(f"FID Score: {fid_score:.2f}")
        return fid_score
    except Exception as e:
        print(f"Error computing FID: {e}")
        return None


def main():
    args = create_argparser().parse_args()
    set_seed(getattr(args, "seed", 0))
    if not hasattr(args, "model_path"):
        args.model_path = ""

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    print("\nArgs:")
    for k in sorted(vars(args)):
        print(k, getattr(args, k))
    print()

    logger.configure(dir=DIR, mode="training", args=args, tag="jax")

    n_devices = jax.local_device_count()
    if args.batch_size % n_devices != 0:
        raise ValueError(
            f"batch_size {args.batch_size} must be divisible by n_devices {n_devices}")

    rng = jax.random.PRNGKey(getattr(args, "seed", 0))
    rng, rng_model = jax.random.split(rng)

    params, modules, cfg = select_model_jax(args, rng_model)

    # Train state and optimizer
    state, tx = create_train_state_pmap(
        params,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    p_state = jax.device_put_replicated(state, jax.local_devices())

    def loss_fn(p, batch, rng_in):
        return vfsddpm_loss(rng_in, p, modules, batch, cfg, train=True)

    p_train_step = train_step_pmap(
        tx, loss_fn, ema_rate=float(str(args.ema_rate).split(",")[0]))

    # Data loaders (PyTorch) on CPU
    train_loader = create_loader(args, split="train", shuffle=True)
    val_loader = create_loader(args, split="val", shuffle=False)

    # Initialize InceptionV3 for FID computation (if enabled)
    inception_fn = None
    if args.compute_fid:
        print("Loading InceptionV3 for FID computation...")
        try:
            inception_fn = fid_jax.get_fid_fn()
            if inception_fn is not None:
                print("InceptionV3 loaded successfully!")
            else:
                print("Warning: InceptionV3 could not be loaded.")
                print("FID computation will be skipped.")
        except Exception as e:
            print(f"Warning: Could not load InceptionV3: {e}")
            print("FID computation will be skipped.")
            inception_fn = None

    # Checkpointing
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    ckpt_dir = os.path.join(DIR, "checkpoints_jax")
    os.makedirs(ckpt_dir, exist_ok=True)

    def save_checkpoint(step_int, rng_save):
        host_state = jax.device_get(p_state)
        ckpt = {
            "params": host_state.params,
            "ema_params": host_state.ema_params,
            "opt_state": host_state.opt_state,
            "step": int(step_int),
            "rng": rng_save,
            # Convert dataclass to dict for Orbax
            "cfg": dataclasses.asdict(cfg),
        }
        checkpointer.save(os.path.join(
            ckpt_dir, f"ckpt_{step_int:06d}"), ckpt, force=True)

    def load_checkpoint(path):
        nonlocal p_state, rng
        if not path:
            return
        if not os.path.exists(path):
            logger.log(f"resume_checkpoint not found: {path}")
            return
        host_state = checkpointer.restore(path)
        p_state = jax.device_put_replicated(
            p_state.replace(
                params=host_state["params"],
                ema_params=host_state["ema_params"],
                opt_state=host_state["opt_state"],
                step=host_state["step"],
            ),
            jax.local_devices(),
        )
        rng = host_state.get("rng", rng)
        logger.log(f"loaded checkpoint {path} at step {host_state['step']}")

    load_checkpoint(args.resume_checkpoint)

    logger.log("starting training (jax pmap)...")
    global_step = 0
    for epoch in range(10**6):  # effectively infinite unless steps reached
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
        for batch in pbar:
            batch_np = numpy_from_torch(batch)
            try:
                batch_sharded = shard_batch(batch_np, n_devices)
            except AssertionError:
                # skip incomplete batch
                continue

            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, n_devices)

            p_state, metrics = p_train_step(p_state, batch_sharded, step_rngs)

            # host metrics
            metrics_host = jax.tree.map(lambda x: np.array(x).mean(), metrics)
            global_step += 1

            # Update progress bar with metrics
            pbar.set_postfix({
                'step': global_step,
                'loss': f"{metrics_host.get('loss', 0):.4f}" if 'loss' in metrics_host else 'N/A'
            })

            if global_step % args.log_interval == 0:
                logger.logkv("step", global_step)
                for k, v in metrics_host.items():
                    if isinstance(v, np.ndarray):
                        v = v.item() if v.size == 1 else v
                    logger.logkv_mean(k, v)
                logger.dumpkvs(global_step)
                if args.use_wandb:
                    wandb.log(metrics_host, step=global_step)

            # Save / Eval / Sample
            if args.save_interval and global_step % args.save_interval == 0:
                save_checkpoint(global_step, rng)
                eval_loss = eval_loop(
                    p_state, modules, cfg, val_loader, n_devices, args.num_eval_batches)
                logger.logkv("eval_loss", eval_loss)
                logger.dumpkvs(global_step)

                # Generate samples and log to wandb
                samples, support = sample_loop(
                    p_state, modules, cfg, val_loader, args.num_sample_batches, rng, args.use_ddim, args.eta)

                if args.use_wandb:
                    log_dict = {"eval_loss": eval_loss}

                    # Log sample images to wandb
                    if samples is not None and support is not None:
                        # Take first few samples for logging (max 16)
                        n_log = min(16, samples.shape[0])
                        sample_images = []

                        for idx in range(n_log):
                            # Denormalize from [-1, 1] to [0, 1] if needed
                            img = samples[idx].transpose(1, 2, 0)  # CHW -> HWC
                            # [-1,1] -> [0,1]
                            img = np.clip((img + 1) / 2, 0, 1)
                            sample_images.append(wandb.Image(
                                img, caption=f"Sample {idx}"))

                        log_dict["samples"] = sample_images

                        # Log support set images
                        support_images = []
                        # Flatten (B, ns, C, H, W) -> (B*ns, C, H, W)
                        support_flat = support.reshape(-1, *support.shape[2:])
                        for idx in range(min(8, support_flat.shape[0])):
                            img = support_flat[idx].transpose(
                                1, 2, 0)  # CHW -> HWC
                            img = np.clip((img + 1) / 2, 0, 1)
                            support_images.append(wandb.Image(
                                img, caption=f"Support {idx}"))

                        log_dict["support_set"] = support_images

                # Compute FID if enabled
                if args.compute_fid and inception_fn is not None:
                    print(f"\nComputing FID at step {global_step}...")
                    fid_score = compute_fid_4096(
                        p_state, modules, cfg, val_loader,
                        args.fid_num_samples, rng, args.use_ddim, args.eta, inception_fn
                    )
                    if fid_score is not None:
                        logger.logkv("fid", fid_score)
                        logger.dumpkvs(global_step)
                        if args.use_wandb:
                            log_dict["fid"] = fid_score

                if args.use_wandb:
                    wandb.log(log_dict, step=global_step)

            # Check stopping conditions
            if (args.lr_anneal_steps and global_step >= args.lr_anneal_steps) or \
               (args.max_steps and global_step >= args.max_steps):
                break
        if (args.lr_anneal_steps and global_step >= args.lr_anneal_steps) or \
           (args.max_steps and global_step >= args.max_steps):
            break

    logger.log("training complete.")
    if args.use_wandb:
        wandb.finish()


def create_argparser():
    defaults = dict(
        model="vfsddpm_jax",
        dataset="cifar100",
        image_size=32,
        sample_size=5,
        patch_size=8,
        hdim=256,
        in_channels=3,
        encoder_mode="vit_set",
        pool="cls",
        context_channels=256,
        mode_context="deterministic",
        mode_conditioning="film",
        augment=False,
        data_dir="/home/gigi/ns_data",
        num_classes=1,
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        max_steps=0,  # 0 means infinite, set to positive number to limit training
        batch_size=16,
        batch_size_eval=1,
        log_interval=100,
        save_interval=10000,
        num_eval_batches=10,
        num_sample_batches=2,
        ema_rate="0.9999",
        resume_checkpoint="",
        clip_denoised=True,
        use_ddim=True,
        eta=0.0,
        tag=None,
        seed=0,
        use_wandb=True,
        wandb_project="fsdm-jax",
        wandb_run_name=None,
        compute_fid=False,
        # Reduced from 4096 for faster eval (still reliable)
        fid_num_samples=1024,
    )
    defaults.update(model_and_diffusion_defaults_jax())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser_jax(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
