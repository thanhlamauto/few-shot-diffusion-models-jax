# Freeze DiT Training Mode

## Purpose

Train **only the encoder** for the first N steps while **freezing DiT weights**. This helps:
1. **Debug encoder learning**: Check if encoder learns meaningful representations
2. **Stabilize training**: Encoder learns first, then DiT adapts to encoder's output
3. **Prevent gradient conflicts**: Encoder and DiT don't fight each other initially

## How It Works

```
Steps 1-20000:  ✅ Encoder trains   ❌ DiT frozen
Steps 20001+:   ✅ Encoder trains   ✅ DiT trains (unfrozen)
```

**What gets trained when DiT is frozen:**
- ✅ Encoder (ViT/sViT)
- ✅ `time_embed` (timestep MLP for encoder)
- ✅ Posterior (if variational mode)
- ❌ DiT (completely frozen)

**Gradients**: DiT gradients are zeroed out during first N steps

## Usage

### Basic: Freeze DiT for 20K steps

```bash
python main_jax.py \
  --model vfsddpm_jax \
  --dataset cifar100 \
  --data_dir /kaggle/working/ns_data \
  --sample_size 6 \
  --image_size 32 \
  --patch_size 2 \
  --batch_size 32 \
  --lr 1e-4 \
  --freeze_dit_steps 20000 \
  --max_steps 300000 \
  --log_interval 1000 \
  --save_interval 20000 \
  --use_wandb True \
  --diffusion_steps 250 \
  --hidden_size 468 \
  --depth 6 \
  --num_heads 9 \
  --mlp_ratio 3.0
```

### Shorter freeze (10K for quick test):

```bash
python main_jax.py \
  ... \
  --freeze_dit_steps 10000 \
  --max_steps 100000
```

### No freeze (normal training):

```bash
python main_jax.py \
  ... \
  --freeze_dit_steps 0  # or just omit this flag
```

## Expected Output

### At step 0:
```
======================================================================
⚠️  FREEZE DiT MODE ENABLED:
   DiT will be FROZEN for first 20000 steps
   Only encoder (+ time_embed + posterior if variational) will train
   DiT will unfreeze at step 20001
======================================================================
```

### During training (steps 1-20000):
```
step: 5000
training_mode: FROZEN_DiT (step 5000/20000)
loss: 0.1234
debug/dit_frozen: 1.0
debug/grad_norm_encoder: 0.52  # Encoder gradients active
debug/grad_norm_dit: 0.00      # DiT gradients zeroed
```

### At step 20001 (unfreeze):
```
======================================================================
🔓 DiT UNFROZEN at step 20001
   Now training BOTH encoder and DiT
======================================================================
```

### After unfreeze (steps 20001+):
```
step: 25000
training_mode: FULL (encoder+DiT)
loss: 0.0987
debug/dit_frozen: 0.0
debug/grad_norm_encoder: 0.48  # Encoder gradients active
debug/grad_norm_dit: 0.35      # DiT gradients active
```

## Monitoring

### WandB Metrics:
- `debug/dit_frozen`: 1.0 = frozen, 0.0 = training
- `training_mode`: "FROZEN_DiT" or "FULL"
- `debug/grad_norm_encoder`: Should be > 0 always
- `debug/grad_norm_dit`: Should be 0 when frozen, > 0 when unfrozen

### Context Magnitude:
Watch `debug/context_*` metrics:
- Should increase during frozen phase (encoder learning)
- Should stabilize after unfreeze

## Typical Use Cases

### 1. Debug Encoder (freeze 20K):
```bash
--freeze_dit_steps 20000 --max_steps 50000
```
→ Check if encoder learns meaningful features before DiT training

### 2. Stabilize Training (freeze 10K):
```bash
--freeze_dit_steps 10000 --max_steps 300000
```
→ Encoder warms up, then full training

### 3. Compare with Normal Training:
**Run A**: `--freeze_dit_steps 0` (baseline)
**Run B**: `--freeze_dit_steps 20000` (freeze mode)
→ Compare FID/quality

## Implementation Details

**File**: `model/set_diffusion/train_util_jax.py`
```python
def train_step_pmap(..., freeze_dit_steps=0):
    # During frozen phase: zero out DiT gradients
    if state.step < freeze_dit_steps:
        grads["dit"] = jax.tree.map(jnp.zeros_like, grads["dit"])
```

**Key Points**:
- Uses `jax.lax.cond` for efficient branching
- Zero gradients = no parameter updates
- EMA still tracks encoder (DiT EMA stays frozen too)
- Checkpoint saves both frozen and unfrozen states

## Tips

1. **Checkpoint before unfreeze**: `--save_interval 20000` to save at step 20K
2. **Watch context norms**: Should be non-zero during frozen phase
3. **Compare FID**: Compute FID at step 20K (frozen encoder) vs 40K (full training)
4. **Resume works**: Can resume from frozen checkpoint and continue normally

## Example: Full Training Pipeline

```bash
# Stage 1: Train encoder only (20K steps)
python main_jax.py \
  --freeze_dit_steps 20000 \
  --max_steps 20000 \
  --save_interval 20000 \
  --wandb_run_name "encoder_only" \
  ...

# Stage 2: Full training (resume from step 20K)
python main_jax.py \
  --freeze_dit_steps 0 \
  --resume True \
  --checkpoint_path checkpoints_jax/ckpt_020000 \
  --max_steps 300000 \
  --wandb_run_name "full_training" \
  ...
```
