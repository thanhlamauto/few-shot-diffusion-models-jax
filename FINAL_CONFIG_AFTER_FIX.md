# Final Configuration After Fix

## Command Example:
```bash
python main_jax.py \
  --batch_size 32 \
  --sample_size 6 \
  --diffusion_steps 250 \
  --compute_fid True \
  --max_steps 300000 \
  --log_interval 300
```

## Resulting Config (Merged):

### Model Architecture:
```yaml
model: vfsddpm_jax
hidden_size: 468           # DiT hidden dimension (từ script_util_jax)
depth: 6                   # Number of DiT blocks (từ script_util_jax)
num_heads: 9               # Attention heads (từ script_util_jax)
mlp_ratio: 3.0             # MLP expansion (từ script_util_jax)
patch_size: 2              # Patch size (từ script_util_jax, OVERRIDE main_jax's 8)
dropout: 0.0               # Dropout rate (từ script_util_jax)
class_cond: False          # Class conditioning disabled
class_dropout_prob: 0.1    # For classifier-free guidance
use_fp16: False            # No mixed precision
```

### Encoder:
```yaml
encoder_mode: vit_set      # Set-based ViT encoder
hdim: 448                  # Encoder output dimension (từ main_jax)
pool: cls                  # CLS token pooling
mode_context: deterministic # No variational posterior
```

### Conditioning:
```yaml
context_channels: 448      # ✅ FIXED! Now matches hdim (was 450)
mode_conditioning: film    # FiLM conditioning (not LAG)
```

### Dataset:
```yaml
dataset: cifar100
data_dir: /kaggle/working/ns_data  # From CLI argument
image_size: 32
in_channels: 3
sample_size: 6             # From CLI (was default 5)
augment: False
num_classes: 1             # Not used
```

### Diffusion:
```yaml
diffusion_steps: 250       # From CLI (was default 1000)
noise_schedule: linear
timestep_respacing: ""     # No respacing
learn_sigma: False
use_kl: False
predict_xstart: False
rescale_timesteps: False
rescale_learned_sigmas: False
```

### Training:
```yaml
batch_size: 32             # From CLI (was default 16)
batch_size_eval: 16
lr: 0.0001                 # 1e-4
weight_decay: 0.0
ema_rate: 0.9999
seed: 0
max_steps: 300000          # From CLI
lr_anneal_steps: 0         # No LR annealing
```

### Logging & Checkpointing:
```yaml
log_interval: 300          # From CLI (was default 100)
save_interval: 20000
resume_checkpoint: ""      # No resume
tag: None
```

### Wandb:
```yaml
use_wandb: True
wandb_project: fsdm-jax
wandb_run_name: None       # Auto-generated
log_support_target: True
vis_num_sets: 2
```

### Evaluation & Sampling:
```yaml
num_eval_batches: 10
num_sample_batches: 2
use_ddim: True
eta: 0.0
clip_denoised: True
```

### FID:
```yaml
compute_fid: True          # From CLI
fid_num_samples: 1024
```

## Key Changes After Fix:

### ✅ Fixed:
- `context_channels: 450` → `448` (now matches hdim)
- No more shape mismatch error!

### Model Size:
- Total parameters: ~43M
- DiT: 468 hidden_size × 6 depth × 9 heads
- Head dimension: 468/9 = 52

### Memory Usage (estimated for TPU v5e-8):
- Batch size 32 with 8 devices = 4 samples per device
- Each sample: 6 images × 32×32×3
- Should fit comfortably in TPU memory

## Priority Order (lowest to highest):
1. `main_jax.py` defaults
2. `script_util_jax.py` defaults (via model_and_diffusion_defaults)
3. Command-line arguments (highest priority)

## Example Override:
```bash
# To change depth to 12 and hidden_size to 768:
python main_jax.py \
  --depth 12 \
  --hidden_size 768 \
  --num_heads 12 \
  --batch_size 16  # Reduce due to larger model
```

**Note:** Always ensure `hidden_size` is divisible by `num_heads`!
