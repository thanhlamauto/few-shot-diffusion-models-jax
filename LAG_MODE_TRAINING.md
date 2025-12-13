# Training with Cross-Attention (Lag Mode)

## Command Changes

### Key Parameters for Lag Mode:

1. **`--mode_conditioning lag`**: Switch from FiLM to cross-attention
2. **`--context_channels`**: Must match `--hdim` for optimal performance
3. **`--hdim`**: Encoder hidden dimension (also used for context tokens)

### Example Command for Lag Mode:

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
  --log_interval 300 \
  --save_interval 20000 \
  --num_eval_batches 10 \
  --num_sample_batches 2 \
  --use_wandb True \
  --wandb_project fsdm-jax \
  --max_steps 300000 \
  --diffusion_steps 250 \
  --hidden_size 468 \
  --depth 6 \
  --num_heads 9 \
  --mlp_ratio 3.0 \
  --hdim 256 \
  --context_channels 256 \
  --mode_conditioning lag \
  --encoder_mode vit_set \
  --encoder_depth 3 \
  --encoder_heads 8 \
  --encoder_dim_head 56 \
  --encoder_mlp_ratio 1.0 \
  --compute_fid True \
  --fid_num_samples 1024
```

### Key Differences from FiLM Mode:

1. **`--mode_conditioning lag`** instead of `film`
2. **`--context_channels`** should equal `--hdim` (e.g., both 256)
3. Conditioning shape changes:
   - **FiLM**: `(b*ns, hdim)` - single vector per image
   - **Lag**: `(b*ns, num_patches, hdim)` - patch tokens per image

### Number of Patch Tokens:

For CIFAR-100 (32x32 images, patch_size=2):
- `num_patches = (32 // 2)² = 16² = 256 tokens per image`

This gives much richer conditioning compared to FiLM's single vector!

## Configuration Validation

The training script will automatically:
- ✅ Check if `context_channels == hdim` (warns if not)
- ✅ Log conditioning shape: `(b*ns, num_patches, hdim)`
- ✅ Print full configuration at training start
- ✅ Show number of patch tokens per image

## Expected Log Output:

```
======================================================================
CONDITIONING CONFIGURATION:
======================================================================
  Mode: LAG
  ✅ Using CROSS-ATTENTION with patch tokens
  Patch tokens per image: 256
  Conditioning shape: (b*ns, 256, 256)
  ✅ context_channels (256) matches hdim (256)
======================================================================

======================================================================
FULL CONFIGURATION (for debugging):
======================================================================
Dataset: cifar100
Image size: 32
Sample size (ns): 6
Batch size: 32

Encoder:
  Mode: vit_set
  Hidden dim (hdim): 256
  Depth: 3
  Heads: 8
  Dim head: 56
  MLP ratio: 1.0
  Tokenize mode: stack
  Pool: cls
  Dropout: 0.0

DiT:
  Hidden size: 468
  Depth: 6
  Heads: 9
  MLP ratio: 3.0
  Patch size: 2
  Context channels: 256
  Mode conditioning: lag

Diffusion:
  Steps: 250
  Noise schedule: linear
  Learn sigma: False

Context:
  Mode: deterministic

Training:
  Learning rate: 0.0001
  Weight decay: 0.0
  Max steps: 300000
  EMA rate: 0.9999
======================================================================
```

## Comparison: FiLM vs Lag Mode

| Aspect | FiLM Mode | Lag Mode |
|--------|-----------|----------|
| Conditioning | Single vector `(b*ns, hdim)` | Patch tokens `(b*ns, num_patches, hdim)` |
| DiT Mechanism | adaLN-Zero (FiLM modulation) | Cross-attention |
| Information | Pooled encoder output | Full patch-level features |
| Parameters | Fewer (no cross-attn) | More (cross-attn layers) |
| Use Case | Simpler, faster | Richer conditioning |

## Notes

- **Memory**: Lag mode uses more memory due to cross-attention over patch tokens
- **Speed**: Slightly slower than FiLM due to cross-attention computation
- **Quality**: Should provide better conditioning for complex scenes
- **Compatibility**: Works with both ViT and sViT encoders
