# Quick FID Testing Commands for Kaggle

## 1. Quick Test with Random Weights (2-3 minutes)
**Purpose**: Verify FID code works without needing a checkpoint

```bash
cd /kaggle/working/few-shot-diffusion-models-jax
bash test_fid_kaggle.sh
```

Or directly:
```bash
python main_jax.py \
  --model vfsddpm_jax \
  --dataset cifar100 \
  --data_dir /kaggle/working/ns_data \
  --sample_size 6 \
  --image_size 32 \
  --patch_size 2 \
  --batch_size 32 \
  --compute_fid True \
  --fid_mode in \
  --fid_num_samples 20 \
  --max_steps 0 \
  --use_wandb False \
  --diffusion_steps 250 \
  --hidden_size 468 \
  --depth 6 \
  --num_heads 9 \
  --mlp_ratio 3.0
```

**Expected**: 
- No errors → ✅ Code works!
- FID will be HIGH (e.g., 200+) because of random weights (this is normal)

---

## 2. Test with Trained Checkpoint

### Quick (20 samples):
```bash
bash test_fid_kaggle.sh yes /kaggle/working/few-shot-diffusion-models-jax/checkpoints_jax/ckpt_020000
```

### Medium (1024 samples, ~10 minutes):
```bash
python main_jax.py \
  --model vfsddpm_jax \
  --dataset cifar100 \
  --data_dir /kaggle/working/ns_data \
  --sample_size 6 \
  --image_size 32 \
  --patch_size 2 \
  --batch_size 32 \
  --compute_fid True \
  --fid_mode in \
  --fid_num_samples 1024 \
  --checkpoint_path /kaggle/working/few-shot-diffusion-models-jax/checkpoints_jax/ckpt_020000 \
  --resume True \
  --max_steps 0 \
  --use_wandb False \
  --diffusion_steps 250 \
  --hidden_size 468 \
  --depth 6 \
  --num_heads 9 \
  --mlp_ratio 3.0
```

### Full Paper Eval (10K samples, ~1 hour):
```bash
python main_jax.py \
  --model vfsddpm_jax \
  --dataset cifar100 \
  --data_dir /kaggle/working/ns_data \
  --sample_size 6 \
  --image_size 32 \
  --patch_size 2 \
  --batch_size 32 \
  --compute_fid True \
  --fid_mode both \
  --fid_num_samples 10000 \
  --checkpoint_path /kaggle/working/few-shot-diffusion-models-jax/checkpoints_jax/ckpt_020000 \
  --resume True \
  --max_steps 0 \
  --use_wandb False \
  --diffusion_steps 250 \
  --hidden_size 468 \
  --depth 6 \
  --num_heads 9 \
  --mlp_ratio 3.0
```

---

## 3. Test Only FID Computation (No Model, Only Real Images)

**Purpose**: Test FID code in isolation (fastest)

```bash
python test_fid_real_only.py \
  --dataset cifar100 \
  --data_dir /kaggle/working/ns_data \
  --n_samples 20
```

**Expected**: FID should be LOW (< 50) since both groups are real images

---

## FID Mode Options

- `in`: IN-distribution FID (train classes) - 60 classes for CIFAR100
- `out`: OUT-distribution FID (test classes) - 20 unseen classes for CIFAR100  
- `both`: Compute both IN and OUT
- `per_class`: Single random class (for debugging)

---

## Troubleshooting

### Error: "checkpoint not found"
→ Use random weights test: `bash test_fid_kaggle.sh`

### Error: "leave_one_out_c() missing argument 't'"
→ Already fixed in latest code

### FID score is very high (> 200)
→ Normal if using random weights
→ With trained checkpoint, should be < 100

### Out of memory
→ Reduce `--fid_num_samples` to 100 or 20
