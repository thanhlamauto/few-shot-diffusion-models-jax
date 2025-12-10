# FID Update: Generation (Not Reconstruction)

## Change Summary

Updated the per-class FID implementation based on user feedback to use **generation** instead of **reconstruction**.

---

## Original Plan vs Final Implementation

### Original Plan (Reconstruction Task)
- 600 sets, each with **1 target + 5 support images**
- Task: Reconstruct the target image given 5 support images
- Real images: The 600 target images
- Generated images: 600 reconstructions

### Final Implementation (Generation Task) ✅
- **100 sets, each with 6 random support images**
- Task: Generate NEW images given support set (not reconstructing any specific target)
- Real images: All 600 class images
- Generated images: 600 new images (6 per support set via leave-one-out)

---

## Why This Is Better

1. **True Few-Shot Generation**
   - Generating new instances of a class given examples
   - Not just reconstructing existing images

2. **More Aligned with Training**
   - Training generates new samples, not reconstructions
   - Evaluation should match training objective

3. **Cleaner Logic**
   - No need to track which image is "target" vs "support"
   - All 6 images in a set are support images
   - `leave_one_out_c` naturally creates 6 conditionings per set

4. **Better Use of Data**
   - Real images: uses ALL 600 class images
   - Generated: creates 600 NEW images
   - Fair comparison between real and generated distributions

---

## How It Works

### Step 1: Create Support Sets
```python
n_sets = 100
for i in range(n_sets):
    # Sample 6 random images from class (with replacement)
    support_indices = np.random.choice(n_class_images, size=6, replace=True)
    support_set = class_images[support_indices]  # (6, C, H, W)
```

### Step 2: Generate via Leave-One-Out
```python
# For each set of 6 images, leave_one_out_c creates 6 conditionings:
# - Conditioning 0: uses images [1,2,3,4,5] as support
# - Conditioning 1: uses images [0,2,3,4,5] as support
# - ... (6 conditionings total)

c_cond, _ = leave_one_out_c(mini_batch, ...)  # (bs*6, hdim)

# Generate 6 samples per set
samples = sample_ema(shape=(bs*6, C, H, W), conditioning=c_cond, ...)
```

### Step 3: Compute FID
```python
# Real: all 600 class images
real_images = class_images[:600]  # (600, C, H, W)

# Generated: 100 sets * 6 samples/set = 600
generated_images = all_generated[:600]  # (600, C, H, W)

fid_score = fid_jax.compute_fid(real_hwc, generated_hwc, inception_fn)
```

---

## Visualization

Each Wandb example shows a 3×6 grid:

```
Row 0: [6 support images] ← Blue border
Row 1: [6 generated images] ← Green border
Row 2: [6 real images] ← Red border
```

This shows:
- **Support**: What the model sees
- **Generated**: What the model creates
- **Real**: What the class actually looks like

---

## Code Changes

### `main_jax.py` Changes:

1. **Line ~280-307**: Changed support set creation
   - From: 600 sets with (1 target + 5 support)
   - To: 100 sets with (6 random support)

2. **Line ~309-350**: Updated generation loop
   - From: Generate 1 sample per set (600 total)
   - To: Generate 6 samples per set (600 total)

3. **Line ~352**: Changed real images
   - From: `batch_sets[:, 0, :, :, :]` (targets only)
   - To: `class_images[:600]` (all class images)

4. **Line ~722-780**: Updated visualizations
   - From: 1 row (target | 5 support | 1 generated)
   - To: 3 rows (6 support | 6 generated | 6 real)

---

## Expected Console Output

```
Computing per-class FID at step 20000...

======================================================================
Computing Per-Class FID with 600 samples
======================================================================

🎯 Randomly selected class: 'apple' (ID: 12)
✅ Loaded 600 images from class 'apple'

🔧 Creating 100 support sets (6 random images each, with replacement)...
   Will generate 600 images total (using first 600 for FID)
✅ Created 100 sets, shape: (100, 6, 3, 32, 32)

🎨 Generating 600 samples (6 per set via leave-one-out)...
Generating samples: 100%|████████████| 100/100 [02:15<00:00, 0.74sets/s]

✅ Generated images: (600, 32, 32, 3)
✅ Real images: (600, 32, 32, 3)
   Image value range: [-0.95, 0.98]

🔄 Computing FID score for class 'apple'...
✅ FID Score: 45.23 (Class: apple)
======================================================================
```

---

## Key Differences from Original

| Aspect | Original (Reconstruction) | Final (Generation) |
|--------|---------------------------|-------------------|
| Task | Reconstruct target given support | Generate new images given support |
| Sets | 600 sets | 100 sets |
| Images per set | 6 (1 target + 5 support) | 6 (all support) |
| Samples per set | 1 | 6 |
| Total samples | 600 | 600 |
| Real images | 600 targets | 600 class images |
| Target concept | Explicit | None |
| Conditioning | 5 images → reconstruct 1 | All 6 via leave-one-out |

---

## Benefits

✅ **Simpler Logic**: No target tracking, all images are support

✅ **More Realistic**: Generating NEW instances, not copying existing ones

✅ **Better Evaluation**: Matches training objective (generation, not reconstruction)

✅ **Cleaner Code**: Leverages existing `leave_one_out_c` fully

✅ **Fair Comparison**: Real vs Generated from same distribution

---

## Testing

Syntax verified:
```bash
python -m py_compile main_jax.py
# ✅ No errors
```

Ready to run:
```bash
python main_jax.py --compute_fid --fid_num_samples 600
```

---

## Files Modified

- ✅ `main_jax.py`: Updated FID computation logic (lines 280-380, 720-780)
- ✅ `PER_CLASS_FID_IMPLEMENTATION.md`: Updated documentation
- ✅ `FID_UPDATE_GENERATION_NOT_RECONSTRUCTION.md`: This file (NEW)

---

## User Feedback That Triggered This Change

> "1 target + 5 support images, with replacement
> 
> I rethink, maybe 1 target is not necessary, just 5 support and generate new images is enough"

**Response:** Agreed and implemented! Using 6 support images (via leave-one-out) to generate 6 new images per set. This is cleaner and more aligned with few-shot generation.
