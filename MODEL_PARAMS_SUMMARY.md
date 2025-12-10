# Model Parameters Summary

## 📝 Configuration:

```bash
python main_jax.py \
    --model vfsddpm_jax \
    --dataset cifar100 \
    --sample_size 6 \
    --image_size 32 \
    --patch_size 2 \
    --batch_size 32 \
    --hidden_size 384 \
    --depth 12 \
    --num_heads 6 \
    --mlp_ratio 4.0 \
    --hdim 384 \              # After fix (was 256)
    --context_channels 384    # After fix (was 256)
```

---

## 🎯 **TOTAL MODEL PARAMETERS: 43.5M**

```
┌────────────────────────────────────────┐
│  TOTAL: 43,509,900 params (43.5M)     │
│                                        │
│  ├─ DiT (Generator):    32.7M (75.2%) │
│  └─ Encoder (Context):  10.8M (24.8%) │
└────────────────────────────────────────┘
```

---

## 📊 Detailed Breakdown:

### **1. DiT (Diffusion Transformer) - 32.7M params**

| Component | Params | Details |
|-----------|--------|---------|
| **Patch Embedding** | 4,608 | Linear: 12 → 384 |
| **Position Embedding** | 98,304 | 256 patches × 384 |
| **Time Embedding** | 245,760 | MLP: 256 → 384 → 384 |
| **Context Projection** | 147,840 | Dense: 384 → 384 ✅ |
| **DiT Blocks (×12)** | 31,938,048 | **2.66M per block** |
| **Final Layer** | 301,068 | Norm + adaLN + Linear |
| **TOTAL** | **32,735,628** | **~32.7M** |

#### **Per DiT Block (2.66M params):**
- LayerNorm (×2): 1,536
- Self-Attention: 591,360
  - QKV projection: 443,520
  - Output projection: 147,840
- MLP: 1,181,568
  - FC1 (384→1536): 591,360
  - FC2 (1536→384): 590,208
- adaLN-Zero (conditioning): 887,040

---

### **2. Encoder (sViT) - 10.8M params**

| Component | Params | Details |
|-----------|--------|---------|
| **SPT Embedding** | 28,032 | Project: 72 → 384 |
| **Position Embedding** | 98,688 | 257 positions × 384 |
| **Transformer Blocks (×6)** | 10,646,784 | **1.77M per block** |
| **Final LayerNorm** | 768 | Norm before output |
| **TOTAL** | **10,774,272** | **~10.8M** |

#### **Per Encoder Block (1.77M params):**
- LayerNorm (×2): 1,536
- Self-Attention: 591,360
- MLP: 1,181,568

---

## 📈 Comparison: Before vs After Fix

| Aspect | Before (hdim=256) | After (hdim=384) | Change |
|--------|-------------------|------------------|--------|
| **Encoder** | 4.8M | 10.8M | +6.0M (+124%) |
| **DiT** | 32.7M | 32.7M | +0.05M (+0.2%) |
| **Context Projection** | 98,688 | 147,840 | +49,152 (+50%) |
| **TOTAL** | **37.5M** | **43.5M** | **+6.0M (+16%)** |

### **Analysis:**

✅ **Encoder increase dominates:**
- hdim 256→384 increases encoder by **+124%**
- But encoder is only ~25% of total model
- → Overall increase: **+16%** (acceptable!)

✅ **DiT barely changes:**
- DiT uses `hidden_size=384` (always)
- Only context projection changes (256→384)
- → DiT increase: **+0.2%** (negligible)

✅ **Context projection:**
- Before: 256 × 384 = 98,688 params
- After: 384 × 384 = 147,840 params
- → +50% params for this layer
- But this is critical for avoiding dimension mismatch! ✅

---

## 💾 Memory Estimates:

### **Training (float32):**

```
Model params:           43.5M × 4 bytes = 174 MB
Optimizer states (Adam): 43.5M × 8 bytes = 348 MB  (m, v)
Gradients:              43.5M × 4 bytes = 174 MB
EMA params:             43.5M × 4 bytes = 174 MB

Subtotal:                              ~870 MB

Activations (batch_size=32, depth=12):
  DiT activations:      ~2-3 GB (estimate)
  Encoder activations:  ~500 MB (estimate)

TOTAL TRAINING MEMORY: ~4-5 GB per device
```

### **Inference (float32):**

```
Model params:           174 MB
Activations:            ~500 MB (batch_size=16)

TOTAL INFERENCE MEMORY: ~700 MB per device
```

---

## 🎯 Key Takeaways:

### **1. Model Size:**
- **43.5M params total** - Medium-sized model
- DiT dominates: **75%** of params (32.7M)
- Encoder: **25%** of params (10.8M)

### **2. After Dimension Fix:**
- **+16% params** (37.5M → 43.5M)
- Mostly from encoder (hdim 256→384)
- **Trade-off is worth it:**
  - ✅ No information bottleneck
  - ✅ No expansion artifacts (256→384)
  - ✅ Better context quality
  - ✅ Expected better FID

### **3. Comparison to Other Models:**

| Model | Params | Notes |
|-------|--------|-------|
| **VFSDDPM (ours)** | **43.5M** | DiT-based, few-shot |
| DiT-S/2 | 33M | Single-class, no context |
| DiT-B/2 | 130M | Larger backbone |
| U-Net (DDPM) | 35M | CNN-based |
| Stable Diffusion | 860M | Text-to-image |

→ Our model is **reasonably sized** for few-shot generation!

### **4. Computational Cost:**

**Per training step (batch_size=32):**
- Forward pass: ~100-150ms (GPU)
- Backward pass: ~200-300ms (GPU)
- Total: ~300-450ms per step

**Full training (200k steps):**
- Time: ~17-25 hours (1 GPU)
- On Kaggle (9h limit): ~72k-108k steps

### **5. Scaling Options:**

**If too large:**
- ✅ Reduce `depth`: 12 → 8 (saves ~10M params)
- ✅ Reduce `hidden_size`: 384 → 256 (saves ~15M params)
- ✅ Reduce encoder depth: 6 → 4 (saves ~3.5M params)

**If want larger:**
- ✅ Increase `depth`: 12 → 16 (adds ~10M params)
- ✅ Increase `hidden_size`: 384 → 512 (adds ~30M params)
- ⚠️ Don't reduce `hdim` below 384 (dimension mismatch issue!)

---

## 📝 Summary Table:

| Component | Params | % of Total | Key Feature |
|-----------|--------|------------|-------------|
| **DiT Blocks** | 31.9M | 73.4% | Main generator |
| **Encoder Blocks** | 10.6M | 24.5% | Context extraction |
| **Embeddings** | 0.5M | 1.2% | Patch + Position |
| **Projections** | 0.4M | 0.9% | Time + Context |
| **TOTAL** | **43.5M** | **100%** | Few-shot diffusion |

---

## ✅ Conclusion:

**43.5M parameters** với config này:
- ✅ Reasonable size cho few-shot learning
- ✅ Balance tốt giữa encoder (25%) và generator (75%)
- ✅ Dimension fix (+16% params) đáng giá cho quality
- ✅ Fit trong Kaggle memory (4-5GB training)
- ✅ Train được trong 9h Kaggle limit (~80k steps)

**Recommendation: Config này ổn! Có thể bắt đầu training!** 🚀
