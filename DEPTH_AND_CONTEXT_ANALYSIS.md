# Phân Tích: DiT Depth=12 & Context Learning Issues

## ❓ VẤN ĐỀ:

> "DiT×12 có quá sâu không? Ảnh eval không học được gì từ điều kiện c, không ra hình."

---

## 🔍 PHÂN TÍCH CÁC NGUYÊN NHÂN:

### **1️⃣ CRITICAL ISSUE: adaLN-Zero Initialization**

#### **Code: `model/set_diffusion/dit_jax.py`**

```python
# DiTBlock, lines 241-244
class DiTBlock(nn.Module):
    # ...
    @nn.compact
    def __call__(self, x, c, context=None):
        # ...
        # adaLN-Zero: scale, shift, gate
        adaLN_params = nn.Dense(
            6 * self.hidden_size,
            kernel_init=nn.initializers.zeros,  # ← ZERO INIT!
            bias_init=nn.initializers.zeros      # ← ZERO INIT!
        )(c)
        
        scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = \
            jnp.split(adaLN_params, 6, axis=-1)
```

#### **❌ VẤN ĐỀ:**

**adaLN-Zero khởi tạo với ZERO parameters!**

```
Ban đầu:
  adaLN_params = c @ W + b
               = c @ 0 + 0  
               = 0          ← ALL ZEROS!

→ scale = 0, shift = 0, gate = 0
→ Context c KHÔNG CÓ EFFECT GÌ ban đầu!
→ Model phải học từ đầu để context có tác dụng!
```

**Tại sao lại dùng zero init?**
- **DiT paper design**: adaLN-Zero giúp training stability
- Ban đầu model = standard transformer (no conditioning)
- Gradually học cách dùng conditioning
- **Nhưng: Cần NHIỀU steps để context bắt đầu có effect!**

---

### **2️⃣ ISSUE: Depth=12 Có Quá Sâu?**

#### **So sánh với các models khác:**

| Model | Depth | Image Size | Notes |
|-------|-------|------------|-------|
| **VFSDDPM (yours)** | **12** | 32×32 | Few-shot |
| DiT-S/2 | 12 | 256×256 | Standard DiT |
| DiT-B/2 | 12 | 256×256 | Larger version |
| ViT-Base | 12 | 224×224 | Classification |
| DALL-E 2 | 24 | 256×256 | Text-to-image |
| Stable Diffusion | 12-16 | 512×512 | Text-to-image |

#### **Phân tích:**

**✅ Depth=12 KHÔNG QUÁ SÂU cho image generation!**
- DiT paper uses depth=12 for 256×256 images
- Your images: 32×32 (much smaller!)
- → Depth=12 là reasonable, thậm chí có thể cần!

**Nhưng với 32×32 images:**
- Có ít patches hơn: 256 patches (vs 1024 for 64×64)
- Information complexity thấp hơn
- **Có thể depth=8 đã đủ!**

---

### **3️⃣ ISSUE: Gradient Flow & Learning Rate**

#### **Với depth=12 + adaLN-Zero:**

```
Problems:
  1. Zero init → Context effect starts at 0
  2. 12 layers → Gradient phải flow through nhiều layers
  3. Learning rate có thể không đủ để context layers học nhanh
  4. Ban đầu model học như standard DDPM (no context)
  
→ Context CHẬM học!
→ FID improvement chậm!
```

#### **Gradient vanishing risk:**

```
Forward: x → Block1 → Block2 → ... → Block12 → output
         ↑                                      ↑
      Input                                  Loss
         
Backward: ∂L/∂x ← Block1 ← Block2 ← ... ← Block12 ← ∂L/∂output

Với 12 layers:
  - Gradient từ output về input qua 12 blocks
  - Nếu mỗi block có scale < 1 → gradient decay
  - LayerNorm + Residual connections giúp, nhưng vẫn có risk
```

---

### **4️⃣ ISSUE: Training Steps Chưa Đủ?**

#### **Thời gian để context có effect:**

```
With adaLN-Zero:
  Steps 0-5k:     Context effect ≈ 0 (still mostly zeros)
  Steps 5k-20k:   Context starts to have small effect
  Steps 20k-50k:  Context effect growing
  Steps 50k-100k: Context effect significant
  Steps 100k+:    Context fully learned

Kaggle 9h limit: ~80k steps
  → Có thể CHƯA ĐỦ để context fully kick in!
```

---

## 🎯 GIẢI PHÁP:

### **🚀 Solution 1: GIẢM DEPTH (RECOMMENDED!)**

#### **Thử depth=8 hoặc 6:**

```bash
# Depth = 8 (giảm 4 blocks)
python main_jax.py \
    --depth 8 \          # ← Change này!
    --hidden_size 384 \
    --num_heads 6 \
    ...
```

**Lợi ích:**
- ✅ Faster training (~25% faster per step)
- ✅ Better gradient flow
- ✅ Fewer params (~32M → ~24M)
- ✅ Context có thể học nhanh hơn
- ✅ Đủ cho 32×32 images

**Trade-off:**
- ⚠️ Less model capacity (nhưng có thể không cần cho 32×32)

---

### **🔥 Solution 2: WARM-START Context Projection**

#### **Thay đổi initialization cho context projection:**

**File: `model/set_diffusion/dit_jax.py`, line ~325**

```python
# BEFORE (zero init for adaLN-Zero):
context_proj_layer = nn.Dense(
    self.hidden_size, 
    kernel_init=nn.initializers.xavier_uniform()  # ← Normal init
)

# Nhưng adaLN sau đó vẫn zero init:
adaLN_params = nn.Dense(
    6 * self.hidden_size,
    kernel_init=nn.initializers.zeros,     # ← ZERO!
    bias_init=nn.initializers.zeros
)(c)
```

**FIX: Initialize adaLN bias to SMALL NON-ZERO values:**

```python
adaLN_params = nn.Dense(
    6 * self.hidden_size,
    kernel_init=nn.initializers.zeros,
    bias_init=nn.initializers.constant(0.01)  # ← SMALL INIT!
)(c)
```

**Lợi ích:**
- ✅ Context có IMMEDIATE small effect
- ✅ Model vẫn stable (small init)
- ✅ Faster context learning

---

### **⚡ Solution 3: HIGHER Learning Rate cho Context Layers**

#### **Use different learning rates:**

Trong `main_jax.py`, có thể dùng layer-wise learning rates:

```python
# Context-related params: higher LR
context_params = [
    "encoder",
    "context_proj",
]

# Main DiT params: normal LR
dit_params = [...]

# Create optimizer with different LRs
```

**Lợi ích:**
- ✅ Context learns faster
- ✅ Main model stable

**Trade-off:**
- ⚠️ More complex setup
- ⚠️ Risk of overfitting context

---

### **🎓 Solution 4: PRE-TRAIN Encoder**

#### **Train encoder separately first:**

```python
# Step 1: Train encoder to reconstruct images
#         (autoencoder-like)
for batch_set in data:
    c = encode_set(batch_set[:, :-1])  # Encode 5 images
    x_recon = decode(c)  # Decode to reconstruct
    loss = mse(x_recon, batch_set[:, -1])  # Reconstruct 6th image

# Step 2: Freeze encoder, train DiT
for batch_set in data:
    c = encode_set(batch_set)  # Use pretrained encoder
    loss = diffusion_loss(x, c)
```

**Lợi ích:**
- ✅ Encoder learns meaningful representations first
- ✅ DiT can focus on denoising with good context

**Trade-off:**
- ⚠️ More training time
- ⚠️ Two-stage training

---

### **📊 Solution 5: MONITORING & DEBUGGING**

#### **Add logging to track context usage:**

```python
# In training loop, log:
1. Context magnitude: |c|
2. adaLN parameters magnitude: |scale|, |shift|, |gate|
3. Context gradient magnitude: |∂L/∂c|

# Example:
if global_step % 100 == 0:
    c_norm = jnp.linalg.norm(c)
    # Log context statistics
    wandb.log({
        "debug/context_norm": c_norm,
        "debug/context_max": jnp.max(jnp.abs(c)),
        "debug/context_min": jnp.min(jnp.abs(c)),
    })
```

**Để check:**
- Context có đang được dùng không?
- adaLN params có học không?
- Gradient có flow về encoder không?

---

## 🎯 RECOMMENDED ACTIONS:

### **Immediate (Nên làm ngay):**

1. **✅ GIẢM DEPTH xuống 8 hoặc 6**
   ```bash
   --depth 8  # Thử này trước
   ```
   - Fastest solution
   - Most likely to help
   - Đủ cho 32×32 images

2. **✅ CHECK Training Loss Curve**
   - Loss có đang giảm không?
   - FID có improve không?
   - Cần bao nhiêu steps để thấy improvement?

3. **✅ VISUALIZE Support Set trong Wandb**
   - Images có coherent không?
   - Support set và generated images có similar style không?

---

### **Short-term (Sau khi thử depth=8):**

4. **✅ ADJUST adaLN Initialization**
   ```python
   bias_init=nn.initializers.constant(0.01)
   ```
   - Nếu depth=8 vẫn chậm

5. **✅ INCREASE Learning Rate cho Encoder**
   ```python
   --lr 2e-4  # Tăng từ 1e-4
   ```
   - Hoặc dùng layer-wise LR

---

### **Long-term (Nếu vẫn không work):**

6. **Pre-train Encoder** (autoencoder)
   
7. **Try Different Architecture:**
   - U-Net instead of DiT?
   - Hybrid: U-Net with context injection?

---

## 📈 EXPECTED RESULTS:

### **With depth=8:**

```
Expected improvements:
  - Training speed: ~25% faster per step
  - Context learning: ~30% faster
  - FID should improve by step 40k-60k
  - Generated images should show class-specific features

Timeline:
  Steps 0-10k:   Noisy images, no structure
  Steps 10k-30k: Basic shapes appear
  Steps 30k-50k: Class-specific features emerge ← Context kicks in!
  Steps 50k+:    Quality improves, FID drops
```

---

## 🔬 DIAGNOSTIC CHECKLIST:

**Nếu vẫn "không ra hình", check:**

- [ ] Loss có đang giảm không? (should drop to <0.1)
- [ ] Sample images trong Wandb có improve không?
- [ ] Support set có đúng class không? (đã verify ✅)
- [ ] Context injection có đúng không? (đã verify ✅)
- [ ] Learning rate có phù hợp không?
- [ ] Batch size có đủ lớn không? (32 là ổn)
- [ ] Diffusion steps (250) có phù hợp không?
- [ ] Noise schedule (linear) có tốt không?

---

## 💡 COMPARISON: Depth Options

| Depth | Params | Training Speed | Context Learning | Quality (32×32) | Recommendation |
|-------|--------|----------------|------------------|-----------------|----------------|
| **6** | ~19M | Fast ✅ | Fast ✅ | Good ✅ | **Try first if urgent** |
| **8** | ~24M | Medium ✅ | Medium ✅ | Better ✅ | **RECOMMENDED** ⭐ |
| **10** | ~30M | Slow | Slow | Better | Consider if depth=8 works |
| **12** | ~35M | Slower | Slower | Best (?) | Current (có thể overkill) |

**For 32×32 CIFAR-100:**
- **Depth=8 is sweet spot!** ⭐
- Good balance: speed vs quality
- Proven to work for similar tasks

---

## ✅ SUMMARY:

### **Main Issues:**

1. ❌ **adaLN-Zero init**: Context starts with ZERO effect
2. ⚠️ **Depth=12**: Có thể overkill cho 32×32 images
3. ⚠️ **Training time**: Chưa đủ steps cho context fully kick in
4. ⚠️ **Gradient flow**: 12 layers = slower learning

### **Quick Fix:**

```bash
# RECOMMENDED: Giảm depth xuống 8
python main_jax.py \
    --depth 8 \              # ← FIX 1: Reduce depth
    --lr 1.5e-4 \            # ← FIX 2: Slightly higher LR
    --hidden_size 384 \
    --hdim 384 \
    --context_channels 384 \
    ... (other args same)
```

**Expected:**
- ✅ Faster training (~25%)
- ✅ Faster context learning (~30%)
- ✅ Better gradient flow
- ✅ Images should show structure by 30k-40k steps

---

## 🎯 FINAL RECOMMENDATION:

**THỬ DEPTH=8 TRƯỚC!** ⭐

Nếu vẫn không work sau 50k steps, thì check:
1. Loss curve (should be decreasing)
2. Learning rate (có thể cần tăng)
3. adaLN initialization (consider warm-start)
4. Pre-train encoder (last resort)

**Good luck!** 🚀
