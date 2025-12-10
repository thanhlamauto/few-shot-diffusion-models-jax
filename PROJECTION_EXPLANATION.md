# How Context Projection Worked Before (256→384)

## 🔍 **Complete Flow:**

### **1. Encoder Output (hdim=256):**

```python
# model/vfsddpm_jax.py, line ~217
def encode_set(self, x_subset, train: bool = False):
    """
    Args:
        x_subset: (bs, ns_subset, C, H, W) - support images
    Returns:
        hc: (bs, hdim) - context vector
    """
    hc = self.encoder.apply(
        self.encoder_params,
        x_subset,
        train=train,
        method=self.encoder.forward_set,
    )
    # hc shape: (bs, 256) when hdim=256 ✅
    return hc
```

**Output:** Context vector `c` with shape `(batch_size, 256)`

---

### **2. Pass to DiT:**

```python
# model/set_diffusion/gaussian_diffusion_jax.py
# training_losses() calls:
model_output = model(x_t, t, c, **model_kwargs)
                        #  ↑
                        # (bs, 256)
```

**Context `c`:** Still `(bs, 256)`

---

### **3. Inside DiT - FiLM Conditioning (THE PROJECTION HAPPENS HERE!):**

```320:337:model/set_diffusion/dit_jax.py
        if self.mode_conditioning == "film":
            # Create Dense layer unconditionally (required for Flax @nn.compact)
            context_proj_layer = nn.Dense(
                self.hidden_size, kernel_init=nn.initializers.xavier_uniform()
            )
            if c is not None:
                context_proj = context_proj_layer(c)
                conditioning = t_emb + context_proj
            else:
                # MUST call layer with dummy input to initialize parameters
                # AND add the zero projection to maintain consistent computation graph
                dummy_c = jnp.zeros(
                    (x.shape[0], self.context_channels), dtype=x.dtype)
                zero_context_proj = context_proj_layer(dummy_c)
                conditioning = t_emb + zero_context_proj  # FIX: consistent with c!=None case
        else:
            conditioning = t_emb
```

**Line 325-326:** The projection layer is created!
```python
context_proj_layer = nn.Dense(
    self.hidden_size,  # 384
    kernel_init=nn.initializers.xavier_uniform()
)
```

**Line 329:** The projection happens here!
```python
context_proj = context_proj_layer(c)
# Input:  c            → (bs, 256)
# Weight: W            → (256, 384)
# Bias:   b            → (384,)
# Output: context_proj → (bs, 384) = c @ W + b
```

**Line 330:** Context is added to time embedding
```python
conditioning = t_emb + context_proj
# t_emb:        (bs, 384)
# context_proj: (bs, 384)
# conditioning: (bs, 384) ✅
```

---

### **4. Mathematical Operation:**

```
Input:  c ∈ ℝ^(bs × 256)
Weight: W ∈ ℝ^(256 × 384)    <- Learned parameters
Bias:   b ∈ ℝ^(384)          <- Learned parameters

Output: context_proj = c @ W + b ∈ ℝ^(bs × 384)
```

**What happens:**
1. Each 256-dim context vector is multiplied by a 256×384 weight matrix
2. Add a 384-dim bias vector
3. Result: 384-dim projected context

---

## ⚠️ **Problems with This Approach:**

### **1. Up-Projection Adds Noise:**
```
256 dims → 384 dims = Adding 128 new dimensions
```
- Where do these extra 128 dimensions come from?
- **Answer:** Linear combination of original 256 dims + learned bias
- **Problem:** This can introduce artifacts or dilute information

### **2. Information Bottleneck:**
```
Encoder sees support images → Compresses to 256 dims → Expands to 384 dims
                                      ↓
                           Information lost here!
```

### **3. Unnecessary Complexity:**
```
256 params in encoder → Dense(256→384) adds 256×384 = 98,304 params
```
- Extra parameters to learn
- Extra computation at every step
- No clear benefit

---

## ✅ **After Fix (hdim=384):**

### **New Flow:**

```python
# 1. Encoder Output
hc = encoder(x_subset)  # (bs, 384) ✅

# 2. Pass to DiT
model(x_t, t, c=hc)     # c: (bs, 384) ✅

# 3. Inside DiT - FiLM
context_proj_layer = nn.Dense(384)  # (384, 384) - Identity-like
context_proj = context_proj_layer(c)  # (bs, 384) → (bs, 384)
conditioning = t_emb + context_proj   # Both 384! ✅
```

**Benefits:**
1. ✅ No dimension change (384 → 384)
2. ✅ No information bottleneck
3. ✅ Projection becomes learned weighting, not expansion
4. ✅ Cleaner, more direct context usage

---

## 📊 **Comparison:**

| Aspect | Before (256) | After (384) |
|--------|--------------|-------------|
| **Encoder output** | 256 dims | 384 dims |
| **Projection** | 256→384 (expand) | 384→384 (weight) |
| **Extra params** | 98,304 (256×384) | 147,456 (384×384) |
| **Information flow** | Bottleneck → Expand | Direct |
| **Complexity** | High | Medium |
| **Context quality** | Diluted | Rich |

---

## 🔬 **Technical Details:**

### **Dense Layer in Flax:**

```python
# Flax nn.Dense automatically infers input dimension
class Dense(nn.Module):
    features: int  # Output dimension (hidden_size=384)
    
    @nn.compact
    def __call__(self, x):
        # x shape: (bs, in_features)
        # Automatically creates: (in_features, features)
        kernel = self.param('kernel', 
                           self.kernel_init, 
                           (x.shape[-1], self.features))
        # When hdim=256: kernel is (256, 384)
        # When hdim=384: kernel is (384, 384)
        return x @ kernel + bias
```

**Key Point:** Input dimension is **inferred at runtime**, so:
- With `hdim=256`: Creates `(256, 384)` weight matrix
- With `hdim=384`: Creates `(384, 384)` weight matrix
- **No error either way!** But 256→384 is suboptimal.

---

## 🎯 **Why 384→384 is Better:**

### **1. No Expansion Artifacts:**
```
Before: 256 dims → Learn to fill 128 extra dims → May add noise
After:  384 dims → Learn to weight existing dims → Cleaner
```

### **2. Richer Encoder:**
```
Before: Encoder forced to compress to 256
After:  Encoder can use full 384 dims to represent support set
```

### **3. Better Gradient Flow:**
```
Before: Gradients → Dense(256→384) → Encoder(384→256) → Bottleneck
After:  Gradients → Dense(384→384) → Encoder(384) → Direct
```

---

## 📝 **Summary:**

**Before (hdim=256, context_channels=256):**
```
Support Set → Encoder → (256) → Dense(256→384) → (384) → DiT
                         ↑              ↑
                   Bottleneck    Up-projection
```

**After (hdim=384, context_channels=384):**
```
Support Set → Encoder → (384) → Dense(384→384) → (384) → DiT
                         ↑              ↑
                    Rich repr    Learned weighting
```

**The projection still exists (Dense layer), but now it's a 384→384 transformation (learned weighting/mixing) rather than a 256→384 expansion (adding dimensions)!** 🎯

---

## 🧪 **Verification:**

You can verify the projection by checking model parameters:

```python
# Before (hdim=256):
# context_proj_layer.kernel.shape = (256, 384)
# context_proj_layer.bias.shape = (384,)

# After (hdim=384):
# context_proj_layer.kernel.shape = (384, 384)
# context_proj_layer.bias.shape = (384,)
```

The Dense layer is still there, but now it's a **square matrix** (384×384) instead of a **rectangular matrix** (256×384)!
