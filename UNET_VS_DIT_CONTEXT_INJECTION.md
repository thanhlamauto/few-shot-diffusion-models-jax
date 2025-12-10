# So Sánh Cách Inject Context: U-Net vs DiT

## 📊 Tổng Quan:

| Aspect | U-Net (main.py) | DiT (main_jax.py) |
|--------|-----------------|-------------------|
| **Architecture** | ResNet + Attention | Pure Transformer |
| **hdim** | 256 | 384 (sau fix) |
| **context_channels** | 256 | 384 (sau fix) |
| **time_embed_dim** | model_channels × 4 | hidden_size (384) |
| **Context injection** | **CONCATENATE** | **ADD** |
| **Projection needed?** | Yes (concat → linear) | Yes (Dense layer) |

---

## 🔴 U-Net: CONCATENATE Approach

### **Code Location: `model/set_diffusion/unet.py`**

#### **1. ResBlock Init (Lines 217-223):**

```python
self.emb_layers = nn.Sequential(
    SiLU(),
    linear(
        emb_channels + context_channels,  # ← INPUT SIZE
        2 * self.out_channels if use_scale_shift_norm else self.out_channels,
    ),
)
```

**Key Point:** Linear layer input = `emb_channels + context_channels`

---

#### **2. ResBlock Forward (Lines 261-305):**

```python
def _forward(self, x, emb, context_emb=None):
    # ... (process input)
    
    # STEP 1: CONCATENATE time and context embedding
    if self.mode_conditioning not in [None, "lag", "None"]:
        if context_emb is None:
            context_emb = th.zeros(emb.shape).to(emb.device)
        # CONCATENATE! (line 278)
        emb = th.cat([emb, context_emb], dim=-1)
    
    # STEP 2: Pass concatenated embedding through linear layer
    emb_out = self.emb_layers(emb).type(h.dtype)
    
    # STEP 3: Apply to features
    while len(emb_out.shape) < len(h.shape):
        emb_out = emb_out[..., None]
    
    if self.use_scale_shift_norm:
        scale, shift = th.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1 + scale) + shift  # FiLM modulation
    else:
        h = h + emb_out  # Bias conditioning
```

---

#### **3. Flow Visualization:**

```
Time embedding:    (batch, time_embed_dim)     e.g., (32, 256)
Context embedding: (batch, context_channels)   e.g., (32, 256)
                              ↓
                        CONCATENATE
                              ↓
Combined:          (batch, 256 + 256)          → (32, 512)
                              ↓
                    Linear(512 → out_channels)
                              ↓
Output:            (batch, out_channels)       → (32, 128)
                              ↓
                        FiLM Modulation
```

---

#### **4. Dimensions in U-Net (main.py):**

```python
# From main.py, line 71-84:
model_channels = 64  # (default from model_and_diffusion_defaults)
hdim = 256
context_channels = 256

# Calculated:
time_embed_dim = model_channels * 4 = 64 * 4 = 256

# In ResBlock:
emb_channels = time_embed_dim = 256
context_channels = 256

# Linear layer input:
input_size = 256 + 256 = 512  ✅ No mismatch!
```

**→ U-Net KHÔNG CÓ dimension mismatch vì dùng CONCATENATE!**

---

## 🟢 DiT: ADD Approach (Sau Khi Fix)

### **Code Location: `model/set_diffusion/dit_jax.py`**

#### **1. Context Projection (Lines 323-337):**

```python
if self.mode_conditioning == "film":
    # Create Dense layer
    context_proj_layer = nn.Dense(
        self.hidden_size,  # 384
        kernel_init=nn.initializers.xavier_uniform()
    )
    
    if c is not None:
        # PROJECT context to hidden_size
        context_proj = context_proj_layer(c)  # (bs, 256) → (bs, 384) BEFORE
                                               # (bs, 384) → (bs, 384) AFTER FIX ✅
        
        # ADD to time embedding
        conditioning = t_emb + context_proj
```

---

#### **2. Flow Visualization:**

**BEFORE Fix (hdim=256, context_channels=256):**
```
Context: (batch, 256)
            ↓
    Dense(256 → 384)  ← UP-PROJECTION (expansion)
            ↓
Context proj: (batch, 384)
            ↓
Time emb:     (batch, 384)
            ↓
        ADD
            ↓
Conditioning: (batch, 384)
```

**AFTER Fix (hdim=384, context_channels=384):**
```
Context: (batch, 384)
            ↓
    Dense(384 → 384)  ← WEIGHTING (no expansion) ✅
            ↓
Context proj: (batch, 384)
            ↓
Time emb:     (batch, 384)
            ↓
        ADD
            ↓
Conditioning: (batch, 384)
```

---

## 🔍 So Sánh Chi Tiết:

### **1. Context Injection Method:**

| | U-Net | DiT |
|---|---|---|
| **Method** | Concatenate then Linear | Project then Add |
| **Formula** | `Linear([t_emb; c])` | `t_emb + Dense(c)` |
| **Combined dim** | 512 (256+256) | 384 (same) |
| **Advantage** | Simple, no dimension constraint | Cleaner separation |
| **Disadvantage** | Larger linear layer | Requires matching dims |

---

### **2. Parameter Count:**

**U-Net (CONCATENATE):**
```
Linear layer input:  512 (time + context)
Linear layer output: out_channels (e.g., 128)

Params per ResBlock: 512 × 128 = 65,536 params
```

**DiT (ADD, before fix):**
```
Context projection:  256 × 384 = 98,304 params
Time projection:     384 × 384 = 147,456 params (in adaLN)

Total: ~245K params per block
```

**DiT (ADD, after fix):**
```
Context projection:  384 × 384 = 147,456 params
Time projection:     384 × 384 = 147,456 params (in adaLN)

Total: ~295K params per block
```

---

### **3. Tại Sao U-Net Không Cần Fix Dimension?**

**U-Net sử dụng CONCATENATE:**
- Time: 256 dims
- Context: 256 dims
- **Combined: 512 dims** → No problem with any hdim!

**DiT sử dụng ADD:**
- Time: 384 dims
- Context: **MUST be 384 dims** to add!
- Before fix: 256 → 384 projection needed
- After fix: 384 → 384 no expansion ✅

---

### **4. Ưu Nhược Điểm:**

#### **U-Net CONCATENATE:**

**✅ Advantages:**
1. Flexible dimensions (no matching required)
2. Time and context can have different dimensions
3. Simpler conceptually
4. Model learns how to combine them

**❌ Disadvantages:**
1. Larger linear layers (512 input vs 384)
2. More parameters per block
3. Less interpretable (mixed representation)
4. Harder to control time vs context influence

---

#### **DiT ADD:**

**✅ Advantages:**
1. Clean separation of time and context
2. Both processed independently then combined
3. Easier to interpret (additive)
4. Can control influence via projection weights
5. More parameter-efficient (no 512 → X layer)

**❌ Disadvantages:**
1. Requires dimension matching!
2. Before fix: expansion artifacts (256 → 384)
3. Need careful dimension planning

---

## 📐 Visualization:

### **U-Net (CONCATENATE):**

```
┌─────────────┐  ┌─────────────┐
│   t_emb     │  │  context_c  │
│   (256)     │  │    (256)    │
└──────┬──────┘  └──────┬──────┘
       │                │
       └────────┬───────┘
                │
         ┌──────▼──────┐
         │ Concatenate │
         │    (512)    │
         └──────┬──────┘
                │
         ┌──────▼──────────┐
         │ Linear(512→128) │
         │  65,536 params  │
         └──────┬──────────┘
                │
         ┌──────▼──────┐
         │   FiLM      │
         │  (scale +   │
         │   shift)    │
         └─────────────┘
```

### **DiT (ADD) - Before Fix:**

```
┌─────────────┐  ┌─────────────┐
│   t_emb     │  │  context_c  │
│   (384)     │  │    (256)    │
└─────────────┘  └──────┬──────┘
                        │
                 ┌──────▼───────────┐
                 │ Dense(256→384)   │
                 │  98,688 params   │
                 │  ↑ EXPANSION!    │
                 └──────┬───────────┘
       ┌────────────────┘
       │
┌──────▼──────┐
│     ADD     │
│   (384)     │
└──────┬──────┘
       │
┌──────▼──────┐
│   adaLN     │
│  (scale +   │
│   shift)    │
└─────────────┘
```

### **DiT (ADD) - After Fix:**

```
┌─────────────┐  ┌─────────────┐
│   t_emb     │  │  context_c  │
│   (384)     │  │    (384)    │ ✅ Match!
└─────────────┘  └──────┬──────┘
                        │
                 ┌──────▼───────────┐
                 │ Dense(384→384)   │
                 │  147,456 params  │
                 │  ↑ WEIGHTING!    │
                 └──────┬───────────┘
       ┌────────────────┘
       │
┌──────▼──────┐
│     ADD     │
│   (384)     │
└──────┬──────┘
       │
┌──────▼──────┐
│   adaLN     │
│  (scale +   │
│   shift)    │
└─────────────┘
```

---

## 🎯 Kết Luận:

### **1. Cả U-Net và DiT đều có context injection qua FiLM!**

- ✅ U-Net: Concatenate → Linear → FiLM
- ✅ DiT: Project → Add → adaLN-Zero (FiLM variant)

### **2. U-Net không có dimension mismatch issue:**

- **Reason:** CONCATENATE cho phép time và context có dims khác nhau
- Time 256 + Context 256 = 512 → Linear(512 → X) ✅

### **3. DiT có dimension mismatch issue (before fix):**

- **Reason:** ADD yêu cầu cùng dimension
- Time 384 + Context 256 → Need projection 256→384 ❌
- **Fix:** Set hdim=384, context_channels=384 ✅

### **4. Trade-off:**

| Aspect | U-Net (Concat) | DiT (Add) |
|--------|----------------|-----------|
| **Flexibility** | High ✅ | Low (needs match) |
| **Params per block** | 65K (example) | 295K (after fix) |
| **Interpretability** | Lower | Higher ✅ |
| **Architecture** | Hybrid (ResNet+Attn) | Pure Transformer ✅ |
| **Context quality** | Mixed | Separated ✅ |

### **5. Cả hai đều VALID!**

- U-Net CONCATENATE: Proven approach, flexible, simple
- DiT ADD: Cleaner, more interpretable, modern

**Fix dimension match trong DiT là để tận dụng ưu điểm của ADD approach!** ✅

---

## 💡 Recommendation:

**Nếu train từ đầu:**
- ✅ Dùng DiT với hdim=384, context_channels=384 (after fix)
- Better for pure Transformer architecture
- Cleaner context separation

**Nếu muốn flexibility:**
- ✅ Dùng U-Net với CONCATENATE approach
- No dimension constraints
- Proven architecture

**Cả hai đều inject context đúng cách, chỉ khác phương pháp!** 🎯
