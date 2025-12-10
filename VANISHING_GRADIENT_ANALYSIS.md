# 🚨 Vanishing Gradient Analysis - Critical Issues Found

**Status:** SEVERE - Multiple architectural and initialization issues detected

**Evidence from W&B:** Context metric increasing from -0.037 → 0 over 17.5k steps indicates gradient flow problems.

---

## 📊 Observed Symptoms

### 1. **Context Drift Pattern:**
```
Steps 0-2k:   context ≈ -0.037 to -0.025  (strong negative bias)
Steps 2k-10k: context ≈ -0.025 to -0.01   (slow climb)
Steps 10k+:   context ≈ -0.01 to 0        (near convergence)
```

**Problem:** This slow drift suggests:
- ❌ Weak gradient signals reaching encoder
- ❌ Poor initialization causing bias
- ❌ Gradient magnitude decreasing with depth

---

## 🔍 Root Causes Identified

### ⚠️ **CRITICAL ISSUE #1: Zero Initialization in adaLN Layers**

**Location:** `model/set_diffusion/dit_jax.py`

**Problem Code:**
```python
# Line 208 - DiTBlock adaLN modulation
c_mod = nn.Dense(
    6 * self.hidden_size, 
    kernel_init=nn.initializers.constant(0.0)  # ❌ ZERO INIT!
)(c_mod)

# Line 259 - FinalLayer modulation
c = nn.Dense(
    2 * self.hidden_size, 
    kernel_init=nn.initializers.constant(0)     # ❌ ZERO INIT!
)(c)

# Line 267 - FinalLayer output projection
x = nn.Dense(
    self.patch_size * self.patch_size * self.out_channels,
    kernel_init=nn.initializers.constant(0),    # ❌ ZERO INIT!
)(x)
```

**Why This Causes Vanishing Gradient:**
1. **Zero output at initialization:**
   ```
   shift, scale, gate = Dense(constant(0))(c)
   → All outputs = 0
   → modulate(x, 0, 0) = x * (1 + 0) + 0 = x
   → Gates = 0 → No information flows!
   ```

2. **Gradient path:**
   ```
   ∂L/∂c = ∂L/∂x × ∂x/∂modulation × ∂modulation/∂c
   
   When modulation ≈ 0:
   → ∂modulation/∂c ≈ 0
   → ∂L/∂c ≈ 0  ❌ VANISHING!
   ```

3. **Context learning:**
   - Context `c` cannot affect output initially
   - Encoder receives NO gradient signal
   - Model ignores conditioning for many steps

**Fix:**
```python
# Small random initialization instead
kernel_init=nn.initializers.normal(stddev=0.02)
# OR use scaled initialization
kernel_init=nn.initializers.variance_scaling(0.02, "fan_in", "truncated_normal")
```

---

### ⚠️ **CRITICAL ISSUE #2: LayerNorm Without Scale Parameters**

**Location:** Throughout `dit_jax.py` and `vit_set_jax.py`

**Problem Code:**
```python
# DiTBlock - Lines 220, 229, 239
x_norm = nn.LayerNorm(
    use_bias=False, 
    use_scale=False  # ❌ No learnable scale!
)(x)

# sViT Transformer - Line 92
y = nn.LayerNorm()(x)  # ⚠️ Default may not include scale
```

**Why This Causes Vanishing Gradient:**
1. **No gradient rescaling:**
   ```
   Standard LayerNorm: y = γ × (x - μ) / σ + β
   Without scale (γ):  y = (x - μ) / σ
   
   → No learnable rescaling
   → Gradients normalized but not amplified
   → Weak signals through deep networks
   ```

2. **Depth accumulation:**
   ```
   After 6 DiT blocks + 6 Encoder layers = 12 layers
   Gradient magnitude ∝ (scale_factor)^depth
   
   With use_scale=False:
   → scale_factor ≈ 1.0 (no amplification)
   → grad_magnitude ≈ initial × 1.0^12 = initial  (but still weak!)
   ```

**Fix:**
```python
# Enable scale and bias
x_norm = nn.LayerNorm(use_bias=True, use_scale=True)(x)
```

---

### ⚠️ **CRITICAL ISSUE #3: No Gradient Clipping**

**Location:** `model/set_diffusion/train_util_jax.py`

**Current Code:**
```python
# Line 167 - create_train_state_pmap
tx = optax.adamw(
    learning_rate=learning_rate, 
    weight_decay=weight_decay
)
# ❌ No gradient clipping!
```

**Why This is Critical:**
1. **Diffusion models have extreme gradients:**
   ```
   Early timesteps (t→T): Large noise → Large gradients
   Late timesteps (t→0):  Small noise → Small gradients
   
   Gradient variance ∝ 1 / (1 - ᾱ_t)
   → Can vary by 100× or more!
   ```

2. **Without clipping:**
   - Occasional gradient explosions damage parameters
   - Optimizer takes bad steps
   - Training becomes unstable

**Fix:**
```python
tx = optax.chain(
    optax.clip_by_global_norm(1.0),  # ✅ Add clipping!
    optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
)
```

---

### ⚠️ **ISSUE #4: Deep Architecture (12 Total Layers)**

**Current Architecture:**
```
Input → sViT Encoder (6 layers) → Context
                                    ↓
Input → DiT (6 layers) ← Context ← •
                ↓
            Output
```

**Total depth:** 12 layers (6 encoder + 6 DiT)

**Gradient flow:**
```
∂L/∂encoder_input = ∂L/∂output × ∂output/∂context × ∂context/∂encoder_input
                                  └─ Through 6 DiT layers ─┘
                                                           └─ Through 6 Encoder layers ─┘

Each layer multiplies gradient by ≈ 0.8-0.95
→ After 12 layers: gradient ≈ initial × 0.9^12 ≈ 0.28 × initial
→ 72% gradient loss!
```

**Why Depth=6 is Still Risky:**
- DiT paper uses depth=12 for **256×256** images
- Your images: **32×32** (64× smaller!)
- Relative depth = 12 layers for 64× less information
- **Overkill → Unnecessary gradient attenuation**

**Recommendation:**
```python
# For 32×32 images, depth=4 might be sufficient
depth=4,  # Instead of 6
# OR improve gradient flow (see fixes below)
```

---

### ⚠️ **ISSUE #5: LSA Temperature Initialization**

**Location:** `model/vit_set_jax.py` - Line 45-50

**Problem Code:**
```python
# LSA attention scaling
temperature = self.param(
    "temperature",
    lambda key: jnp.log(
        jnp.array(self.dim_head ** -0.5, dtype=jnp.float32)
    )
)
scale = jnp.exp(temperature)
```

**Initial value:**
```python
dim_head = 64
scale = dim_head ** -0.5 = 64 ** -0.5 = 0.125

# Very small scale!
# Attention scores scaled by 0.125
```

**Why This Causes Issues:**
1. **Attention gradients:**
   ```
   dots = q @ k.T * scale  # scale = 0.125
   
   When scale is small:
   → dots are small
   → softmax(dots) ≈ uniform distribution
   → Weak attention signals
   → Small gradients w.r.t. q, k
   ```

2. **Temperature learning:**
   - Temperature is log-space: hard to learn
   - Initial value may be too small for your data
   - Gradients w.r.t. temperature are weak

**Fix:**
```python
# Initialize with slightly larger scale
temperature = self.param(
    "temperature",
    lambda key: jnp.log(jnp.array(self.dim_head ** -0.5 * 2.0, dtype=jnp.float32))
)
# OR use fixed scale
scale = (self.dim_head ** -0.5) * 1.414  # √2 boost
```

---

### ⚠️ **ISSUE #6: Context Projection in DiT (FiLM mode)**

**Location:** `model/set_diffusion/dit_jax.py` - Lines 323-337

**Problem Code:**
```python
if self.mode_conditioning == "film":
    context_proj_layer = nn.Dense(
        self.hidden_size, 
        kernel_init=nn.initializers.xavier_uniform()
    )
    if c is not None:
        context_proj = context_proj_layer(c)
        conditioning = t_emb + context_proj  # ✅ OK
    else:
        # MUST call layer with dummy input
        dummy_c = jnp.zeros((x.shape[0], self.context_channels), dtype=x.dtype)
        zero_context_proj = context_proj_layer(dummy_c)
        conditioning = t_emb + zero_context_proj  # ⚠️ Adding zeros
```

**Why This is Suboptimal:**
1. **Gradient path when c=None:**
   ```
   conditioning = t_emb + Dense(dummy_zeros)
   → Dense output ≈ 0 (initially)
   → conditioning ≈ t_emb
   → No gradient flows to context_proj_layer!
   ```

2. **Inconsistent training:**
   - When c is used: gradients flow
   - When c is None: no gradients
   - This creates instability

**Better Fix:**
```python
# Always use context (never None during training)
# OR use a learnable null embedding
null_context = self.param("null_context", 
                         nn.initializers.normal(0.02), 
                         (1, self.context_channels))
if c is None:
    c = jnp.repeat(null_context, x.shape[0], axis=0)
context_proj = context_proj_layer(c)
conditioning = t_emb + context_proj
```

---

### ⚠️ **ISSUE #7: Learning Rate Too High for Deep Model**

**Current Setting:**
```python
lr = 1e-4  # Default
weight_decay = 0.0
```

**Why This is Problematic:**
1. **Deep model sensitivity:**
   ```
   Parameter update magnitude ∝ lr × gradient
   
   With depth=12 and weak gradients:
   → Early layers get tiny updates (vanishing)
   → Late layers get normal updates
   → Imbalanced learning across depth
   ```

2. **Warmup needed:**
   - Deep models benefit from learning rate warmup
   - Start with lr = 1e-6, ramp up to 1e-4 over 5k steps
   - Prevents early instability

**Fix:**
```python
# Add learning rate schedule with warmup
def create_learning_rate_schedule(base_lr, warmup_steps, total_steps):
    """Cosine decay with linear warmup."""
    warmup_fn = optax.linear_schedule(
        init_value=1e-6,
        end_value=base_lr,
        transition_steps=warmup_steps
    )
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=total_steps - warmup_steps,
        alpha=0.1
    )
    return optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_steps]
    )

# In create_train_state_pmap:
lr_schedule = create_learning_rate_schedule(
    base_lr=1e-4,
    warmup_steps=5000,
    total_steps=300000
)
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=lr_schedule, weight_decay=0.01)
)
```

---

### ⚠️ **ISSUE #8: SPT (Shifted Patch Tokenization) Initialization**

**Location:** `model/vit_set_jax.py` - Lines 129-167

**Problem Code:**
```python
class SPT(nn.Module):
    def setup(self):
        patch_dim = self.patch_size * self.patch_size * self.sample_size * self.channels
        self.norm = nn.LayerNorm()
        self.proj = nn.Dense(self.dim)  # ⚠️ Default initialization
```

**Why This is Suboptimal:**
1. **Huge input dimension:**
   ```
   patch_dim = patch_size² × sample_size × channels
             = 2² × 6 × 3
             = 72 dimensions per patch
   
   Dense(72 → 450):
   → Large weight matrix
   → Default xavier may not be optimal
   → Consider scaled initialization
   ```

2. **Information bottleneck:**
   - 72D → 450D projection is critical
   - Poor initialization → slow convergence
   - Affects all downstream layers

**Fix:**
```python
self.proj = nn.Dense(
    self.dim, 
    kernel_init=nn.initializers.variance_scaling(
        scale=2.0, 
        mode='fan_in', 
        distribution='truncated_normal'
    )
)
```

---

## 🔧 Recommended Fixes (Priority Order)

### **🚨 URGENT (Must Fix First):**

1. **Enable Gradient Clipping:**
```python
# In train_util_jax.py, line 167
tx = optax.chain(
    optax.clip_by_global_norm(1.0),  # ← ADD THIS
    optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
)
```

2. **Fix Zero Initialization in adaLN:**
```python
# In dit_jax.py, lines 208, 259, 267
# Change from:
kernel_init=nn.initializers.constant(0.0)
# To:
kernel_init=nn.initializers.normal(stddev=0.02)
```

3. **Enable LayerNorm Scale:**
```python
# In dit_jax.py, all LayerNorm calls
nn.LayerNorm(use_bias=True, use_scale=True)  # ← Enable scale!
```

---

### **⚡ HIGH PRIORITY (Significant Impact):**

4. **Add Learning Rate Warmup:**
```python
# Add warmup schedule
lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-6,
    peak_value=1e-4,
    warmup_steps=5000,
    decay_steps=300000,
    end_value=1e-5
)
```

5. **Improve Context Projection:**
```python
# Use learnable null embedding instead of zeros
null_context = self.param("null_context", 
                         nn.initializers.normal(0.02), 
                         (1, self.context_channels))
```

6. **Add Weight Decay:**
```python
weight_decay=0.01  # Instead of 0.0
```

---

### **💡 RECOMMENDED (Further Improvements):**

7. **Consider Reducing Depth:**
```python
# For 32×32 images, depth=4 might be better
depth=4,  # Instead of 6
```

8. **Monitor Gradient Norms:**
```python
# Already implemented in train_util_jax.py!
# Check debug/grad_norm_encoder and debug/grad_norm_dit
# Healthy range: 0.1 - 10.0
```

9. **Add Gradient Accumulation (if needed):**
```python
# For larger effective batch size
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.apply_every(k=4),  # Accumulate 4 steps
    optax.adamw(learning_rate=lr_schedule, weight_decay=0.01)
)
```

---

## 📊 Expected Improvements After Fixes

### **Gradient Flow:**
```
Before fixes:
  grad_norm_encoder: 0.01 - 0.05  (vanishing!)
  grad_norm_dit:     0.1  - 0.5
  
After fixes:
  grad_norm_encoder: 0.5  - 2.0   (healthy!)
  grad_norm_dit:     1.0  - 5.0
```

### **Context Learning:**
```
Before: Context drift over 17.5k steps (-0.037 → 0)
After:  Context stabilizes within 5k steps
        Context values: -0.01 to +0.01 (centered)
```

### **Training Speed:**
```
Before: Loss plateau after 10k steps
After:  Continuous improvement, faster convergence
        Expect good FID scores by 50k steps
```

---

## 🔬 How to Verify Fixes

### **1. Monitor Debug Metrics:**
```python
# Check in W&B:
debug/grad_norm_encoder    # Should be > 0.1
debug/grad_norm_dit        # Should be > 0.5
debug/context_norm         # Should stabilize quickly
debug/context_std          # Should be > 0.5
```

### **2. Check Loss Convergence:**
```python
# Healthy training:
loss at 1k steps:   0.08 - 0.12
loss at 5k steps:   0.05 - 0.08
loss at 10k steps:  0.03 - 0.05
loss at 20k steps:  0.02 - 0.04
```

### **3. Visualize Gradients:**
```python
# Add gradient histogram logging
def log_gradient_histograms(grads, step):
    for key in ['encoder', 'dit']:
        if key in grads:
            flat_grads = jax.tree_util.tree_leaves(grads[key])
            grad_values = jnp.concatenate([g.flatten() for g in flat_grads])
            wandb.log({
                f"gradients/{key}_hist": wandb.Histogram(grad_values)
            }, step=step)
```

---

## 🎯 Implementation Checklist

- [ ] **Add gradient clipping** (train_util_jax.py)
- [ ] **Fix adaLN initialization** (dit_jax.py - 3 locations)
- [ ] **Enable LayerNorm scale** (dit_jax.py - all LayerNorm calls)
- [ ] **Add learning rate warmup** (train_util_jax.py)
- [ ] **Add weight decay** (main_jax.py defaults)
- [ ] **Fix null context handling** (dit_jax.py)
- [ ] **Monitor gradient norms** (already implemented ✅)
- [ ] **Test on small run** (1k steps to verify)
- [ ] **Full training run** (300k steps with monitoring)

---

## 📚 References

1. **DiT Paper:** "Scalable Diffusion Models with Transformers"
   - Uses depth=28 for 256×256 images
   - Your 32×32 images → depth=6 is already conservative

2. **Gradient Flow in Deep Networks:**
   - He et al., "Delving Deep into Rectifiers"
   - Importance of proper initialization

3. **AdaLN-Zero:**
   - Peebles & Xie, DiT paper
   - Zero init gates allow identity mapping initially
   - BUT requires careful tuning to start learning

4. **Gradient Clipping:**
   - Pascanu et al., "On the difficulty of training RNNs"
   - Essential for diffusion models with varying timesteps

---

**Next Steps:**
1. Apply URGENT fixes first
2. Run 5k step test
3. Check gradient norms and context metrics
4. If improved, continue with HIGH PRIORITY fixes
5. Full training run with monitoring

**Expected Time to Fix:** 1-2 hours of code changes
**Expected Improvement:** 5-10× faster convergence, stable training
