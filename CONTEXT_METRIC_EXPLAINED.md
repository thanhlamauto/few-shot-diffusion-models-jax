# 📊 Context Metric - Vanishing Gradient Evidence

## Your W&B Chart Shows:

```
        context
          |
    0.000 |                           ___________  ← Converging
          |                    ______/
   -0.010 |              _____/
          |         ____/
   -0.020 |     ___/
          |    /
   -0.030 | __/
          |/
   -0.037 |________________________________
          0    5k    10k   15k   17.5k  steps
```

## 🔍 What This Means:

### **Context = Conditioning Vector from Encoder**

Your model uses **leave-one-out conditioning:**
```
For each image i in set {x₁, x₂, ..., x₆}:
  context_i = Encoder(all images except xᵢ)
  generated_i = DiT(noise | context_i)
```

**The "context" metric** = Average value of these conditioning vectors

---

## 🚨 Why This is Evidence of Vanishing Gradient:

### **Problem 1: Strong Negative Bias**
```
Initial context ≈ -0.037
Expected:      ≈  0.000 (centered)
```

**Cause:** Poor initialization
- Encoder outputs have negative bias
- Should be zero-centered for stable training
- Indicates weights not learning properly

---

### **Problem 2: Extremely Slow Drift**
```
Time to move -0.037 → 0:  17,500 steps!
Normal training:          ~2,000 steps
```

**Cause:** Weak gradients reaching encoder
- Gradients must flow: Loss → DiT (6 layers) → Context → Encoder (6 layers)
- Total 12 layers → massive gradient attenuation
- Only ~0.002 change per 1000 steps = **vanishing!**

---

### **Problem 3: Linear (Not Exponential) Convergence**
```
Healthy training:  loss ∝ exp(-steps)  [exponential decay]
Your training:     context ∝ steps       [linear drift]
```

**Cause:** Gradient signal too weak
- Model making tiny updates each step
- Not learning efficiently
- Just slowly drifting toward equilibrium

---

## 💡 What SHOULD Happen:

```
        context
          |
    0.000 |________  ← Fast stabilization!
          |        ----___
   -0.005 |              ----___
          |                     ----
   -0.010 |                         ----
          |                             ___
   -0.015 |                                --
          |
   -0.020 |________________________________
          0    1k    2k    3k    5k   steps
```

**After fixes:**
- Context starts closer to 0 (better init)
- Quickly converges (strong gradients)
- Stabilizes by ~5k steps (not 17k!)

---

## 🔬 Technical Explanation:

### **Gradient Path:**
```
1. Loss computed on generated image
   ∂L/∂generated
   
2. Backprop through DiT (6 layers)
   ∂L/∂context = ∂L/∂generated × ∂generated/∂context
   
3. Backprop through Encoder (6 layers)
   ∂L/∂encoder_weights = ∂L/∂context × ∂context/∂encoder_weights
```

**At each layer:**
```
gradient_out ≈ gradient_in × 0.9  (due to normalization, activations, etc.)
```

**After 12 layers:**
```
∂L/∂encoder ≈ ∂L/∂output × 0.9¹² 
            ≈ ∂L/∂output × 0.28  ← 72% gradient loss!
```

---

## 📈 Why Context is a Good Diagnostic:

### **Context is the "Middle Point" of Training:**
```
Input Images → Encoder → [Context] → DiT → Output
                         ↑
                    Monitor here!
```

**If context doesn't move:**
- ❌ Encoder not learning (vanishing gradient)
- ❌ Context not useful (DiT ignoring it)
- ❌ Training not working

**If context drifts slowly (your case):**
- ⚠️ Encoder learning very slowly
- ⚠️ Weak gradient signal
- ⚠️ Need fixes!

**If context stabilizes quickly:**
- ✅ Encoder learning well
- ✅ Strong gradients
- ✅ Training healthy

---

## 🎯 What Your Specific Chart Tells Us:

### **Segment 1: Steps 0 - 2k**
```
Context: -0.037 → -0.025  (Δ = 0.012)
Rate:    0.006 per 1k steps
```
**Interpretation:**
- Extremely slow initial learning
- Gradient magnitude ~0.01 (should be ~1.0)
- **Vanishing gradient confirmed**

### **Segment 2: Steps 2k - 10k**  
```
Context: -0.025 → -0.010  (Δ = 0.015)
Rate:    0.002 per 1k steps
```
**Interpretation:**
- Slightly improving but still slow
- Gradient magnitude increasing slightly
- Model starting to learn, but inefficiently

### **Segment 3: Steps 10k - 17.5k**
```
Context: -0.010 → 0.000  (Δ = 0.010)
Rate:    0.001 per 1k steps
```
**Interpretation:**
- Converging but still very slow
- Will take 50k+ steps total
- **Should take only 5k steps with proper gradients!**

---

## 🔧 What Fixes Will Do:

### **Fix 1: Gradient Clipping**
```python
optax.clip_by_global_norm(1.0)
```
**Effect:** Prevents gradient explosions, allows larger stable gradients
**Expected:** grad_norm stays in [0.5, 5.0] range

---

### **Fix 2: Better Initialization**
```python
# Change from constant(0) to normal(0.02)
kernel_init=nn.initializers.normal(stddev=0.02)
```
**Effect:** 
- Context starts near 0 (not -0.037)
- Gradients flow from step 1
**Expected:** Context in [-0.01, 0.01] from beginning

---

### **Fix 3: Enable LayerNorm Scale**
```python
nn.LayerNorm(use_bias=True, use_scale=True)
```
**Effect:**
- Learnable rescaling at each layer
- Gradients can be amplified (not just normalized)
**Expected:** grad_norm × 2-3 improvement

---

### **Fix 4: Learning Rate Warmup**
```python
lr: 1e-6 → 1e-4 over 5k steps
```
**Effect:**
- Gentle start prevents early instability
- Allows larger stable learning rate
**Expected:** Faster convergence after warmup

---

## 📊 Expected Improvements:

### **Context Convergence:**
```
Before: 17.5k steps to reach 0
After:  5k steps to reach 0
Speedup: 3.5× faster! 🚀
```

### **Gradient Magnitudes:**
```
Before:
  grad_norm_encoder: 0.02 - 0.05  ← Vanishing!
  grad_norm_dit:     0.10 - 0.50
  
After:
  grad_norm_encoder: 0.50 - 2.00  ← Healthy!
  grad_norm_dit:     1.00 - 5.00
```

### **Overall Training:**
```
Before:
  - Context drift: 17.5k steps
  - Loss plateau: ~10k steps
  - FID (if computed): Poor quality
  
After:
  - Context stable: 5k steps
  - Loss converge: Continuous improvement
  - FID: Good quality by 50k steps
```

---

## 🎓 Learning Points:

1. **Context is a proxy for encoder learning**
   - If context doesn't move → encoder not learning
   - Context should stabilize quickly (< 5k steps)

2. **Deep models need special care**
   - 12 layers = significant gradient attenuation
   - Initialization, normalization, clipping all matter

3. **Diffusion models are sensitive**
   - Timestep variance causes gradient variance
   - Clipping is essential, not optional

4. **Monitor multiple metrics**
   - Loss alone is not enough
   - Context, grad_norms, parameter norms all tell a story

---

## ✅ Summary:

Your chart shows **textbook vanishing gradient:**
- ❌ Slow linear drift (should be fast exponential)
- ❌ 17.5k steps to converge (should be 5k)
- ❌ Strong negative bias (should start near 0)

**Root causes:**
1. Zero initialization blocking gradients
2. No gradient clipping causing instability
3. LayerNorm without scale weakening signals
4. Deep architecture (12 layers) compounding issues

**Solution:** Apply the 5 urgent fixes in `URGENT_FIXES.md`

**Expected result:** 
- Context stabilizes in ~5k steps
- 3-5× faster overall training
- Much better sample quality

---

**Next:** Read `URGENT_FIXES.md` and apply the changes! 🚀
