# FID Config Mismatch Fix

## 🔴 Problem

### **Error Message:**
```
flax.errors.ScopeParamShapeError: For parameter "scale" in "/to_patch_embedding/norm", 
the given initializer is expected to generate shape (60,), but the existing parameter it 
received has shape (72,).
```

### **Root Cause:**

**SPT (Shifted Patch Tokenization) patch dimension calculation:**
```python
patch_dim = patch_size² × sample_size × channels

Training checkpoint (shape=72):
72 = 2² × 6 × 3  → sample_size = 6 ✅

FID generation (shape=60):
60 = 2² × 5 × 3  → sample_size = 5 ❌
```

**Why the mismatch?**

1. **Training:** Command used `--sample_size 6`
   - Checkpoint saved with `sample_size=6`
   - SPT LayerNorm scale shape: `(72,)`

2. **FID Generation:** Config object has `sample_size=5` (default from `main_jax.py` line 974)
   - But actual data has 6 images per set
   - Encoder expects `sample_size=5` → SPT scale shape: `(60,)`
   - **Mismatch!** Checkpoint has (72,) but model expects (60,)

---

## ✅ Solution

### **Auto-Detection & Config Adjustment**

Added in `compute_fid_per_class()` (main_jax.py):

```python
# CRITICAL FIX: Ensure cfg.sample_size matches actual data
actual_ns = batch_sets.shape[1]  # Get actual number of images per set from data

if cfg.sample_size != actual_ns:
    print(f"⚠️  Config mismatch detected: cfg.sample_size={cfg.sample_size}, but data has {actual_ns} images/set")
    print(f"   Creating temporary config with sample_size={actual_ns} for FID generation...")
    
    # Create a new config with corrected sample_size
    import dataclasses
    cfg_fid = dataclasses.replace(cfg, sample_size=actual_ns)
else:
    cfg_fid = cfg

# Use cfg_fid instead of cfg for all FID operations
c_cond, _ = leave_one_out_c(cond_rng, sub, modules, mini_batch, cfg_fid, train=False)
```

---

## 📊 Flow Diagram

### **Before Fix:**
```
Checkpoint: sample_size=6 (SPT scale=72)
     ↓ load
EMA Params: SPT scale shape = (72,)
     ↓
FID Generation:
     cfg.sample_size = 5 (default)  ❌ MISMATCH!
     data.shape[1] = 6 (actual)
     ↓
Encoder.apply(data, cfg):
     SPT expects shape (60,)  ← from cfg.sample_size=5
     But params have (72,)    ← from checkpoint
     → ERROR!
```

### **After Fix:**
```
Checkpoint: sample_size=6 (SPT scale=72)
     ↓ load
EMA Params: SPT scale shape = (72,)
     ↓
FID Generation:
     cfg.sample_size = 5 (default)
     data.shape[1] = 6 (actual)
     ↓ AUTO-DETECT
     actual_ns = 6
     cfg_fid = replace(cfg, sample_size=6)  ✅ FIXED!
     ↓
Encoder.apply(data, cfg_fid):
     SPT expects shape (72,)  ← from cfg_fid.sample_size=6
     Params have (72,)        ← from checkpoint
     → SUCCESS!
```

---

## 🔍 Why This Happened

### **Config Lifecycle Issue:**

1. **Training Start:**
   ```bash
   python main_jax.py --sample_size 6 ...
   ```
   - `args.sample_size = 6`
   - `cfg = VFSDDPMConfig(sample_size=6)`
   - Model initialized with `sample_size=6`

2. **Checkpoint Save:**
   - Parameters saved with shapes based on `sample_size=6`
   - Config saved separately: `cfg: dataclasses.asdict(cfg)`

3. **Checkpoint Load:**
   ```python
   # In main_jax.py
   host_state = checkpointer.restore(path)
   p_state = p_state.replace(params=host_state["params"], ...)
   # ⚠️ But cfg is NOT restored from checkpoint!
   # cfg still uses defaults or command-line args
   ```

4. **FID Generation:**
   - `cfg` object passed to `compute_fid_per_class()` may have wrong `sample_size`
   - Data has 6 images/set (hardcoded in FID generation)
   - **Mismatch!**

---

## 🛡️ Prevention

### **Option 1: Always Match Data (Current Fix)**
✅ **Implemented:** Auto-detect `sample_size` from data shape
- Robust: Works regardless of cfg value
- Flexible: Handles different set sizes

### **Option 2: Save/Load Config with Checkpoint**
Improve checkpoint saving:
```python
def save_checkpoint(step_int, rng_save):
    ckpt = {
        "params": ...,
        "cfg": dataclasses.asdict(cfg),  # ✅ Already saved
        # ...
    }
    checkpointer.save(...)

def load_checkpoint(path):
    host_state = checkpointer.restore(path)
    
    # Restore cfg from checkpoint
    if "cfg" in host_state:
        cfg_dict = host_state["cfg"]
        cfg = VFSDDPMConfig(**cfg_dict)  # ✅ Restore config
```

### **Option 3: Validate Config at Model Init**
Add validation in `init_models()`:
```python
def init_models(rng: PRNGKey, cfg: VFSDDPMConfig, params=None):
    if params is not None:
        # Validate params match config
        enc_params = params["encoder"]
        # Check SPT scale shape matches expected
        expected_patch_dim = cfg.patch_size ** 2 * cfg.sample_size * cfg.in_channels
        # Assert shape consistency
```

---

## 📝 Related Files Modified

1. **`main_jax.py`:**
   - Line ~315-327: Added config mismatch detection
   - Line ~334: Use `cfg_fid` instead of `cfg`
   - Line ~345, 359, 395-402: Use `actual_ns` instead of hardcoded `6`

---

## 🧪 Testing

### **Test 1: Config Mismatch Detection**
```python
# Scenario: Train with sample_size=6, test with default=5
python main_jax.py --sample_size 6 --max_steps 20000
# At step 20000, FID should print:
# ⚠️  Config mismatch detected: cfg.sample_size=5, but data has 6 images/set
#    Creating temporary config with sample_size=6 for FID generation...
```

### **Test 2: No Mismatch**
```python
# Scenario: Consistent sample_size
python main_jax.py --sample_size 6 --max_steps 20000
# At step 20000, FID should work without warnings
```

---

## ✅ Summary

| Issue | Before | After |
|-------|--------|-------|
| **Config Source** | Defaults or args | Auto-detected from data |
| **sample_size** | May mismatch data | Always matches data shape |
| **SPT shape** | Mismatch (60 vs 72) | Correct (72 vs 72) |
| **Error** | ScopeParamShapeError | ✅ Fixed |

**Key Improvement:** FID generation is now **robust** to config mismatches by auto-detecting actual data shape.
