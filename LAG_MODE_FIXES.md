# Critical Fixes for Lag Mode (Cross-Attention with Patch Tokens)

## Issues Fixed

### 1. ✅ SPT Stacking + Leave-One-Out Mismatch

**Problem**: When leave-one-out creates `x_subset` with shape `(b, ns-1, C, H, W)`, but sViT's SPT expects `sample_size` for `patch_dim` calculation.

**Fix**: Pad `x_subset` back to `sample_size` before encoding:
```python
if cfg.encoder_mode == "vit_set" and x_subset.shape[1] < cfg.sample_size:
    pad_size = cfg.sample_size - x_subset.shape[1]
    pad_images = jnp.zeros((b, pad_size, *x_subset.shape[2:]), dtype=x_subset.dtype)
    x_subset = jnp.concatenate([x_subset, pad_images], axis=1)
    # Also pad t_emb accordingly
```

**Location**: `model/vfsddpm_jax.py:leave_one_out_c()`

### 2. ✅ Token Indexing (CLS + TIME offset)

**Problem**: Need to correctly skip CLS and TIME tokens when extracting patch tokens.

**Fix**: 
- **sViT**: Always has TIME token (even if `t_emb=None`, creates zero token) → offset = 2
- **ViT**: Always has TIME token → offset = 2
- Both use `x_set_tokens[:, 2:, :]` to extract patch tokens

**Location**: `model/vfsddpm_jax.py:encode_set()`

### 3. ✅ Posterior Does NOT Overwrite Tokens

**Problem**: Ensure `sample_context()` (variational) only affects KL loss, not conditioning tokens.

**Fix**: 
- `sample_context()` only operates on pooled `hc` → returns `c_vec` for KL
- Tokens are collected separately and used directly for cross-attention
- `c_vec` is only appended to `c_list` for logging/debug, NOT used in lag mode

**Location**: `model/vfsddpm_jax.py:leave_one_out_c()`

### 4. ✅ Added Comprehensive Asserts

**Asserts in `encode_set()`**:
1. `tokens.ndim == 3`
2. `tokens.shape[-1] == cfg.hdim`
3. `tokens.shape[1] == (image_size // patch_size) ** 2`

**Asserts in `leave_one_out_c()`**:
4. `token_set.shape[0] == b`
5. `token_set.shape[1] == ns`
6. `token_set.shape[3] == cfg.hdim`
7. `c.shape == (b*ns, num_patches, cfg.hdim)`

**Asserts in `vfsddpm_loss()`**:
8. `x_in.shape[0] == c.shape[0]` (batch match)
9. `c.ndim == 3` for lag mode

**Location**: Multiple files, see code

### 5. ✅ Actual ns Detection in forward_set()

**Problem**: `forward_set()` was using `self.ns` (config value) instead of actual input size.

**Fix**: Calculate `actual_ns` from `img.shape[1]`:
```python
if img.ndim == 5:
    actual_ns = img.shape[1]  # (b, ns, c, h, w)
else:
    actual_ns = self.ns  # fallback
```

**Location**: `model/vit_set_jax.py:forward_set()`

## Shape Flow Verification

### Leave-One-Out Flow:
```
batch_set: (b, ns=6, C, H, W)
  ↓ leave-one-out
x_subset: (b, ns-1=5, C, H, W)
  ↓ pad (if sViT)
x_subset: (b, sample_size=6, C, H, W)  # padded
  ↓ encode_set(return_tokens=True)
hc: (b, hdim)
tokens: (b, num_patches=256, hdim)
  ↓ collect from ns iterations
token_list: [tokens_0, tokens_1, ..., tokens_5]  # length=6
  ↓ stack
token_set: (b, ns=6, num_patches=256, hdim)
  ↓ reshape
c: (b*ns=6, num_patches=256, hdim)  # Final conditioning
```

### DiT Cross-Attention:
```
x: (b*ns, C, H, W)  # Flattened images
c: (b*ns, num_patches, hdim)  # Patch tokens
  ↓ DiTBlock cross-attention
context_proj = Dense(hidden_size)(c)  # (b*ns, num_patches, hidden_size)
cross_attn(query=x_norm, key=context_proj, value=context_proj)
```

## Testing Checklist

- [ ] Test with `mode_conditioning="lag"` and `encoder_mode="vit_set"`
- [ ] Test with `mode_conditioning="lag"` and `encoder_mode="vit"`
- [ ] Verify no shape mismatches during training
- [ ] Check that tokens are not overwritten by posterior
- [ ] Verify cross-attention receives correct token format
- [ ] Test with variational mode (`mode_context="variational"`)
- [ ] Test with different `sample_size` values
- [ ] Monitor `context_norm` metrics (should not collapse to 0)

## Known Limitations

1. **Padding overhead**: When leave-one-out, we pad subset back to `sample_size`. This creates "dummy" images that don't contribute to encoding but are needed for SPT shape consistency.

2. **Memory**: Lag mode uses more memory due to cross-attention over `num_patches` tokens (e.g., 256 tokens × hdim).

3. **Context collapse**: If encoder output collapses, cross-attention becomes ineffective. Monitor `context_norm` metrics.

## Recommendations

1. **Weight decay**: Avoid heavy weight decay on encoder output head
2. **KL annealing**: If variational, use KL warmup/annealing
3. **Cross-attention gate**: Consider adding gate to cross-attention if training is unstable:
   ```python
   x = x + gate * cross_attn_x  # instead of x = x + cross_attn_x
   ```
4. **Dropout RNG**: Ensure proper RNG keys for dropout when `train=True`
