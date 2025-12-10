# 📊 W&B Metrics - Công thức và Giải thích

Tài liệu này mô tả chi tiết các metrics được log lên Weights & Biases trong quá trình training VFSDDPM-JAX.

---

## 🎯 Core Training Metrics

### 1. **loss** (Tổng Loss)
**Vị trí:** `model/vfsddpm_jax.py` (dòng 331-335)

**Công thức:**

**Chế độ Deterministic:**
```
loss = mean_flat(diffusion_loss).mean()
```

**Chế độ Variational:**
```
loss = mean_flat(diffusion_loss).mean() + KL_divergence
```

**Ý nghĩa:**
- Metric chính để đo training progress
- Giảm → model đang học tốt
- Bao gồm reconstruction loss và regularization (nếu variational)

**Code:**
```python
total = mean_flat(losses["loss"]).mean()
if klc is not None:
    total = total + klc
losses["loss"] = total
```

---

### 2. **mse** (Mean Squared Error)
**Vị trí:** Được tính bởi `GaussianDiffusion.training_losses()`

**Công thức:**
```
mse = (1 / (B × C × H × W)) × Σ(ε_θ(x_t, t, c) - ε)²
```

Trong đó:
- `x_t` = noisy image tại timestep t
- `t` = random timestep ∈ [0, T]
- `c` = conditioning vector từ leave-one-out encoding
- `ε_θ` = predicted noise bởi DiT model
- `ε ~ N(0, I)` = true noise đã được thêm vào
- `B` = batch size, `C` = channels, `H` = height, `W` = width

**Ý nghĩa:**
- Đo độ chính xác của noise prediction
- Giảm → model dự đoán noise tốt hơn → sample quality tốt hơn

**Quá trình tính:**
1. Sample timestep: `t ~ Uniform(0, T)`
2. Add noise: `x_t = √(ᾱ_t) × x_0 + √(1-ᾱ_t) × ε`
3. Predict: `ε_pred = model(x_t, t, c)`
4. Compute MSE: `mse = mean((ε_pred - ε)²)`

---

### 3. **context** (Conditioning Vector)
**Vị trí:** `model/vfsddpm_jax.py` - `leave_one_out_c()` (dòng 250-294)

**Công thức:**

**Deterministic Mode:**
```
c_i = Encoder({x_1, ..., x_n} \ {x_i})
```

**Variational Mode:**
```
h = Encoder({x_1, ..., x_n} \ {x_i})
μ, log(σ²) = Posterior(h)
c_i = μ + σ × ε,  where ε ~ N(0, 1)
```

**Chi tiết:**
- **Input:** Set of images `{x_1, ..., x_n}` (thường n=6)
- **Leave-one-out:** Để predict x_i, dùng {x_1,...,x_n}\{x_i} làm context
- **Encoder:** ViT hoặc sViT (Set Transformer)
- **Output shape:** 
  - FiLM mode: `(B×n, hdim)`
  - LAG mode: `(B×n, 1, hdim)` (1 token)

**Code:**
```python
for i in range(ns):
    idx = [k for k in range(ns) if k != i]
    x_subset = batch_set[:, idx]  # (b, ns-1, C, H, W)
    hc = encode_set(params["encoder"], enc, x_subset, cfg, train=train)
    c_vec, klc = sample_context(rngs[i], hc, cfg, posterior, params_post)
    c_list.append(c_vec[:, None, ...])
```

---

### 4. **klc** (KL Divergence - Variational Mode Only)
**Vị trí:** `model/vfsddpm_jax.py` - `gaussian_kl()` (dòng 108-119)

**Công thức:**
```
KL(q(z|x) || p(z)) = 0.5 × Σ[
    σ²_q / σ²_p 
    + (μ_q - μ_p)² / σ²_p 
    - 1 
    + log(σ²_p / σ²_q)
]
```

**Với prior p = N(0, I):**
```
KL(q || p) = 0.5 × Σ[σ²_q + μ²_q - 1 - log(σ²_q)]
```

**Chuyển sang bits:**
```
klc = mean(KL) / log(2)
```

**Ý nghĩa:**
- Regularization term cho variational posterior
- Đo "khoảng cách" giữa learned distribution và prior
- Quá cao → posterior collapse (ko học được gì)
- Quá thấp → under-regularized

**Code:**
```python
def gaussian_kl(qm: Array, qlogvar: Array, pm: Array, plogvar: Array) -> Array:
    qv = jnp.exp(qlogvar)
    pv = jnp.exp(plogvar)
    return 0.5 * (
        (qv / pv)
        + ((qm - pm) ** 2) / pv
        - 1.0
        + (plogvar - qlogvar)
    )
```

---

### 5. **eval_loss** (Validation Loss)
**Vị trí:** `main_jax.py` - `eval_loop()` (dòng 140-162)

**Công thức:**
```
eval_loss = (1/N) × Σ vfsddpm_loss(batch_i, train=False)
```

**Ý nghĩa:**
- Đánh giá generalization trên validation set
- Dùng EMA parameters (không phải training params)
- `train=False` → không dropout, batch norm eval mode

**Code:**
```python
params_eval = flax.jax_utils.unreplicate(p_state.ema_params)
for _ in range(num_batches):
    loss_dict = vfsddpm_loss(
        jax.random.PRNGKey(0), params_eval, modules, 
        batch_np, cfg, train=False
    )
    losses.append(np.array(loss_dict["loss"]))
return float(np.mean(losses))
```

---

## 🔬 Debug & Monitoring Metrics

### 6. **debug/context_norm** (L2 Norm)
**Vị trí:** `model/set_diffusion/train_util_jax.py` (dòng 229)

**Công thức:**
```
||c||₂ = √(Σᵢ c²ᵢ)
```

**Ý nghĩa:**
- Đo magnitude tổng thể của context vector
- Quá lớn → potential numerical instability
- Quá nhỏ → context không chứa đủ information

**Code:**
```python
metrics["debug/context_norm"] = jnp.linalg.norm(context)
```

---

### 7. **debug/context_mean** (Mean Absolute Value)
**Công thức:**
```
mean(|c|) = (1/D) × Σᵢ |cᵢ|
```
Trong đó D = dimension của context vector

**Ý nghĩa:**
- Average magnitude của context features
- Useful để detect feature collapse

---

### 8. **debug/context_max** (Max Absolute Value)
**Công thức:**
```
max(|c|) = maxᵢ |cᵢ|
```

**Ý nghĩa:**
- Phát hiện outlier values trong context
- Quá lớn → có feature dominate

---

### 9. **debug/context_std** (Standard Deviation)
**Công thức:**
```
σ_c = √[(1/D) × Σᵢ (cᵢ - μ_c)²]
```

**Ý nghĩa:**
- Đo diversity của context features
- Quá thấp → features uniform (bad)
- Healthy range: 0.1 - 1.0

**Code:**
```python
metrics["debug/context_std"] = jnp.std(context)
```

---

### 10. **debug/grad_norm_encoder** & **debug/grad_norm_dit**
**Vị trí:** `model/set_diffusion/train_util_jax.py` (dòng 235-245)

**Công thức:**
```
||∇θ||₂ = √[Σ_all_layers Σ_all_params (∂L/∂θᵢ)²]
```

**Chi tiết tính toán:**
```python
flat_grads = jax.tree_util.tree_leaves(grad_tree)
grad_norm = sqrt(sum(sum(g²) for g in flat_grads))
```

**Ý nghĩa:**
- Monitor training stability
- Gradient explosion: grad_norm > 100
- Gradient vanishing: grad_norm < 1e-6
- Healthy range: 0.1 - 10.0

**Hành động:**
- Nếu explode → giảm learning rate hoặc thêm gradient clipping
- Nếu vanish → tăng learning rate hoặc check architecture

---

### 11. **debug/param_norm_encoder** & **debug/param_norm_dit**
**Công thức:**
```
||θ||₂ = √[Σ_all_layers Σ_all_params θ²ᵢ]
```

**Ý nghĩa:**
- Theo dõi weight magnitude qua training
- Tăng liên tục → potential weight drift
- Useful để so sánh với grad_norm:
  ```
  relative_grad = grad_norm / param_norm
  ```

**Code:**
```python
flat_params = jax.tree_util.tree_leaves(param_tree)
param_norm = jnp.sqrt(sum(jnp.sum(p**2) for p in flat_params))
metrics[f"debug/param_norm_{key}"] = param_norm
```

---

## 📈 Evaluation Metrics

### 12. **fid** (Fréchet Inception Distance)
**Vị trí:** `main_jax.py` - `compute_fid_per_class()` (dòng 212-414)

**Công thức:**
```
FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2√(Σ_real Σ_gen))
```

**Chi tiết:**
1. Extract features từ InceptionV3 (pool_3 layer):
   ```
   f_real = InceptionV3(x_real)  # (N, 2048)
   f_gen = InceptionV3(x_gen)    # (N, 2048)
   ```

2. Compute statistics:
   ```
   μ_real = mean(f_real, axis=0)
   μ_gen = mean(f_gen, axis=0)
   Σ_real = cov(f_real)
   Σ_gen = cov(f_gen)
   ```

3. Compute FID:
   ```
   diff = μ_real - μ_gen
   covmean = sqrtm(Σ_real @ Σ_gen)
   FID = diff.T @ diff + trace(Σ_real + Σ_gen - 2×covmean)
   ```

**Ý nghĩa:**
- **Lower is better** (0 = perfect match)
- FID < 10: Excellent quality
- FID 10-30: Good quality
- FID 30-50: Acceptable
- FID > 50: Poor quality

**Lưu ý:**
- Tính per-class (random 1 class mỗi lần eval)
- Default: 1024 samples per class
- Cần ít nhất 600 samples để FID stable

---

## 🎨 Visualization Metrics (W&B Images)

### 13. **train/support_target_set_{i}**
- Hiển thị leave-one-out split trong training
- Target (red border) vs Support (blue border)
- Logged mỗi `log_interval` steps

### 14. **generation/set_{i}**
- Support images (top row)
- Generated samples (bottom row)
- Logged mỗi `save_interval` steps

### 15. **fid_eval/example_{i}**
- Support set (blue border)
- Generated images (green border)
- Real images (red border)
- Logged khi compute FID

---

## 📊 Step-by-step Training Process

### Forward Pass:
```
1. Batch: (B, ns, C, H, W) in [-1, 1]
2. For each image i in set:
   a. support = set \ {image_i}
   b. c_i = Encoder(support)
   c. If variational: c_i = μ + σ×ε
3. Flatten: x = (B×ns, C, H, W)
4. Sample t ~ Uniform(0, T)
5. Add noise: x_t = √ᾱ_t × x + √(1-ᾱ_t) × ε
6. Predict: ε_pred = DiT(x_t, t, c)
7. Loss: MSE(ε_pred, ε) + KL (if variational)
```

### Backward Pass:
```
1. Compute gradients: ∇θ L
2. Update params: θ ← θ - lr × ∇θ
3. Update EMA: θ_ema ← β × θ_ema + (1-β) × θ
4. Log metrics to W&B
```

### Sampling (DDIM):
```
1. Start: x_T ~ N(0, I)
2. For t = T, T-1, ..., 1:
   a. ε_θ = DiT(x_t, t, c)
   b. x̂_0 = (x_t - √(1-ᾱ_t) × ε_θ) / √ᾱ_t
   c. σ_t = η × √((1-α_t)/(1-ᾱ_t)) × √(1-ᾱ_t/ᾱ_{t-1})
   d. x_{t-1} = √ᾱ_{t-1} × x̂_0 + √(1-ᾱ_{t-1}-σ²_t) × ε_θ + σ_t × ε
3. Return: x_0
```

---

## 🎯 Typical Value Ranges

| Metric | Initial | Mid-Training | Well-Trained |
|--------|---------|--------------|--------------|
| **loss** | 0.08-0.15 | 0.03-0.05 | 0.02-0.035 |
| **mse** | 0.08-0.15 | 0.03-0.05 | 0.02-0.035 |
| **klc** | 5-20 bits | 1-5 bits | 0.5-2 bits |
| **eval_loss** | Similar to loss | Track with loss | < loss (good generalization) |
| **context_norm** | 50-150 | 60-120 | 60-80 |
| **context_std** | 0.5-1.2 | 0.6-1.0 | 0.6-0.8 |
| **grad_norm** | 0.5-5.0 | 0.1-2.0 | 0.05-0.5 |
| **fid** | N/A | 40-80 | 10-30 |

---

## ⚠️ Warning Signs

### Training Instability:
- ❌ **Loss spikes:** Sudden jumps in loss → learning rate too high
- ❌ **grad_norm > 100:** Gradient explosion → add gradient clipping
- ❌ **context_norm exploding:** Encoder instability → check normalization
- ❌ **eval_loss >> loss:** Overfitting → need regularization

### Poor Convergence:
- ❌ **Loss plateau early:** Stuck in local minimum → increase model capacity
- ❌ **FID not improving:** Sample quality issue → check conditioning
- ❌ **klc → 0:** Posterior collapse → adjust KL weight

### Debugging Tips:
1. **Compare grad_norm vs param_norm:**
   ```python
   relative_grad = grad_norm / param_norm
   # Healthy: 1e-4 to 1e-2
   ```

2. **Monitor context statistics:**
   ```python
   # Should have diversity
   context_std > 0.5
   # Should not dominate
   context_max / context_mean < 5
   ```

3. **Check eval_loss vs loss gap:**
   ```python
   # Generalization gap
   gap = eval_loss - loss
   # Healthy: gap < 0.01
   # Overfitting: gap > 0.02
   ```

---

## 📝 Logging Configuration

### Log Intervals:
- **Training metrics:** Every `log_interval` steps (default: 100)
- **Evaluation:** Every `save_interval` steps (default: 20,000)
- **FID:** Same as evaluation (expensive)
- **Checkpoints:** Same as evaluation

### W&B Settings:
```python
wandb.init(
    project="fsdm-jax",
    name=args.wandb_run_name,
    config=vars(args),
)

# Log training metrics
wandb.log({
    "loss": loss,
    "mse": mse,
    "debug/grad_norm_dit": grad_norm,
    # ... other metrics
}, step=global_step)

# Log images
wandb.log({
    "train/support_target_set_0": wandb.Image(fig),
    "generation/set_0": wandb.Image(fig),
}, step=global_step)
```

---

## 🔍 References

### Code Locations:
1. **Main training loop:** `main_jax.py` lines 524-815
2. **Loss computation:** `model/vfsddpm_jax.py` - `vfsddpm_loss()`
3. **Training step:** `model/set_diffusion/train_util_jax.py` - `train_step_pmap()`
4. **FID computation:** `metrics/fid_jax.py` - `compute_fid()`
5. **Sampling:** `model/set_diffusion/gaussian_diffusion_jax.py` - `ddim_sample_loop()`

### Key Papers:
1. **DDPM:** Denoising Diffusion Probabilistic Models (Ho et al., 2020)
2. **DiT:** Scalable Diffusion Models with Transformers (Peebles & Xie, 2023)
3. **FID:** GANs Trained by a Two Time-Scale Update Rule (Heusel et al., 2017)

---

**Generated:** 2025-12-10  
**Model:** VFSDDPM-JAX (DiT backbone)  
**Framework:** JAX/Flax with pmap parallelization
