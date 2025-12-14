# Phân tích các nguyên nhân tràn RAM

## 1. Context Tensor được lưu trong Losses (CRITICAL)

**Vị trí**: `model/vfsddpm_jax.py:634`
```python
losses["context"] = c  # ❌ Lưu tensor lớn trong losses dict
```

**Vấn đề**:
- Trong lag mode, `c` có shape `(b*ns, num_patches, hdim)`
- Với `batch_size=32, sample_size=6`: `(192, 256, 256)` = ~12.5M values = ~50MB
- Tensor này được truyền qua metrics và có thể không được giải phóng ngay
- JAX có thể giữ reference đến tensor này trong computation graph

**Giải pháp**: Chỉ lưu scalar metrics, không lưu tensor:
```python
# Thay vì:
losses["context"] = c

# Nên:
# Chỉ tính metrics từ c, không lưu tensor
losses["debug/context_norm"] = jnp.linalg.norm(c)
# Không lưu c vào losses
```

## 2. Token List Accumulation trong Leave-One-Out Loop

**Vị trí**: `model/vfsddpm_jax.py:530, 549`
```python
token_list.append(tokens)  # Tích lũy tokens trong loop
...
token_set = jnp.stack(token_list, axis=1)  # Stack tất cả tokens
```

**Vấn đề**:
- Mỗi `tokens` có shape `(b, num_patches, hdim)`
- Với `ns=6`: tích lũy 6 tensors, sau đó stack
- Tạo intermediate memory: `(b, ns, num_patches, hdim)` trước khi reshape

**Giải pháp**: Có thể pre-allocate array thay vì append:
```python
# Pre-allocate array
token_set = jnp.zeros((b, ns, num_patches, hdim))
for i in range(ns):
    ...
    token_set = token_set.at[:, i, :, :].set(tokens)
```

## 3. Debug Metrics tính toán trên Context Tensor lớn

**Vị trí**: `model/set_diffusion/train_util_jax.py:252-257`
```python
if "context" in losses and losses["context"] is not None:
    context = losses["context"]  # Lấy tensor lớn
    metrics["debug/context_norm"] = jnp.linalg.norm(context)
    metrics["debug/context_mean"] = jnp.mean(jnp.abs(context))
    metrics["debug/context_max"] = jnp.max(jnp.abs(context))
    metrics["debug/context_std"] = jnp.std(context)
```

**Vấn đề**:
- Tính toán trên tensor lớn `(192, 256, 256)`
- Tạo nhiều intermediate tensors (abs, mean, max, std)
- Có thể tốn thêm memory

**Giải pháp**: Tính metrics trực tiếp trong loss function, không lưu context:
```python
# Trong vfsddpm_loss:
losses["debug/context_norm"] = jnp.linalg.norm(c)
# Không lưu c vào losses
```

## 4. Metrics được chuyển sang NumPy Array

**Vị trí**: `main_jax.py:1022`
```python
metrics_host = jax.tree.map(lambda x: np.array(x).mean(), metrics)
```

**Vấn đề**:
- Chuyển JAX arrays sang NumPy arrays
- Có thể giữ reference đến original arrays
- NumPy arrays không được garbage collect ngay

**Giải pháp**: Chỉ chuyển scalar values:
```python
# Chỉ lấy scalar metrics, không chuyển toàn bộ
metrics_host = {k: float(v) if v.size == 1 else np.array(v).mean() 
                for k, v in metrics.items()}
```

## 5. JIT Compilation Cache

**Vấn đề**:
- JAX JIT compilation cache có thể tích lũy
- Mỗi lần compile tạo cached computation graph
- Cache không được clear tự động

**Giải pháp**: 
- Restart session sau một số steps
- Hoặc clear cache: `jax.clear_backends()` (nếu cần)

## 6. EMA Parameters

**Vị trí**: `model/set_diffusion/train_util_jax.py:235`
```python
new_ema_params = _tree_update_ema(state.ema_params, new_params, rate=ema_rate)
```

**Vấn đề**:
- EMA params tốn thêm 1x model size
- Với model lớn, có thể tốn nhiều memory

**Giải pháp**: 
- Chỉ lưu EMA khi cần (không lưu mỗi step)
- Hoặc dùng in-place update nếu có thể

## 7. Optimizer States (Adam)

**Vấn đề**:
- Adam optimizer lưu 2 states per parameter (momentum, variance)
- Tốn thêm 2x model size
- Với model lớn, có thể tốn nhiều memory

**Giải pháp**:
- Dùng optimizer nhẹ hơn (SGD) nếu có thể
- Hoặc dùng gradient accumulation để giảm batch size

## 8. Gradient Computation

**Vấn đề**:
- Backward pass tốn memory gấp đôi forward pass
- Gradients được lưu cho tất cả parameters
- Với model lớn, có thể tốn nhiều memory

**Giải pháp**:
- Gradient checkpointing (remat)
- Hoặc gradient accumulation

## 9. Leave-One-Out Loop tạo nhiều Intermediate Tensors

**Vị trí**: `model/vfsddpm_jax.py:495-543`
```python
for i in range(ns):  # Chạy ns lần
    x_subset = batch_set[:, idx]  # Tạo subset
    # Pad, encode, etc. - tạo nhiều intermediate tensors
```

**Vấn đề**:
- Mỗi iteration tạo nhiều intermediate tensors
- JAX có thể giữ reference đến các tensors này
- Với `ns=6`, tạo 6x intermediate memory

**Giải pháp**:
- Gradient checkpointing cho encoder forward
- Hoặc clear intermediate tensors sau mỗi iteration

## 10. Memory Fragmentation

**Vấn đề**:
- JAX memory allocator có thể fragment memory
- Sau nhiều allocations/deallocations, memory có thể fragment
- Dẫn đến không thể allocate large tensors dù có free memory

**Giải pháp**:
- Restart session sau một số steps
- Hoặc pre-allocate large tensors

## Tổng kết và Ưu tiên Fix

### Critical (Fix ngay):
1. **Bỏ `losses["context"] = c`** - Lưu tensor lớn không cần thiết
2. **Tính debug metrics trực tiếp** - Không lưu context tensor

### High Priority:
3. **Gradient checkpointing** - Giảm memory trong backward pass
4. **Giảm batch_size/sample_size** - Giải pháp nhanh nhất

### Medium Priority:
5. **Pre-allocate token_set** - Thay vì append + stack
6. **Chỉ chuyển scalar metrics** - Không chuyển toàn bộ arrays

### Low Priority:
7. **EMA optimization** - Chỉ lưu khi cần
8. **Optimizer optimization** - Dùng optimizer nhẹ hơn
