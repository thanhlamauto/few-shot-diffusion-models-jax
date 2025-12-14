# Sample Size Logic Explanation

## Khi bạn set `--sample_size 5`:

### 1. Dataset Level
**File**: `dataset/base.py:49`
```python
self.sample_size = sample_size + 1  # 5 + 1 = 6
```

**Kết quả**: Dataset tạo sets với **6 images** mỗi set
- Mỗi set có `ns = 6` images
- Log sẽ hiển thị: "Set 0 has 6 images"

### 2. Model Config Level
**File**: `model/select_model_jax.py:19`
```python
cfg.sample_size = args.sample_size  # 5 (không +1)
```

**Kết quả**: Model được init với `cfg.sample_size = 5`
- Encoder (sViT) được build với `sample_size = 5`
- Patch dimension được tính với `sample_size = 5`

### 3. Training Level (Sau fix_set_size)
**File**: `main_jax.py:1062`
```python
batch_np_fixed = fix_set_size(jnp.array(batch_np), cfg.sample_size)
# Crop từ (8, 6, ...) → (8, 5, ...)
```

**Kết quả**: Batch được crop về **5 images** trước khi vào train_step
- `ns = 5` trong training
- JAX chỉ compile 1 version với shape `(8, 5, ...)`

### 4. Leave-One-Out Level
**File**: `model/vfsddpm_jax.py:leave_one_out_c()`
```python
for i in range(ns):  # ns = 5
    idx = [k for k in range(ns) if k != i]  # [0,1,2,3] hoặc [0,1,2,4], etc.
    x_subset = batch_set[:, idx]  # (b, 4, C, H, W)
```

**Kết quả**: Support set có **4 images** (5 - 1 = 4)
- Mỗi ảnh được generate với support set gồm 4 ảnh còn lại
- Đủ để học representation tốt ✅

## Tóm tắt

| Level | Value | Giải thích |
|-------|-------|------------|
| **CLI arg** | `--sample_size 5` | User input |
| **Dataset** | `ns = 6` | `sample_size + 1` (design choice) |
| **Model config** | `cfg.sample_size = 5` | Model expect 5 images |
| **After fix_set_size** | `ns = 5` | Crop từ 6 → 5 |
| **In training** | `ns = 5` | Shape cố định cho JIT |
| **Support set** | `ns - 1 = 4` | Leave-one-out: 4 images |

## Tại sao dataset tạo `sample_size + 1`?

Có thể là design choice để:
- Có thêm 1 ảnh dự phòng
- Hoặc để match với một số dataset khác
- Hoặc là bug/legacy code

**Quan trọng**: `fix_set_size()` đảm bảo batch luôn có đúng `cfg.sample_size` trước khi vào JIT, nên không có vấn đề.

## Kết luận

Với `--sample_size 5`:
- **Dataset tạo**: 6 images/set
- **Training dùng**: 5 images/set (sau crop)
- **Support set**: 4 images (đủ để học representation)
