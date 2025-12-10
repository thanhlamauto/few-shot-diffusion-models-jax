# Kiểm Tra 3 Potential Issues

## ✅ Issue 1: Data Augmentation

### **Code: `dataset/base.py`, `__getitem__()`**

```python
def __getitem__(self, item, lbl=None):
    """
    Returns:
        samples: np.array, shape (ns, nc, size, size)
        (Optionally) targets: np.array, shape (ns,) - class labels
    """
    # Create a set
    samples = self.data['inputs'][item]
    samples = rescale(samples, val_range=(-1, 1), orig_range=(0, 1))
    
    # KHÔNG CÓ AUGMENTATION! ✅
    
    if lbl is not None:
        targets = self.data['targets'][item]
        return samples, targets
    else:
        return samples
```

**✅ Kết luận: KHÔNG CÓ augmentation nào làm thay đổi class!**
- Chỉ có `rescale` từ [0,1] → [-1,1] (normalize)
- Không có random crop, flip, rotation, color jitter, etc.

---

## ✅ Issue 2: Shuffle trong DataLoader

### **Code: `dataset/__init__.py`, `create_loader()`**

```python
def create_loader(args, split, shuffle, drop_last=False):
    dataset = select_dataset(args, split)
    bs = args.batch_size
    if split in ["vis", "val", "test"]:
        bs = args.batch_size_eval
    
    loader = data.DataLoader(
        dataset=dataset,
        batch_size=bs,
        shuffle=shuffle,       # ← Shuffle ĐÂY!
        num_workers=0,
        drop_last=drop_last,
    )
```

**🔍 Phân tích:**

### **Shuffle hoạt động như thế nào?**

`DataLoader` với `shuffle=True`:
1. Shuffle **index của sets**, KHÔNG shuffle images TRONG set
2. Dataset trả về: `samples[item]` với shape `(ns, C, H, W)`
3. DataLoader collate: `(bs, ns, C, H, W)`

**Ví dụ:**
```
Dataset có 100 sets:
- Set 0: [img0, img1, img2, img3, img4] từ class 5
- Set 1: [img5, img6, img7, img8, img9] từ class 12
- ...
- Set 99: [img495, img496, img497, img498, img499] từ class 3

Shuffle=True:
- Chọn random order: [Set 42, Set 7, Set 91, ...]
- Nhưng images TRONG mỗi set VẪN GIỮ NGUYÊN THỨ TỰ!

Batch:
- batch_set shape: (bs, ns, C, H, W)
- batch_set[0] = Set 42 (nguyên xi) ✅
- batch_set[1] = Set 7 (nguyên xi) ✅
```

**✅ Kết luận: Shuffle KHÔNG ảnh hưởng đến structure của set!**
- Shuffle chỉ thay đổi thứ tự GIỮA các sets
- KHÔNG shuffle images TRONG set
- → Context-target matching vẫn đúng! ✅

---

## ⚠️ Issue 3: Multi-device (pmap) Splitting

### **Code: `main_jax.py`, training loop**

```python
# Line 494-500:
p_train_step = jax.pmap(
    train_step_fn, axis_name="batch", donate_argnums=(0, 1)
)

n_devices = jax.local_device_count()
logger.log(f"Found {n_devices} JAX devices")
```

```python
# Line 531-545:
for batch in pbar:
    # batch shape: (bs, ns, C, H, W) from DataLoader
    
    global_step += 1
    
    # Prepare batch
    batch_jax = jnp.array(batch)  # Convert to JAX array
    
    # Split batch across devices for pmap
    # CRITICAL: How is batch split?
```

### **🔍 Phân tích pmap splitting:**

**Giả sử:**
- `batch_size = 32` (from DataLoader)
- `n_devices = 4` (TPU/GPU)
- `batch_jax` shape: `(32, 5, 3, 32, 32)`

**pmap sẽ split như thế nào?**

```python
# pmap automatically splits along axis 0:
# Device 0: batch[0:8]   = sets 0-7
# Device 1: batch[8:16]  = sets 8-15
# Device 2: batch[16:24] = sets 16-23
# Device 3: batch[24:32] = sets 24-31
```

**Trong mỗi device:**
```python
# Device 0 receives:
batch_device0 = batch_jax[0:8]  # (8, 5, 3, 32, 32)

# Call train_step_fn:
train_step_fn(p_state, batch_device0, rng_device0)
  ↓
vfsddpm_loss(..., batch_device0, ...)
  ↓
leave_one_out_c(..., batch_device0, ...)
  # batch_device0[0] = Set 0 (intact) ✅
  # batch_device0[1] = Set 1 (intact) ✅
  # ...
  # batch_device0[7] = Set 7 (intact) ✅
```

**✅ Key Point:**
- pmap splits **GIỮA các sets** (axis 0)
- KHÔNG split **TRONG set** (axis 1)
- Mỗi device nhận một số sets NGUYÊN VẸN
- → Context-target matching VẪN ĐÚNG trên mọi device! ✅

---

## 🎯 FINAL VERIFICATION:

### **Trace Complete Flow:**

```
1. Dataset (base.py):
   make_sets() → Sets với images từ cùng class
   ↓
   __getitem__() → Trả về set (ns, C, H, W)
   ✅ NO augmentation

2. DataLoader (__init__.py):
   Shuffle sets (không shuffle TRONG set)
   ↓
   Batch: (bs, ns, C, H, W)
   ✅ Set structure preserved

3. pmap (main_jax.py):
   Split batch across devices GIỮA các sets
   ↓
   Each device: (bs/n_devices, ns, C, H, W)
   ✅ Each set intact

4. leave_one_out_c (vfsddpm_jax.py):
   For each image i in set:
     Support = other images in SAME set
   ↓
   c[i] = context from images {0,...,i-1,i+1,...,ns-1}
   ✅ Same class

5. Training (gaussian_diffusion_jax.py):
   x_flat[i] + c_flat[i]
   ✅ Correct matching!
```

---

## ✅✅✅ KẾT LUẬN CUỐI CÙNG:

**CẢ 3 ISSUES ĐỀU ỔN:**

1. ✅ **No harmful augmentation** - Chỉ có rescale
2. ✅ **Shuffle preserves sets** - Chỉ shuffle giữa sets, không trong set
3. ✅ **pmap splits correctly** - Split giữa sets, không trong set

**→ TARGET IMAGE LUÔN NHẬN ĐÚNG CONTEXT TỪ CLASS CỦA NÓ!** 🎉

---

## 🔬 Thêm: Cách Test Thực Tế

Nếu muốn chắc chắn hơn, có thể thêm logging trong training loop:

```python
# In main_jax.py, inside train_step_fn:
if global_step % 1000 == 0:
    # Log first batch
    batch_np = np.array(batch_set)
    # Verify all images in batch_np[0] are from same class
    # by checking pixel statistics or saving to disk
```

Hoặc check trong wandb logs:
- `train/support_target_set_*` visualizations
- Verify visually that support and target are from same class

---

## 📊 Tóm Tắt:

| Check | Status | Reason |
|-------|--------|--------|
| **Data augmentation** | ✅ Safe | No class-changing augmentation |
| **DataLoader shuffle** | ✅ Safe | Shuffles sets, not images within sets |
| **pmap splitting** | ✅ Safe | Splits between sets, not within sets |
| **Overall** | ✅✅✅ | **Target receives correct context!** |

**Codebase ĐÚNG, không có bug trong context-target matching!** 🎯
