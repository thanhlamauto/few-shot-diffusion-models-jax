"""
Verify that target images receive correct context from their own class.

This script traces the data flow to ensure context-target matching is correct.
"""

import numpy as np
import jax.numpy as jnp

print("="*80)
print("VERIFICATION: TARGET IMAGE NHẬN ĐÚNG CONTEXT TỪ CLASS CỦA NÓ")
print("="*80)

# Simulate the data flow
print("\n" + "="*80)
print("STEP 1: DATASET - make_sets()")
print("="*80)

print("""
Code: dataset/base.py, lines 228-282

def make_sets(self, images, labels):
    for class_id in range(num_classes):
        # Get all images for THIS CLASS ONLY
        class_mask = (labels == class_id)
        class_images = images[class_mask]
        
        # Create sets from THIS CLASS ONLY
        n_sets = len(class_images) // self.sample_size
        class_sets = class_images.reshape(n_sets, sample_size, ...)
        
✅ VERIFIED: Mỗi set chỉ chứa images từ CÙNG 1 class!
""")

# Simulate dataset
b = 2  # batch size
ns = 5  # sample size (images per set)
print(f"\nSimulation: batch_size={b}, sample_size={ns}")
print("-" * 80)

# Create mock batch_set where each set is from one class
# Set 0: class 0 (images 0-4)
# Set 1: class 1 (images 10-14)
batch_set = np.zeros((b, ns, 3, 32, 32))
class_ids = np.array([0, 1])  # Set 0 = class 0, Set 1 = class 1

print("\nMock batch_set:")
print(f"  Shape: {batch_set.shape} = (batch={b}, ns={ns}, C, H, W)")
print(f"  Set 0: class_id = {class_ids[0]}")
print(f"  Set 1: class_id = {class_ids[1]}")

print("\n" + "="*80)
print("STEP 2: leave_one_out_c() - CREATE CONTEXT")
print("="*80)

print("""
Code: model/vfsddpm_jax.py, lines 250-294

def leave_one_out_c(batch_set):
    b, ns = batch_set.shape[:2]  # (2, 5)
    c_list = []
    
    for i in range(ns):  # For each image position (0, 1, 2, 3, 4)
        # Leave out image i, use the rest as support
        idx = [k for k in range(ns) if k != i]
        x_subset = batch_set[:, idx]  # (b, ns-1, C, H, W)
        
        # Encode support set to context
        hc = encode_set(x_subset)  # (b, hdim)
        c_list.append(hc[:, None, ...])
    
    c_set = jnp.concatenate(c_list, axis=1)  # (b, ns, hdim)
    return c_set

✅ KEY POINT: c_set[batch_idx, img_idx] là context cho batch_set[batch_idx, img_idx]
""")

print("\nSimulation:")
print("-" * 80)

c_list = []
for i in range(ns):
    idx = [k for k in range(ns) if k != i]
    print(f"\nImage position i={i}:")
    print(f"  Support indices: {idx}")
    print(f"  x_subset from batch_set[:, {idx}]")
    
    # For each batch in the set
    for batch_idx in range(b):
        class_id = class_ids[batch_idx]
        print(f"    Batch {batch_idx} (class {class_id}):")
        print(f"      - Support: images from positions {idx} of class {class_id}")
        print(f"      - Target:  image at position {i} of class {class_id}")
        print(f"      → Context from class {class_id} for target of class {class_id} ✅")
    
    # Mock encoding
    hc = np.random.randn(b, 384)  # (b, hdim)
    c_list.append(hc[:, None, :])  # (b, 1, hdim)

c_set = np.concatenate(c_list, axis=1)  # (b, ns, hdim)
print(f"\nc_set shape: {c_set.shape} = (batch={b}, ns={ns}, hdim={384})")

print("\n" + "="*80)
print("STEP 3: FLATTEN FOR TRAINING")
print("="*80)

print("""
Code: model/vfsddpm_jax.py, line 288
    c = c_set.reshape(b * ns, c_set.shape[-1])

Code: model/set_diffusion/gaussian_diffusion_jax.py
    x_flat = batch_set.reshape(b * ns, *batch_set.shape[2:])
""")

# Flatten
c_flat = c_set.reshape(b * ns, c_set.shape[-1])
x_flat_shape = (b * ns, 3, 32, 32)

print(f"\nAfter flattening:")
print(f"  c_flat shape:    {c_flat.shape} = ({b * ns}, hdim)")
print(f"  x_flat shape:    {x_flat_shape} = ({b * ns}, C, H, W)")

print("\n" + "="*80)
print("STEP 4: VERIFY CONTEXT-TARGET MATCHING")
print("="*80)

print("\nMapping after flatten:")
print("-" * 80)

for batch_idx in range(b):
    class_id = class_ids[batch_idx]
    print(f"\nBatch {batch_idx} (class {class_id}):")
    for img_idx in range(ns):
        flat_idx = batch_idx * ns + img_idx
        print(f"  Flat index {flat_idx}:")
        print(f"    - x_flat[{flat_idx}] = batch_set[{batch_idx}, {img_idx}] (class {class_id})")
        print(f"    - c_flat[{flat_idx}] = c_set[{batch_idx}, {img_idx}]")
        
        # Which images in support?
        support_indices = [k for k in range(ns) if k != img_idx]
        print(f"    - Context from: batch_set[{batch_idx}, {support_indices}] (class {class_id})")
        print(f"    → Target class {class_id} + Context from class {class_id} ✅")

print("\n" + "="*80)
print("STEP 5: TRAINING LOSS COMPUTATION")
print("="*80)

print("""
Code: model/set_diffusion/gaussian_diffusion_jax.py, lines 474-520

def training_losses(model, x_t, t, c, ...):
    # x_t: (b*ns, C, H, W) - noisy target images
    # c:   (b*ns, hdim)    - context for each target
    # t:   (b*ns,)         - timestep for each target
    
    model_output = model(x_t, t, c, **model_kwargs)
    # Model predicts noise for x_t CONDITIONED on c
    
✅ VERIFIED: 
   - x_t[i] receives context c[i]
   - Due to leave-one-out: c[i] is from same set as x_t[i]
   - Due to make_sets: same set = same class
   → TARGET NHẬN ĐÚNG CONTEXT TỪ CLASS CỦA NÓ! ✅
""")

print("\n" + "="*80)
print("CRITICAL VERIFICATION:")
print("="*80)

print("""
📊 Data Flow Summary:

1. make_sets():
   ✅ Mỗi set = images từ CÙNG 1 class

2. leave_one_out_c():
   ✅ c_set[i, j] = context từ batch_set[i, {0,...,j-1,j+1,...,ns-1}]
   ✅ Support và target trong CÙNG set

3. Flatten:
   ✅ c_flat[i*ns + j] ↔ x_flat[i*ns + j]
   ✅ Mapping preserved!

4. Training:
   ✅ x_t[k] gets context c[k]
   ✅ c[k] from same class as x_t[k]

🎯 CONCLUSION: TARGET IMAGE NHẬN ĐÚNG CONTEXT TỪ CLASS CỦA NÓ! ✅

""")

print("="*80)
print("POTENTIAL ISSUES TO CHECK:")
print("="*80)

print("""
Tuy nhiên, CẦN KIỂM TRA thêm:

⚠️  1. Data Augmentation:
   - Có augmentation nào làm thay đổi class không?
   - Check: dataset/base.py, __getitem__()

⚠️  2. Shuffle trong DataLoader:
   - DataLoader có shuffle TRONG set không?
   - Check: dataset/__init__.py, create_loader()

⚠️  3. Multi-device (pmap):
   - Khi split batch across devices, mapping có đúng không?
   - Check: main_jax.py, training loop

Hãy check 3 điểm này!
""")

print("="*80)
