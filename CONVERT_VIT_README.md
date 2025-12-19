# Hướng dẫn Convert ViT Weights từ PyTorch sang JAX

## Tổng quan

Script `convert_vit_pytorch_to_jax.py` được thiết kế để chuyển đổi weights của Vision Transformer (ViT) từ PyTorch sang định dạng JAX/Flax để sử dụng với model `vit_jax.py`.

## Kiến trúc được hỗ trợ

Script hỗ trợ 2 kiến trúc ViT:

### 1. Timm-style ViT (timm library)
- Cấu trúc: `module.backbone.*`
- Patch embedding: Conv2d thay vì Linear
- Attention: có bias cho qkv projection
- Ví dụ: `vit_cifar10_patch4_input32 (1).pth`

### 2. Custom ViT (từ vit.py trong repo)
- Cấu trúc: `transformer.layers.*`
- Patch embedding: Linear layer
- Attention: không có bias cho qkv (tùy chọn)

## Kiến trúc của checkpoint `vit_cifar10_patch4_input32 (1).pth`

Từ phân tích checkpoint, đây là timm-style ViT với các thông số:

- **Image size**: 32x32 (CIFAR-10)
- **Patch size**: 4x4
- **Embedding dimension (dim)**: 192
- **Depth**: 9 layers
- **Heads**: 3
- **Dim head**: 64
- **MLP ratio**: 2 (384 / 192)

### Cấu trúc parameters:

```
module.backbone.cls_token          : (1, 1, 192)
module.backbone.pos_embed          : (1, 65, 192)  # 1 cls + 64 patches (8x8)
module.backbone.patch_embed.proj   : Conv2d(3, 192, kernel_size=4, stride=4)
module.backbone.blocks.{i}.norm1  : LayerNorm(192)
module.backbone.blocks.{i}.attn.qkv: Linear(192 -> 576) với bias
module.backbone.blocks.{i}.attn.proj: Linear(192 -> 192) với bias
module.backbone.blocks.{i}.norm2  : LayerNorm(192)
module.backbone.blocks.{i}.mlp.fc1: Linear(192 -> 384) với bias
module.backbone.blocks.{i}.mlp.fc2: Linear(384 -> 192) với bias
```

## Cách sử dụng

### 1. Inspect checkpoint (chỉ xem cấu trúc)

```bash
python convert_vit_pytorch_to_jax.py \
    --pytorch_ckpt "vit_cifar10_patch4_input32 (1).pth" \
    --inspect_only
```

### 2. Convert weights

```bash
python convert_vit_pytorch_to_jax.py \
    --pytorch_ckpt "vit_cifar10_patch4_input32 (1).pth" \
    --output "vit_cifar10_patch4_input32_jax.npz"
```

### 3. Load weights trong JAX

```python
from convert_vit_pytorch_to_jax import load_jax_weights
from model.vit_jax import ViT
import jax.numpy as jnp
from jax import random

# Load weights
params = load_jax_weights('vit_cifar10_patch4_input32_jax.npz')

# Khởi tạo model với cùng kiến trúc
model = ViT(
    image_size=(32, 32),
    patch_size=(4, 4),
    num_classes=10,  # CIFAR-10
    dim=192,
    depth=9,
    heads=3,
    mlp_dim=384,  # dim * 2
    dim_head=64,
    pool='cls',
    channels=3,
)

# Sử dụng model
key = random.PRNGKey(0)
dummy_img = jnp.ones((1, 3, 32, 32))
output = model.apply({'params': params['params']}, dummy_img, train=False)
```

## Mapping giữa PyTorch và JAX

### Patch Embedding
- **PyTorch (timm)**: `Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)`
- **JAX**: `Dense(patch_dim, dim)` với `patch_dim = 3 * patch_h * patch_w`
- **Conversion**: Unfold Conv2d weights thành Linear weights

### Attention
- **PyTorch**: `qkv` là một Linear layer `(dim -> inner_dim*3)` với bias
- **JAX**: Tách thành `query`, `key`, `value` riêng biệt, mỗi cái có `kernel` và `bias`
- **Kernel shape**: 
  - PyTorch: `(inner_dim*3, dim)`
  - JAX: `query/key/value.kernel: (dim, heads, dim_head)`
- **Bias shape**:
  - PyTorch: `(inner_dim*3,)`
  - JAX: `query/key/value.bias: (heads, dim_head)`

### Output Projection
- **PyTorch**: `(dim, dim)`
- **JAX**: `out.kernel: (heads, dim_head, dim)`

### LayerNorm
- **PyTorch**: `weight` và `bias` shape `(dim,)`
- **JAX**: `scale` và `bias` shape `(dim,)` (tương tự)

### FeedForward
- **PyTorch**: `fc1: (dim -> mlp_dim)`, `fc2: (mlp_dim -> dim)`
- **JAX**: `Dense_0: (dim -> mlp_dim)`, `Dense_1: (mlp_dim -> dim)`
- **Kernel**: Transpose từ PyTorch format

## Lưu ý

1. **Time embedding**: Nếu checkpoint không có time embedding, script sẽ khởi tạo random weights. Bạn có thể fine-tune sau.

2. **Classification head**: Nếu checkpoint không có classification head (như timm backbone), script sẽ khởi tạo random weights. Bạn cần fine-tune hoặc thay thế bằng head phù hợp.

3. **Parameter structure**: Flax model trong `vit_jax.py` sử dụng cấu trúc nested dưới `params` key. Script tự động wrap parameters vào cấu trúc này.

4. **Verification**: Sau khi convert, nên test với một vài sample để đảm bảo output tương tự (hoặc gần giống) với PyTorch model.

## Troubleshooting

### Lỗi: "Cannot determine ViT architecture"
- Kiểm tra xem checkpoint có đúng format không
- Thử inspect checkpoint trước để xem cấu trúc

### Lỗi: "Cannot infer 'dim' from checkpoint"
- Kiểm tra xem checkpoint có `cls_token` hoặc `pos_embedding` không
- Với timm-style, kiểm tra `module.backbone.cls_token`

### Output khác với PyTorch
- Kiểm tra lại các tham số architecture (dim, depth, heads, etc.)
- Đảm bảo image preprocessing giống nhau (normalization, etc.)
- Kiểm tra xem có layer nào bị thiếu không (time embedding, classification head)

