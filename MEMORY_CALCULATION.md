# Tính toán Memory Usage cho Lag Mode

## Setting hiện tại (train_lag_mode.sh)
- `batch_size=32`
- `sample_size=6`
- `hidden_size=468`
- `depth=6`
- `encoder_depth=6`
- `hdim=256`
- `mode_conditioning=lag`

## Memory Breakdown

### 1. Model Parameters
- DiT: ~15M params * 4 bytes = ~60MB
- Encoder: ~10M params * 4 bytes = ~40MB
- **Total params: ~100MB**

### 2. Optimizer States (Adam)
- Momentum: 1x params = ~100MB
- Variance: 1x params = ~100MB
- **Total optimizer: ~200MB**

### 3. EMA Parameters
- 1x params = ~100MB

### 4. Context Tensor (Lag Mode) - CRITICAL
- Shape: `(batch_size * sample_size, num_patches, hdim)`
- Với setting hiện tại: `(32*6, 256, 256)` = `(192, 256, 256)`
- Memory: `192 * 256 * 256 * 4 bytes = 50.3 MB` (chỉ context tensor)
- **Nhưng trong computation graph, có thể tốn 10-50x hơn do intermediate tensors**

### 5. Forward Pass Activations

#### Encoder (sViT) - Leave-One-Out Loop
- Chạy `sample_size` lần (6 lần)
- Mỗi lần:
  - Input: `(batch_size, sample_size-1, 3, 32, 32)` = `(32, 5, 3, 32, 32)` = ~3MB
  - Patch tokens: `(32, 256, 256)` = ~8MB
  - Transformer activations: ~50-100MB per layer
  - 6 layers * 6 iterations = **~3-6GB**

#### DiT Forward
- Input: `(batch_size*sample_size, 3, 32, 32)` = `(192, 3, 32, 32)` = ~0.7MB
- Patch embeddings: `(192, 256, 468)` = ~90MB
- Transformer activations: ~200-500MB per layer
- 6 layers = **~1.2-3GB**
- Cross-attention với context: `(192, 256, 256)` = ~50MB per layer
- 6 layers = **~300MB**

### 6. Backward Pass (Gradients)
- Gradients: ~2x forward activations
- **~8-18GB**

### 7. Intermediate Tensors
- Token list accumulation
- Padding operations
- Reshape operations
- **~2-5GB**

## Tổng Memory Estimate

### Conservative Estimate:
- Model + Optimizer + EMA: ~400MB
- Forward activations: ~5-10GB
- Backward gradients: ~10-20GB
- Intermediate tensors: ~2-5GB
- Context + cross-attention: ~1-2GB
- **Total: ~18-37GB per batch**

### Với JAX overhead và fragmentation:
- **Actual usage: ~50-100GB per batch**
- Với 330GB RAM, chỉ có thể fit **3-6 batches** trong memory cùng lúc

## Giải pháp: Ultra Low Memory Config

### train_lag_mode_ultra_low_mem.sh
- `batch_size=4` (giảm 8x)
- `sample_size=3` (giảm 2x)

### Memory với config mới:

#### Context Tensor:
- `(4*3, 256, 256)` = `(12, 256, 256)` = **~3MB** (giảm 94%)

#### Encoder Loops:
- 3 lần thay vì 6 lần (giảm 50%)
- **~1.5-3GB** thay vì 3-6GB

#### DiT Forward:
- Input: `(12, 3, 32, 32)` = ~0.05MB
- Activations: **~300-800MB** thay vì 1.2-3GB

#### Backward:
- **~2-3GB** thay vì 8-18GB

### Tổng với config mới:
- **~5-10GB per batch** (giảm 80-90%)
- Với 330GB RAM, có thể fit **30-60 batches** trong memory

## So sánh Memory

| Config | batch_size | sample_size | Context Memory | Est. Total/Batch | Safe for 330GB? |
|--------|-----------|-------------|----------------|------------------|-----------------|
| Original | 32 | 6 | 50MB | 50-100GB | ❌ No |
| Low mem | 8 | 4 | 8MB | 15-30GB | ⚠️ Maybe |
| **Ultra low** | **4** | **3** | **3MB** | **5-10GB** | ✅ **Yes** |

## Khuyến nghị

1. **Bắt đầu với ultra low mem config** (`batch_size=4, sample_size=3`)
2. **Monitor memory usage** trong vài steps đầu
3. **Nếu còn dư memory** (>50GB free), có thể tăng dần:
   - `batch_size=6, sample_size=4`
   - `batch_size=8, sample_size=4`
4. **Nếu vẫn OOM**, giảm thêm:
   - `batch_size=2, sample_size=3`
   - Hoặc giảm model size (`hidden_size`, `depth`)

## Lưu ý

- JIT compilation (bước đầu) tốn thêm memory tạm thời
- Gradient checkpointing có thể giảm thêm 30-50% memory
- Memory fragmentation có thể làm giảm usable memory
- Nên restart session sau mỗi 10-20K steps để clear cache
