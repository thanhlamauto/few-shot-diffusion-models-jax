---
name: Add Enhanced Grad Metrics and Context Debug Metrics
overview: Thêm raw/scaled grad norms, c_norm/t_emb ratio sau layer norm, và các debug metrics khác để log lên wandb
todos:
  - id: add-raw-scaled-grad-norms
    content: Add raw grad norms (before LR scaling) and scaled grad norms (after LR scaling) with clear naming in train_util_jax.py
    status: pending
  - id: add-effective-step-size
    content: Add effective step size metrics (grad_norm * lr) for encoder and DiT
    status: pending
    dependencies:
      - add-raw-scaled-grad-norms
  - id: add-c-norm-t-ratio
    content: Add c_norm/t_emb ratio metrics after layer norm in vfsddpm_jax.py
    status: pending
  - id: add-debug-metrics
    content: Add additional debug metrics (context norm std, LR scale info, etc.)
    status: pending
    dependencies:
      - add-c-norm-t-ratio
  - id: log-context-scale-params
    content: Log context_scale parameter statistics (mean, std, min, max) from DiTBlock params
    status: pending
---

# Thêm Enhanced Grad Metrics và Context Debug Metrics

## Mục tiêu

1. Thêm raw grad norm (trước LR scaling) và scaled grad norm (sau LR scaling) cho encoder và DiT
2. Thêm metrics cho c_norm/t_emb ratio sau layer norm
3. Thêm các debug metrics khác (context scale stats, LR info, etc.)

## Implementation

### 1. Raw và Scaled Grad Norms (train_util_jax.py)

- Tính raw grad norm TRƯỚC khi scale với LR
- Giữ scaled grad norm (sau khi scale) với tên rõ ràng
- Thêm effective step size (grad_norm * lr)

### 2. Context Norm Metrics sau LayerNorm (vfsddpm_jax.py)

- Simulate LayerNorm trên c (giống DiTBlock)
- Tính c_norm/t_emb ratio sau layer norm
- Thêm magnitude comparisons

### 3. Debug Metrics bổ sung

- Context norm std (should be ~1.0 after LayerNorm)
- LR scale info
- Log context_scale parameter values (mean, std, min, max) từ DiTBlock params
- Các metrics khác hữu ích

### 4. Log Context Scale Parameters

- Extract context_scale và context_scale_lag từ DiT params
- Log statistics: mean, std, min, max của scale parameters
- Giúp debug xem scale parameters có học được không