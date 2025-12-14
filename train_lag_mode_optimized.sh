#!/bin/bash

# Training command with DiT Cross-Attention (Lag Mode) - OPTIMIZED FOR MEMORY
# Sử dụng các tối ưu memory:
# 1. Context pooling: giảm Nk từ 256 -> 64 (giảm 16x attention memory)
# 2. Cross-attn chỉ ở 4 layer cuối (giảm 50% cross-attn layers)
# 3. Remat: gradient checkpointing (giảm 30-50% peak memory)
# 4. Batch size và sample size nhỏ

python main_jax.py \
  --model vfsddpm_jax \
  --encoder_mode vit_set \
  --dataset cifar100 \
  --data_dir /kaggle/working/ns_data \
  --sample_size 4 \
  --image_size 32 \
  --patch_size 2 \
  --batch_size 8 \
  --dropout 0.2 \
  --lr 1e-4 \
  --log_interval 300 \
  --save_interval 10000 \
  --num_eval_batches 10 \
  --num_sample_batches 2 \
  --use_wandb True \
  --wandb_project fsdm-jax \
  --max_steps 300000 \
  --diffusion_steps 250 \
  --hidden_size 468 \
  --depth 6 \
  --num_heads 6 \
  --mlp_ratio 3.0 \
  --compute_fid True \
  --fid_num_samples 10000 \
  --encoder_depth 6 \
  --encoder_heads 12 \
  --encoder_dim_head 64 \
  --encoder_mlp_ratio 1.0 \
  --encoder_tokenize_mode stack \
  --hdim 256 \
  --context_channels 256 \
  --mode_conditioning lag \
  --context_pool_size 64 \
  --cross_attn_layers "2,3,4,5" \
  --use_remat True
