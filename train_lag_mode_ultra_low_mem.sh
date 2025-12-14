#!/bin/bash

# Training command with DiT Cross-Attention (Lag Mode) - ULTRA LOW MEMORY VERSION
# Tối ưu cho Kaggle TPU với 330GB RAM
# 
# Memory breakdown với setting này:
# - Context: (4*3, 256, 256) = ~0.8MB (giảm 98% so với batch_size=32, sample_size=6)
# - Batch: 4 sets * 3 images = 12 images total
# - Encoder loops: 3 lần thay vì 6
# - Tổng memory: ~50-80GB (an toàn cho 330GB)

python main_jax.py \
  --model vfsddpm_jax \
  --encoder_mode vit_set \
  --dataset cifar100 \
  --data_dir /kaggle/working/ns_data \
  --sample_size 3 \
  --image_size 32 \
  --patch_size 2 \
  --batch_size 4 \
  --lr 1e-4 \
  --log_interval 300 \
  --save_interval 20000 \
  --num_eval_batches 10 \
  --num_sample_batches 2 \
  --use_wandb True \
  --wandb_project fsdm-jax \
  --max_steps 300000 \
  --diffusion_steps 250 \
  --hidden_size 468 \
  --depth 6 \
  --num_heads 9 \
  --mlp_ratio 3.0 \
  --compute_fid True \
  --fid_num_samples 1024 \
  --encoder_depth 6 \
  --encoder_heads 12 \
  --encoder_dim_head 64 \
  --encoder_mlp_ratio 1.0 \
  --encoder_tokenize_mode stack \
  --hdim 256 \
  --context_channels 256 \
  --mode_conditioning lag \
  --dropout 0.2
