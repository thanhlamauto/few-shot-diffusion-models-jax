#!/bin/bash

# Training command with DiT Cross-Attention (Lag Mode) - MINIMAL MEMORY VERSION
# Tối ưu tối đa cho Kaggle TPU với 330GB RAM
# 
# Giảm mạnh:
# - batch_size: 8 -> 2 (giảm 4x)
# - sample_size: 4 -> 3 (giảm 25%)
# - num_heads: 6 -> 4 (giảm 33% attention memory)
# - encoder_heads: 12 -> 8 (giảm 33%)

python main_jax.py \
  --model vfsddpm_jax \
  --encoder_mode vit_set \
  --dataset cifar100 \
  --data_dir /kaggle/working/ns_data \
  --sample_size 3 \
  --image_size 32 \
  --patch_size 2 \
  --batch_size 2 \
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
  --num_heads 4 \
  --mlp_ratio 3.0 \
  --compute_fid True \
  --fid_num_samples 10000 \
  --encoder_depth 6 \
  --encoder_heads 8 \
  --encoder_dim_head 64 \
  --encoder_mlp_ratio 1.0 \
  --encoder_tokenize_mode stack \
  --hdim 256 \
  --context_channels 256 \
  --mode_conditioning lag
