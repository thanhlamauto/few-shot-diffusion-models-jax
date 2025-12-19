#!/bin/bash
# Command để train với pretrained ViT encoder và freeze encoder
# Pretrained ViT architecture: dim=192, depth=9, heads=3, dim_head=64, mlp_ratio=2.0

python main_jax.py \
  --model vfsddpm_jax --encoder_mode vit --dataset cifar100 --data_dir /kaggle/working/ns_data \
  --sample_size 6 --image_size 32 --patch_size 4 --batch_size 256 \
  --encoder_lr 1e-3 \
  --dit_lr 2e-4 \
  --weight_decay=0.01 \
  --pool mean_patch \
  --hdim 192 \
  --context_channels 192 \
  --in_channels 3 \
  --use_vae True \
  --encoder_uses_vae False \
  --log_interval 300 --save_interval 20000 --num_eval_batches 10 --num_sample_batches 2 \
  --use_wandb True --wandb_project fsdm-jax --max_steps 300000 \
  --diffusion_steps 150 --hidden_size 456 --depth 12 --num_heads 8 --mlp_ratio 4.0 \
  --compute_fid True --fid_num_samples 512 \
  --encoder_depth 9 \
  --encoder_heads 3 \
  --debug_metrics True \
  --use_context_layernorm False \
  --encoder_dim_head 64 \
  --encoder_mlp_ratio 2.0 \
  --input_dependent True \
  --generation_split train \
  --mode_conditioning lag \
  --dropout 0.2 --use_ddim True --ddim_num_steps=150 \
  --pretrained_encoder_path /kaggle/input/vit-jax/jax/default/1/vit_cifar10_patch4_input32_jax.npz \
  --freeze_encoder_steps 1

