#!/bin/bash
# Fixed command with corrected syntax

python main_jax.py \
  --model vfsddpm_jax --encoder_mode vit_set --dataset cifar100 --data_dir /kaggle/working/ns_data \
  --sample_size 6 --image_size 32 --patch_size 1 --batch_size 32 \
  --encoder_lr 1.2e-3 \
  --dit_lr 2e-4 \
  --weight_decay 0.01 \
  --pool mean \
  --use_vae True \
  --log_interval 300 --save_interval 20000 --num_eval_batches 10 --num_sample_batches 2 \
  --use_wandb True --wandb_project fsdm-jax --max_steps 300000 \
  --diffusion_steps 150 --hidden_size 512 --depth 12 --num_heads 8 --mlp_ratio 4.0 \
  --compute_fid True --fid_num_samples 512 \
  --encoder_depth 6 \
  --encoder_heads 12 \
  --debug_metrics True \
  --use_context_layernorm False \
  --encoder_dim_head 64 \
  --encoder_mlp_ratio 3.0 \
  --input_dependent True \
  --encoder_tokenize_mode stack --dropout 0.2 --use_ddim True --ddim_num_steps 150

