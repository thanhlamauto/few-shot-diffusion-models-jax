#!/bin/bash

# Test FID calculation on Kaggle
# This script computes both IN-distribution (train classes) and OUT-distribution (test classes) FID
# following the paper methodology

echo "========================================"
echo "Testing FID Calculation on Kaggle"
echo "========================================"

# Configuration
USE_CHECKPOINT="${1:-no}"  # Pass "yes" to use checkpoint, default is "no" (random weights)
CHECKPOINT_PATH="${2:-/kaggle/working/few-shot-diffusion-models-jax/checkpoints_jax/ckpt_020000}"
DATA_DIR="/kaggle/working/ns_data"
DATASET="cifar100"
FID_MODE="in"  # Options: "in", "out", "both", "per_class"
FID_NUM_SAMPLES=20  # Use 20 for quick test, 1024 for real eval, 10000 for paper results
SAMPLE_SIZE=6

echo ""
echo "Settings:"
if [ "$USE_CHECKPOINT" = "yes" ]; then
    echo "  Mode: Using trained checkpoint"
    echo "  Checkpoint: $CHECKPOINT_PATH"
else
    echo "  Mode: Random weights (quick test)"
    echo "  Note: FID will be HIGH (bad) but verifies code works"
fi
echo "  Dataset: $DATASET"
echo "  FID Mode: $FID_MODE"
echo "  FID Samples: $FID_NUM_SAMPLES"
echo ""
echo "Note: max_steps=0 will skip all training and only compute FID"
echo ""

# Build command
CMD="python main_jax.py \
  --model vfsddpm_jax \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --sample_size $SAMPLE_SIZE \
  --image_size 32 \
  --patch_size 2 \
  --batch_size 32 \
  --lr 1e-4 \
  --log_interval 100 \
  --save_interval 999999 \
  --num_eval_batches 0 \
  --num_sample_batches 0 \
  --use_wandb False \
  --max_steps 0 \
  --diffusion_steps 250 \
  --hidden_size 468 \
  --depth 6 \
  --num_heads 9 \
  --mlp_ratio 3.0 \
  --compute_fid True \
  --fid_mode $FID_MODE \
  --fid_num_samples $FID_NUM_SAMPLES"

# Add checkpoint args if needed
if [ "$USE_CHECKPOINT" = "yes" ]; then
    CMD="$CMD --checkpoint_path $CHECKPOINT_PATH --resume True"
fi

# Run
eval $CMD

echo ""
echo "========================================"
echo "FID Calculation Complete!"
if [ "$USE_CHECKPOINT" != "yes" ]; then
    echo ""
    echo "✅ If this ran without errors, the FID code works!"
    echo "   (FID score will be high with random weights)"
    echo ""
    echo "To test with trained checkpoint, run:"
    echo "  bash test_fid_kaggle.sh yes /path/to/checkpoint"
fi
echo "========================================"
