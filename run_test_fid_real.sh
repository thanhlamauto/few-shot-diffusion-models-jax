#!/bin/bash

# Quick test: Compute FID between two random subsets of real train images
# This verifies FID computation code works
# Expected: FID should be very low (close to 0) since both are real images

echo "========================================"
echo "Testing FID on Real Images (20 samples)"
echo "========================================"
echo ""
echo "This will:"
echo "  1. Load 20 random images from train set"
echo "  2. Split into 2 groups (10 + 10)"
echo "  3. Compute FID between the groups"
echo "  4. Expected result: FID close to 0"
echo ""

python test_fid_real_only.py \
  --dataset cifar100 \
  --data_dir /kaggle/working/ns_data \
  --sample_size 6 \
  --n_samples 20

echo ""
echo "========================================"
echo "Test Complete!"
echo ""
echo "If FID is low (< 50), the code works correctly."
echo "You can now test with generated images."
echo "========================================"
