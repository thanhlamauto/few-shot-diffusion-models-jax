"""
Quick test: Compute FID between two random subsets of real train images.
This verifies that the FID computation code works correctly.
Expected result: FID should be very low (close to 0) since both sets are real images.
"""

import argparse
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import select_dataset
from metrics import fid_jax

def test_fid_real_images(args):
    print("\n" + "="*70)
    print("Testing FID Computation on Real Images Only")
    print("="*70)
    
    # Load train dataset
    print(f"\n📂 Loading {args.dataset} train dataset...")
    dataset = select_dataset(args, split="train")
    
    all_images = dataset.data['inputs']  # (n_sets, ns, C, H, W)
    print(f"✅ Dataset loaded: {all_images.shape}")
    
    # Flatten to get individual images
    all_images_flat = all_images.reshape(-1, *all_images.shape[2:])  # (N, C, H, W)
    print(f"   Total images: {len(all_images_flat)}")
    
    # Randomly sample n_samples images
    n_samples = args.n_samples
    indices = np.random.choice(len(all_images_flat), size=n_samples, replace=False)
    sampled_images = all_images_flat[indices]
    
    print(f"\n🎲 Randomly sampled {n_samples} images")
    
    # Split into two groups
    split_point = n_samples // 2
    group1 = sampled_images[:split_point]
    group2 = sampled_images[split_point:]
    
    print(f"   Group 1: {len(group1)} images")
    print(f"   Group 2: {len(group2)} images")
    
    # Convert to HWC format for InceptionV3
    group1_hwc = group1.transpose(0, 2, 3, 1)  # (N, H, W, C)
    group2_hwc = group2.transpose(0, 2, 3, 1)  # (N, H, W, C)
    
    print(f"\n   Image value range: [{group1_hwc.min():.2f}, {group1_hwc.max():.2f}]")
    print(f"   Shape: {group1_hwc.shape}")
    
    # Load InceptionV3
    print(f"\n🔧 Loading InceptionV3 for FID computation...")
    inception_fn = fid_jax.get_fid_fn()
    
    # Compute FID
    print(f"\n🔄 Computing FID between two random subsets of real images...")
    print(f"   (Expected: FID should be LOW, close to 0, since both are real)")
    
    fid_score = fid_jax.compute_fid(
        group1_hwc,
        group2_hwc,
        inception_fn=inception_fn,
        batch_size=min(64, len(group1))
    )
    
    print(f"\n✅ FID Score: {fid_score:.2f}")
    print(f"\n{'='*70}")
    
    if fid_score < 50:
        print("✅ SUCCESS: FID is low as expected (both groups are real images)")
    else:
        print("⚠️  WARNING: FID is higher than expected. This might indicate:")
        print("   1. Sample size is too small (try --n_samples 100 or more)")
        print("   2. Images are from very different classes")
        print("   3. There might be an issue with the FID computation")
    
    print("="*70 + "\n")
    
    return fid_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FID computation on real images")
    
    # Dataset args
    parser.add_argument("--dataset", type=str, default="cifar100", help="Dataset name")
    parser.add_argument("--data_dir", type=str, default="/kaggle/working/ns_data", help="Data directory")
    parser.add_argument("--sample_size", type=int, default=6, help="Images per set")
    parser.add_argument("--num_classes", type=int, default=1, help="Classes per task")
    parser.add_argument("--augment", action="store_true", default=False, help="Data augmentation")
    
    # Test args
    parser.add_argument("--n_samples", type=int, default=20, 
                        help="Number of images to sample (will be split into 2 groups)")
    
    args = parser.parse_args()
    
    # Verify n_samples is even
    if args.n_samples % 2 != 0:
        print(f"⚠️  Warning: n_samples={args.n_samples} is odd, using {args.n_samples-1}")
        args.n_samples -= 1
    
    test_fid_real_images(args)
