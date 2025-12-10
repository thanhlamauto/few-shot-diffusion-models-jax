"""
Calculate new config: depth=6 but same total params as depth=12, hidden=384
"""

import math

print("="*80)
print("CALCULATE NEW CONFIG: depth=6, ~43.5M params")
print("="*80)

# Current config
current_depth = 12
current_hidden = 384
current_heads = 6
current_total = 43.5e6

# Fixed components (don't change much with hidden_size)
# Encoder depends on hdim, but we keep hdim=384
encoder_params = 10.8e6
embeddings_and_final = 0.8e6  # rough estimate

# Current DiT blocks only
current_dit_blocks = current_total - encoder_params - embeddings_and_final
print(f"\nCurrent config:")
print(f"  depth={current_depth}, hidden_size={current_hidden}, num_heads={current_heads}")
print(f"  Total params: {current_total/1e6:.1f}M")
print(f"  DiT blocks: {current_dit_blocks/1e6:.1f}M")

# Per block params (approximation)
# Main components: Attention (~4H^2) + MLP (~8H^2) + adaLN (~6H^2) ≈ 18H^2
H_current = current_hidden
per_block_current = current_dit_blocks / current_depth

print(f"  Per block: {per_block_current/1e6:.2f}M")

# Target: depth=6, same total params
target_depth = 6
target_total = current_total
target_dit_blocks = target_total - encoder_params - embeddings_and_final

print(f"\nTarget config:")
print(f"  depth={target_depth}")
print(f"  Target DiT blocks: {target_dit_blocks/1e6:.1f}M")

per_block_target = target_dit_blocks / target_depth
print(f"  Per block needed: {per_block_target/1e6:.2f}M")

# Estimate new hidden_size
# per_block ≈ 18 * H^2 (rough approximation)
# per_block_target / per_block_current ≈ (H_new / H_current)^2
ratio = per_block_target / per_block_current
H_new = H_current * math.sqrt(ratio)

print(f"\nRatio: {ratio:.2f}")
print(f"Estimated H_new: {H_new:.0f}")

# Try different options (must be divisible by num_heads)
options = [
    (512, 8),   # 64 per head
    (576, 12),  # 48 per head
    (576, 8),   # 72 per head
    (640, 10),  # 64 per head
    (768, 12),  # 64 per head
]

print("\n" + "="*80)
print("CANDIDATE CONFIGS:")
print("="*80)

print(f"\n{'hidden_size':<12} {'num_heads':<10} {'head_dim':<10} {'Est. Params':<15} {'Diff from Target':<20}")
print("-" * 80)

best_option = None
best_diff = float('inf')

for H_new, heads_new in options:
    # More accurate calculation
    # Per block components:
    # LayerNorm: 4H
    # QKV: H*(3H) + 3H = 3H^2 + 3H
    # Attn proj: H*H + H = H^2 + H
    # MLP: H*(4H) + 4H + 4H*H + H = 8H^2 + 5H
    # adaLN: H*(6H) + 6H = 6H^2 + 6H
    # Total: 18H^2 + 19H
    
    per_block = 18 * H_new**2 + 19 * H_new
    total_blocks = per_block * target_depth
    
    # Embeddings scale with hidden_size too
    # Patch embed: patch_dim * H
    # Pos embed: num_patches * H
    # Time embed: ~H^2
    # Context proj: ~H^2 (384 * H if hdim=384)
    patch_dim = 12  # 3*2*2
    num_patches = 256
    embeddings_new = (patch_dim * H_new + 
                      num_patches * H_new + 
                      256 * H_new +  # time MLP approx
                      384 * H_new)  # context proj
    
    # Final layer: H * patch_dim_out
    final_new = H_new * 12
    
    total_params = encoder_params + total_blocks + embeddings_new + final_new
    
    diff = total_params - target_total
    diff_pct = (diff / target_total) * 100
    
    head_dim = H_new // heads_new
    
    print(f"{H_new:<12} {heads_new:<10} {head_dim:<10} {total_params/1e6:>12.1f}M   {diff/1e6:>+10.1f}M ({diff_pct:>+5.1f}%)")
    
    if abs(diff) < abs(best_diff):
        best_diff = diff
        best_option = (H_new, heads_new, total_params)

print("\n" + "="*80)
print("RECOMMENDED CONFIG:")
print("="*80)

H_rec, heads_rec, params_rec = best_option

print(f"""
✅ BEST MATCH:
   --depth 6 \\
   --hidden_size {H_rec} \\
   --num_heads {heads_rec} \\
   --mlp_ratio 4.0 \\
   --hdim 384 \\
   --context_channels 384
   
   Total params: {params_rec/1e6:.1f}M
   Difference from target: {best_diff/1e6:+.1f}M ({(best_diff/target_total)*100:+.1f}%)
   
   Head dimension: {H_rec//heads_rec}
""")

print("="*80)
print("COMPARISON:")
print("="*80)

print(f"""
BEFORE (depth=12, hidden=384):
  Total params:     43.5M
  DiT blocks:       31.9M (2.66M × 12)
  Per-block:        2.66M
  Training speed:   Baseline

AFTER (depth=6, hidden={H_rec}):
  Total params:     {params_rec/1e6:.1f}M
  DiT blocks:       {(params_rec - encoder_params - 1e6)/1e6:.1f}M ({((params_rec - encoder_params - 1e6)/(target_depth*1e6)):.1f}M × {target_depth})
  Per-block:        {((params_rec - encoder_params - 1e6)/(target_depth*1e6)):.1f}M
  Training speed:   ~100% FASTER (half depth, but wider)
  
KEY CHANGES:
  ✅ Depth: 12 → 6 (50% reduction)
  ✅ Hidden size: 384 → {H_rec} ({((H_rec-384)/384)*100:+.0f}%)
  ✅ Num heads: 6 → {heads_rec}
  ✅ Total params: Similar (~43M)
  
ADVANTAGES:
  ✅ Shallower network → Better gradient flow
  ✅ Wider layers → More capacity per layer
  ✅ Faster per step (fewer transformer blocks)
  ✅ Context should learn faster
  
TRADE-OFFS:
  ⚠️  Wider = more memory per layer
  ⚠️  But fewer layers = similar total memory
""")

print("="*80)
print("FULL TRAINING COMMAND:")
print("="*80)

print(f"""
python main_jax.py \\
    --model vfsddpm_jax \\
    --dataset cifar100 \\
    --data_dir /kaggle/working/ns_data \\
    --sample_size 6 \\
    --image_size 32 \\
    --patch_size 2 \\
    --batch_size 32 \\
    --lr 1e-4 \\
    --log_interval 100 \\
    --save_interval 20000 \\
    --num_eval_batches 10 \\
    --num_sample_batches 2 \\
    --use_wandb \\
    --wandb_project fsdm-jax \\
    --max_steps 200000 \\
    --diffusion_steps 250 \\
    --hidden_size {H_rec} \\
    --depth 6 \\
    --num_heads {heads_rec} \\
    --mlp_ratio 4.0 \\
    --hdim 384 \\
    --context_channels 384 \\
    --compute_fid \\
    --fid_num_samples 600
""")

print("="*80)
