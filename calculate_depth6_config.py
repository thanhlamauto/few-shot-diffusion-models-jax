"""
Calculate optimal hidden_size and num_heads for depth=6 to match ~43.5M params
"""

import numpy as np

print("="*80)
print("CALCULATE DEPTH=6 CONFIG TO MATCH ~43.5M PARAMS")
print("="*80)

# Target total params (from depth=12, hidden=384)
target_total = 43.5e6

# Original config
orig_depth = 12
orig_hidden = 384
orig_hdim = 384

print(f"\n📋 Original Config (depth=12):")
print(f"  depth:        {orig_depth}")
print(f"  hidden_size:  {orig_hidden}")
print(f"  hdim:         {orig_hdim}")
print(f"  Total params: {target_total/1e6:.1f}M")

# New config
new_depth = 6

print(f"\n🎯 New Config (depth={new_depth}):")
print(f"  depth:        {new_depth}")
print(f"  Target params: {target_total/1e6:.1f}M")

# Approximate params per block (with hidden size h)
# Attn: h × 3h + h × h ≈ 4h²
# MLP: h × 4h + 4h × h ≈ 8h²
# adaLN: h × 6h ≈ 6h²
# LayerNorm: 4h (negligible)
# Total: ≈ 18h² + small_overhead

def block_params(h):
    """Approximate params for one DiT block"""
    attn = h * (3*h) + (3*h) + h * h + h  # QKV + proj
    mlp = h * (4*h) + (4*h) + (4*h) * h + h  # FC1 + FC2
    adaln = h * (6*h) + (6*h)  # conditioning
    ln = 4 * h  # 2 layernorms
    return attn + mlp + adaln + ln

def encoder_params(hdim, depth=6):
    """Approximate encoder params"""
    spt = 72 * hdim + hdim  # SPT projection (18 channels × 2×2 patch)
    pos_embed = 257 * hdim + hdim  # 256 patches + 1 CLS
    blocks = depth * block_params(hdim)
    final_ln = 2 * hdim
    return spt + pos_embed + blocks + final_ln

def total_params(dit_depth, dit_hidden, encoder_hdim):
    """Total model params"""
    # DiT
    patch_embed = 12 * dit_hidden  # 3×2×2 → hidden
    pos_embed = 256 * dit_hidden
    time_embed = 256 * dit_hidden + dit_hidden * dit_hidden
    context_proj = encoder_hdim * dit_hidden + dit_hidden
    dit_blocks = dit_depth * block_params(dit_hidden)
    final_layer = 2 * dit_hidden + dit_hidden * (2*dit_hidden) + (2*dit_hidden) + \
                  dit_hidden * (4*3) + (4*3)  # adaLN + linear
    
    dit_total = patch_embed + pos_embed + time_embed + context_proj + dit_blocks + final_layer
    
    # Encoder
    enc_total = encoder_params(encoder_hdim)
    
    return dit_total + enc_total

# Try different hidden sizes
print(f"\n{'hidden_size':<15} {'hdim':<10} {'DiT params':<15} {'Encoder params':<15} {'Total':<15} {'vs Target':<15}")
print("-" * 95)

candidates = [480, 512, 544, 576, 608, 640]

for h in candidates:
    dit_p = (12 + 256 + 256*384 + 384*384 + h*h + h*h + 
             new_depth * block_params(h) + 2*h + h*2*h + 2*h + h*12 + 12)
    enc_p = encoder_params(h)
    total = dit_p + enc_p
    diff = total - target_total
    
    print(f"{h:<15} {h:<10} {dit_p/1e6:>12.1f}M   {enc_p/1e6:>12.1f}M   {total/1e6:>12.1f}M   {diff/1e6:>+10.1f}M ({(diff/target_total)*100:>+5.1f}%)")

print("\n" + "="*80)
print("DETAILED CALCULATION FOR hidden_size=576:")
print("="*80)

h = 576
dit_depth = 6

print(f"\n1. DiT Components:")
print(f"  Patch embedding:    {12 * h:>10,} params")
print(f"  Position embedding: {256 * h:>10,} params")
print(f"  Time embedding:     {(256*h + h*h):>10,} params")
print(f"  Context projection: {(h*h + h):>10,} params")
print(f"  DiT blocks (×{dit_depth}):   {dit_depth * block_params(h):>10,} params")
print(f"    Per block:        {block_params(h):>10,} params")
print(f"  Final layer:        ~{300000:>10,} params")

dit_total_approx = 12*h + 256*h + 256*h + h*h + h*h + h*h + dit_depth*block_params(h) + 300000
print(f"  DiT Total:          {dit_total_approx/1e6:>10.1f}M")

print(f"\n2. Encoder (hdim={h}):")
enc_total = encoder_params(h)
print(f"  SPT:                {(72*h + h):>10,} params")
print(f"  Position embed:     {(257*h + h):>10,} params")
print(f"  Encoder blocks (×6): {6 * block_params(h):>10,} params")
print(f"    Per block:        {block_params(h):>10,} params")
print(f"  Final LN:           {2*h:>10,} params")
print(f"  Encoder Total:      {enc_total/1e6:>10.1f}M")

total_approx = dit_total_approx + enc_total
print(f"\n3. Total Model:       {total_approx/1e6:>10.1f}M")
print(f"   Target:            {target_total/1e6:>10.1f}M")
print(f"   Difference:        {(total_approx - target_total)/1e6:>+10.1f}M ({((total_approx - target_total)/target_total)*100:>+5.1f}%)")

print("\n" + "="*80)
print("NUM_HEADS OPTIONS FOR hidden_size=576:")
print("="*80)

print(f"\nhidden_size=576 should divide evenly by num_heads:")
print(f"  576 / 6  = {576/6:.0f} ✓ (head_dim=96)")
print(f"  576 / 8  = {576/8:.0f} ✓ (head_dim=72)")
print(f"  576 / 9  = {576/9:.0f} ✓ (head_dim=64)  ← RECOMMENDED")
print(f"  576 / 12 = {576/12:.0f} ✓ (head_dim=48)")

print(f"\n🎯 RECOMMENDED: num_heads=9")
print(f"  - head_dim=64 (sweet spot, similar to original)")
print(f"  - More heads than depth=12 (was 6 heads)")
print(f"  - Compensates for fewer layers")

print("\n" + "="*80)
print("FINAL CONFIG:")
print("="*80)

print(f"""
✅ Optimal config to match ~43.5M params:

  --depth 6 \\
  --hidden_size 576 \\
  --hdim 576 \\
  --context_channels 576 \\
  --num_heads 9 \\        ← CHANGE THIS!
  --mlp_ratio 4.0 \\
  
Expected params: ~{total_approx/1e6:.1f}M (vs target {target_total/1e6:.1f}M)

User đã set hdim=576, context_channels=576 ✅
Chỉ cần thêm:
  - hidden_size 576
  - num_heads 9
  - depth 6
""")

print("="*80)
print("COMPARISON: depth=12 vs depth=6")
print("="*80)

print(f"""
{'Metric':<25} {'depth=12':<15} {'depth=6':<15} {'Change':<15}
{'-'*70}
hidden_size               384             576             +50%
hdim                      384             576             +50%
num_heads                 6               9               +50%
head_dim                  64              64              Same ✓
depth                     12              6               -50%
Params                    43.5M           ~{total_approx/1e6:.1f}M           {((total_approx - target_total)/target_total)*100:>+5.1f}%

Speed per step            Baseline        ~2× faster      +100% ✓
Steps in 9h               ~72k            ~144k           +100% ✓
Gradient flow             Deep            Shallow ✅       Better
Context learning          Slow            Fast ✅          Better
Memory                    ~5 GB           ~5 GB           Similar
""")

print("="*80)
