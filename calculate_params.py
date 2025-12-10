"""
Calculate model parameters for the given configuration
"""

import numpy as np

print("="*80)
print("CALCULATE MODEL PARAMETERS")
print("="*80)

# Configuration from command
config = {
    "sample_size": 6,
    "image_size": 32,
    "patch_size": 2,
    "hidden_size": 384,
    "depth": 12,
    "num_heads": 6,
    "mlp_ratio": 4.0,
    "hdim": 384,  # After fix
    "context_channels": 384,  # After fix
    "in_channels": 3,
    "learn_sigma": False,
}

print("\n📋 Configuration:")
print("-" * 80)
for k, v in config.items():
    print(f"  {k:20s} = {v}")

print("\n" + "="*80)
print("1. DiT (Diffusion Transformer)")
print("="*80)

# DiT components
H = config["hidden_size"]
D = config["depth"]
num_heads = config["num_heads"]
mlp_ratio = config["mlp_ratio"]
patch_size = config["patch_size"]
img_size = config["image_size"]
in_ch = config["in_channels"]
context_ch = config["context_channels"]

# Calculate derived values
num_patches = (img_size // patch_size) ** 2
mlp_hidden = int(H * mlp_ratio)

print(f"\nDerived values:")
print(f"  num_patches = {num_patches} ({img_size}//{patch_size})^2")
print(f"  mlp_hidden = {mlp_hidden} (hidden_size × mlp_ratio)")

# 1. Patch Embedding
patch_dim = in_ch * patch_size * patch_size  # 3 * 2 * 2 = 12
patch_embed_params = patch_dim * H  # projection
print(f"\n1.1 Patch Embedding:")
print(f"  Linear: {patch_dim} → {H}")
print(f"  Params: {patch_embed_params:,}")

# 2. Position Embedding
pos_embed_params = num_patches * H
print(f"\n1.2 Position Embedding:")
print(f"  Shape: ({num_patches}, {H})")
print(f"  Params: {pos_embed_params:,}")

# 3. Time Embedding (MLP: model_channels → 4*model_channels → hidden_size)
# Assuming model_channels = 256 (default)
model_ch = 256
time_embed_params = (model_ch * H) + (H * H)  # Two layers
print(f"\n1.3 Time Embedding:")
print(f"  MLP: {model_ch} → {H} → {H}")
print(f"  Params: {time_embed_params:,}")

# 4. Context Projection (in DiT forward, FiLM mode)
context_proj_params = context_ch * H + H  # Dense layer + bias
print(f"\n1.4 Context Projection:")
print(f"  Dense: {context_ch} → {H}")
print(f"  Params: {context_proj_params:,}")

# 5. DiT Blocks
print(f"\n1.5 DiT Blocks (×{D}):")

# Per block:
# - LayerNorm (scale + bias): 2 * H
# - Self-Attention:
#   - qkv projection: H → 3*H
#   - output projection: H → H
# - MLP:
#   - fc1: H → mlp_hidden
#   - fc2: mlp_hidden → H
# - adaLN-Zero conditioning:
#   - H → 6*H (scale, shift for attn and mlp, gate for both)

ln_params = 2 * H  # scale + bias

# Self-Attention
qkv_params = H * (3 * H) + (3 * H)  # weight + bias
attn_proj_params = H * H + H
attn_total = qkv_params + attn_proj_params

# MLP
fc1_params = H * mlp_hidden + mlp_hidden
fc2_params = mlp_hidden * H + H
mlp_total = fc1_params + fc2_params

# adaLN-Zero (conditioning projection)
adaln_params = H * (6 * H) + (6 * H)  # project to 6*H for scale/shift/gate

block_params = ln_params * 2 + attn_total + mlp_total + adaln_params

print(f"  Per block:")
print(f"    LayerNorm (×2):        {ln_params * 2:,}")
print(f"    Self-Attention:        {attn_total:,}")
print(f"      - QKV projection:    {qkv_params:,}")
print(f"      - Output projection: {attn_proj_params:,}")
print(f"    MLP:                   {mlp_total:,}")
print(f"      - FC1 ({H}→{mlp_hidden}): {fc1_params:,}")
print(f"      - FC2 ({mlp_hidden}→{H}): {fc2_params:,}")
print(f"    adaLN-Zero:            {adaln_params:,}")
print(f"    Total per block:       {block_params:,}")
print(f"  All {D} blocks:           {block_params * D:,}")

# 6. Final Layer
final_norm_params = 2 * H
final_adaln_params = H * (2 * H) + (2 * H)  # scale + shift
out_ch = in_ch * 2 if config["learn_sigma"] else in_ch
final_linear_params = H * (patch_size * patch_size * out_ch) + (patch_size * patch_size * out_ch)

final_layer_params = final_norm_params + final_adaln_params + final_linear_params

print(f"\n1.6 Final Layer:")
print(f"  LayerNorm:       {final_norm_params:,}")
print(f"  adaLN:           {final_adaln_params:,}")
print(f"  Linear:          {final_linear_params:,}")
print(f"  Total:           {final_layer_params:,}")

# Total DiT
dit_total = (patch_embed_params + pos_embed_params + time_embed_params + 
             context_proj_params + block_params * D + final_layer_params)

print(f"\n{'='*80}")
print(f"DiT Total Parameters: {dit_total:,}")
print(f"{'='*80}")

print("\n" + "="*80)
print("2. Encoder (sViT)")
print("="*80)

hdim = config["hdim"]
sample_size = config["sample_size"]

# sViT uses SPT (Shifted Patch Tokenization)
# Input: (bs, ns, C, H, W) → (bs, C*ns, H, W)
# SPT processes C*ns channels

spt_in_ch = in_ch * sample_size  # 3 * 6 = 18
spt_patch_dim = spt_in_ch * patch_size * patch_size  # 18 * 2 * 2 = 72

# SPT projection
spt_proj_params = spt_patch_dim * hdim + hdim

print(f"\n2.1 SPT (Shifted Patch Tokenization):")
print(f"  Input channels: {in_ch} × {sample_size} = {spt_in_ch}")
print(f"  Patch dim: {spt_patch_dim}")
print(f"  Projection: {spt_patch_dim} → {hdim}")
print(f"  Params: {spt_proj_params:,}")

# Position embedding for encoder
encoder_num_patches = num_patches  # Same as DiT
encoder_pos_embed = encoder_num_patches * hdim + hdim  # +1 for CLS token

print(f"\n2.2 Position Embedding:")
print(f"  Patches: {encoder_num_patches} + 1 CLS")
print(f"  Params: {encoder_pos_embed:,}")

# sViT Transformer blocks (assume same depth as DiT for simplicity)
# Actually, default is often 4-6 layers for encoder
encoder_depth = 6  # Common setting

# Per encoder block (similar to DiT but simpler, no conditioning)
enc_ln_params = 2 * hdim
enc_qkv_params = hdim * (3 * hdim) + (3 * hdim)
enc_attn_proj_params = hdim * hdim + hdim
enc_attn_total = enc_qkv_params + enc_attn_proj_params

enc_mlp_hidden = int(hdim * mlp_ratio)
enc_fc1_params = hdim * enc_mlp_hidden + enc_mlp_hidden
enc_fc2_params = enc_mlp_hidden * hdim + hdim
enc_mlp_total = enc_fc1_params + enc_fc2_params

enc_block_params = enc_ln_params * 2 + enc_attn_total + enc_mlp_total

print(f"\n2.3 Transformer Blocks (×{encoder_depth}):")
print(f"  Per block:")
print(f"    LayerNorm (×2):  {enc_ln_params * 2:,}")
print(f"    Self-Attention:  {enc_attn_total:,}")
print(f"    MLP:             {enc_mlp_total:,}")
print(f"    Total per block: {enc_block_params:,}")
print(f"  All {encoder_depth} blocks:  {enc_block_params * encoder_depth:,}")

# Final norm
enc_final_norm = 2 * hdim

encoder_total = (spt_proj_params + encoder_pos_embed + 
                enc_block_params * encoder_depth + enc_final_norm)

print(f"\n2.4 Final LayerNorm: {enc_final_norm:,}")

print(f"\n{'='*80}")
print(f"Encoder Total Parameters: {encoder_total:,}")
print(f"{'='*80}")

print("\n" + "="*80)
print("3. Posterior (Optional, for VAE mode)")
print("="*80)

# If mode_context = "variational", posterior is used
# Typically small MLPs: hdim → 2*hdim (mean + logvar)
posterior_params = hdim * (2 * hdim) + (2 * hdim)

print(f"  If using VAE (mode_context='variational'):")
print(f"  MLP: {hdim} → {2*hdim}")
print(f"  Params: {posterior_params:,}")
print(f"  (Not counted in total if mode_context='deterministic')")

print("\n" + "="*80)
print("TOTAL MODEL PARAMETERS")
print("="*80)

total_params = dit_total + encoder_total
# Not counting posterior if deterministic

print(f"\n  Encoder:     {encoder_total:15,} params")
print(f"  DiT:         {dit_total:15,} params")
print(f"  " + "─" * 40)
print(f"  TOTAL:       {total_params:15,} params")
print(f"  TOTAL:       {total_params / 1e6:15.2f} M params")

print("\n" + "="*80)
print("COMPARISON WITH DIFFERENT hdim:")
print("="*80)

# Original (hdim=256)
original_hdim = 256
original_context_ch = 256

# Encoder with hdim=256
spt_in_ch_orig = in_ch * sample_size
spt_patch_dim_orig = spt_in_ch_orig * patch_size * patch_size
spt_proj_orig = spt_patch_dim_orig * original_hdim + original_hdim
pos_embed_orig = encoder_num_patches * original_hdim + original_hdim

enc_block_orig = (
    2 * 2 * original_hdim +  # LayerNorm
    original_hdim * (3 * original_hdim) + (3 * original_hdim) +  # QKV
    original_hdim * original_hdim + original_hdim +  # Proj
    original_hdim * int(original_hdim * mlp_ratio) + int(original_hdim * mlp_ratio) +  # FC1
    int(original_hdim * mlp_ratio) * original_hdim + original_hdim  # FC2
)
enc_final_norm_orig = 2 * original_hdim
encoder_orig = spt_proj_orig + pos_embed_orig + enc_block_orig * encoder_depth + enc_final_norm_orig

# DiT with context_channels=256
context_proj_orig = original_context_ch * H + H

# DiT stays same except context projection
dit_orig = dit_total - context_proj_params + context_proj_orig

total_orig = encoder_orig + dit_orig

print(f"\nWith hdim=256, context_channels=256 (BEFORE FIX):")
print(f"  Encoder:     {encoder_orig:15,} params")
print(f"  DiT:         {dit_orig:15,} params")
print(f"  TOTAL:       {total_orig:15,} params ({total_orig/1e6:.2f}M)")

print(f"\nWith hdim=384, context_channels=384 (AFTER FIX):")
print(f"  Encoder:     {encoder_total:15,} params")
print(f"  DiT:         {dit_total:15,} params")
print(f"  TOTAL:       {total_params:15,} params ({total_params/1e6:.2f}M)")

print(f"\nDifference:")
print(f"  +{total_params - total_orig:15,} params (+{((total_params - total_orig)/total_orig)*100:.1f}%)")

print("\n" + "="*80)
print("KEY TAKEAWAYS:")
print("="*80)

print(f"""
1. DiT (main generator):     ~{dit_total/1e6:.1f}M params
   - Most params in {D} transformer blocks
   - Each block: ~{block_params/1e6:.2f}M params

2. Encoder (context):        ~{encoder_total/1e6:.1f}M params
   - Processes support set (6 images)
   - {encoder_depth} transformer blocks

3. After dimension fix (256→384):
   - Total params: {total_params/1e6:.1f}M
   - Increase: +{((total_params - total_orig)/total_orig)*100:.1f}%
   - Trade-off for better context quality! ✅

4. Breakdown:
   - DiT blocks:         {(block_params * D)/1e6:.1f}M ({((block_params * D)/total_params)*100:.1f}%)
   - Encoder blocks:     {(enc_block_params * encoder_depth)/1e6:.1f}M ({((enc_block_params * encoder_depth)/total_params)*100:.1f}%)
   - Embeddings/others:  {(total_params - block_params * D - enc_block_params * encoder_depth)/1e6:.1f}M
""")

print("="*80)
