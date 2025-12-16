# model/set_diffusion/dit.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .nn import timestep_embedding, SiLU, zero_module

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(self, patch_size, embed_dim, in_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """2D sinusoidal positional embedding"""
    assert embed_dim % 2 == 0
    
    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        omega = torch.arange(embed_dim // 2, dtype=torch.float32)
        omega /= embed_dim / 2.
        omega = 1. / (10000 ** omega)
        pos = pos.reshape(-1)
        out = torch.einsum('m,d->md', pos, omega)
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        emb = torch.cat([emb_sin, emb_cos], dim=1)
        return emb
    
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = torch.cat([emb_h, emb_w], dim=1)
    return emb

def modulate(x, shift, scale):
    """FiLM modulation"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """DiT block with adaLN-Zero conditioning"""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, context_channels=0, mode_conditioning="film", use_context_layernorm=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.mode_conditioning = mode_conditioning
        self.use_context_layernorm = use_context_layernorm
        
        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # MLP
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        
        # Conditioning projection (adaLN-Zero)
        # 6 for adaLN-Zero: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        cond_dim = hidden_size + context_channels if mode_conditioning == "film" else hidden_size
        self.adaLN_modulation = nn.Sequential(
            SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        
        # Cross-attention for few-shot conditioning (if mode_conditioning == "lag")
        if mode_conditioning == "lag" and context_channels > 0:
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
            self.context_proj = nn.Linear(context_channels, hidden_size)
    
    def forward(self, x, c):
        """
        x: (B, N, hidden_size) - patch tokens
        c: (B, hidden_size) or (B, context_channels) - conditioning
        """
        # Get adaLN-Zero parameters
        if self.mode_conditioning == "film" and c is not None:
            # Concatenate time embedding with context if available
            if c.dim() == 2 and c.shape[-1] != self.hidden_size:
                # c is context embedding, need to combine with time embedding
                # This should be handled at a higher level
                pass
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(c).chunk(6, dim=-1)
        else:
            # No conditioning or lag mode
            shift_msa = scale_msa = gate_msa = shift_mlp = scale_mlp = gate_mlp = \
                torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        
        # Self-attention with adaLN-Zero
        x_norm = self.norm1(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)
        attn_out, _ = self.attn(x_modulated, x_modulated, x_modulated)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # Cross-attention for lag conditioning
        if self.mode_conditioning == "lag" and c is not None:
            x_norm_cross = self.norm_cross(x)
            c_proj = self.context_proj(c)  # (B, N_c, hidden_size) or (B, hidden_size)
            # Optional LayerNorm over context tokens before cross-attention
            if self.use_context_layernorm:
                # Handle both 2D and 3D context: apply LayerNorm over last dim
                context_norm = nn.functional.layer_norm(
                    c_proj, c_proj.shape[-1:], eps=1e-6
                )
                c_proj = context_norm
            if c_proj.dim() == 2:
                c_proj = c_proj.unsqueeze(1)  # (B, 1, hidden_size)
            cross_out, _ = self.cross_attn(x_norm_cross, c_proj, c_proj)
            x = x + cross_out
        
        # MLP with adaLN-Zero
        x_norm2 = self.norm2(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_modulated2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x

class FinalLayer(nn.Module):
    """Final layer to output patches"""
    def __init__(self, hidden_size, patch_size, out_channels, context_channels=0, mode_conditioning="film"):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        cond_dim = hidden_size + context_channels if mode_conditioning == "film" else hidden_size
        self.adaLN_modulation = nn.Sequential(
            SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiTModel(nn.Module):
    """
    Diffusion Transformer Model - replacement for UNetModel
    Compatible interface with UNetModel for few-shot diffusion
    """
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,  # hidden_size in DiT
        out_channels,
        num_res_blocks,  # depth in DiT
        attention_resolutions,  # not used in DiT, kept for compatibility
        context_channels=256,
        dropout=0.0,
        channel_mult="",  # not used in DiT
        conv_resample=True,  # not used in DiT
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,  # not used in DiT
        num_heads_upsample=-1,  # not used in DiT
        use_scale_shift_norm=True,  # not used in DiT
        resblock_updown=False,  # not used in DiT
        use_new_attention_order=False,  # not used in DiT
        mode_conditioning="film",
        transformer_depth=1,  # not used, use num_res_blocks as depth
        patch_size=2,  # DiT-specific: patch size for image
        mlp_ratio=4.0,  # DiT-specific: MLP expansion ratio
    ):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = model_channels
        self.depth = num_res_blocks
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio
        self.context_channels = context_channels
        self.mode_conditioning = mode_conditioning
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        
        # Calculate number of patches
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.num_patches = (image_size // patch_size) ** 2
        self.num_patches_side = image_size // patch_size
        
        # Patch embedding
        self.patch_embed = PatchEmbed(patch_size, model_channels, in_channels)
        
        # Positional embedding (learnable or sinusoidal)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, model_channels)
        )
        # Initialize with sinusoidal
        pos_embed = get_2d_sincos_pos_embed(model_channels, self.num_patches_side)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Timestep embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Class label embedding (optional)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        
        # Context embedding projection (for few-shot conditioning)
        if context_channels > 0 and mode_conditioning == "film":
            self.context_proj = nn.Linear(context_channels, time_embed_dim)
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=model_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                context_channels=context_channels if mode_conditioning == "lag" else 0,
                mode_conditioning=mode_conditioning,
            )
            for _ in range(self.depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(
            hidden_size=model_channels,
            patch_size=patch_size,
            out_channels=out_channels,
            context_channels=context_channels if mode_conditioning == "film" else 0,
            mode_conditioning=mode_conditioning
        )
    
    def convert_to_fp16(self):
        """Convert model to float16"""
        self.blocks.apply(lambda m: m.to(torch.float16) if isinstance(m, (nn.Linear, nn.LayerNorm)) else m)
        self.final_layer.apply(lambda m: m.to(torch.float16) if isinstance(m, (nn.Linear, nn.LayerNorm)) else m)
    
    def convert_to_fp32(self):
        """Convert model to float32"""
        self.blocks.apply(lambda m: m.to(torch.float32) if isinstance(m, (nn.Linear, nn.LayerNorm)) else m)
        self.final_layer.apply(lambda m: m.to(torch.float32) if isinstance(m, (nn.Linear, nn.LayerNorm)) else m)
    
    def forward(self, x, timesteps, c=None, y=None):
        """
        Forward pass compatible with UNetModel interface
        
        :param x: (B, C, H, W) input tensor
        :param timesteps: (B,) timestep tensor
        :param c: conditioning tensor
            - If mode_conditioning == "film": (B, context_channels) or None
            - If mode_conditioning == "lag": (B, N_c, context_channels) or None
        :param y: (B,) class labels (optional)
        :return: (B, C, H, W) output tensor
        """
        B, C, H, W = x.shape
        assert H == W == self.image_size, f"Input size {H}x{W} must match image_size {self.image_size}"
        
        # Handle padding for 28x28 images (like MNIST)
        pad = None
        if H == 28:
            pad = (2, 2, 2, 2)
            x = F.pad(x, pad)
            H = W = 32
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, hidden_size)
        x = x + self.pos_embed
        
        # Timestep embedding
        t_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_size))  # (B, time_embed_dim)
        
        # Class label embedding (if provided)
        if y is not None:
            assert self.num_classes is not None
            t_emb = t_emb + self.label_emb(y)
        
        # Combine time embedding with context for film conditioning
        if self.mode_conditioning == "film":
            if c is not None:
                # c: (B, context_channels)
                c_emb = self.context_proj(c)  # (B, time_embed_dim)
                conditioning = t_emb + c_emb  # (B, time_embed_dim)
            else:
                conditioning = t_emb
            # Project to hidden_size for adaLN-Zero
            conditioning = nn.Linear(t_emb.shape[-1], self.hidden_size)(conditioning)
        else:
            # For lag mode, use t_emb for adaLN and c separately for cross-attention
            conditioning = nn.Linear(t_emb.shape[-1], self.hidden_size)(t_emb)
        
        # Apply DiT blocks
        for block in self.blocks:
            if self.mode_conditioning == "lag" and c is not None:
                # Pass context to cross-attention
                x = block(x, c)
            else:
                x = block(x, conditioning)
        
        # Final layer
        if self.mode_conditioning == "film":
            x = self.final_layer(x, conditioning)
        else:
            x = self.final_layer(x, conditioning)
        
        # Reshape to image
        # x: (B, N, patch_size^2 * out_channels)
        x = x.reshape(B, self.num_patches_side, self.num_patches_side, 
                     self.patch_size, self.patch_size, self.out_channels)
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(B, self.out_channels, H, W)
        
        # Remove padding if added
        if pad:
            x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
        
        return x