"""
Script để convert weights từ PyTorch ViT sang JAX/Flax format
Dành cho model vit_jax.py

Usage:
    python convert_vit_pytorch_to_jax.py \\
        --pytorch_ckpt "vit_cifar10_patch4_input32 (1).pth" \\
        --output vit_cifar10_patch4_input32_jax.npz
"""

import argparse
import torch
import numpy as np
import jax.numpy as jnp
from flax import linen as nn
from flax.training import checkpoints
import pickle
from pathlib import Path


def inspect_pytorch_checkpoint(ckpt_path):
    """Kiểm tra cấu trúc của PyTorch checkpoint"""
    print(f"\n{'='*70}")
    print(f"Inspecting PyTorch checkpoint: {ckpt_path}")
    print(f"{'='*70}\n")
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Kiểm tra xem đây là state_dict hay dict chứa state_dict
    if isinstance(ckpt, dict):
        print("Checkpoint keys:", list(ckpt.keys()))
        
        # Thường thì weights nằm trong 'state_dict', 'model_state_dict', 'student', hoặc 'teacher'
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            print("\nFound 'state_dict' key")
        elif 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
            print("\nFound 'model_state_dict' key")
        elif 'student' in ckpt:
            state_dict = ckpt['student']
            print("\nFound 'student' key (using student model)")
        elif 'teacher' in ckpt:
            state_dict = ckpt['teacher']
            print("\nFound 'teacher' key (using teacher model)")
        else:
            # Có thể là state_dict trực tiếp
            state_dict = ckpt
            print("\nUsing checkpoint as state_dict directly")
    else:
        state_dict = ckpt
        print("\nCheckpoint is state_dict directly")
    
    # Kiểm tra xem state_dict có phải là dict/tensor hay không
    if isinstance(state_dict, dict):
        print(f"\nTotal parameters: {len(state_dict)}")
        print("\nParameter names:")
        for i, (key, value) in enumerate(state_dict.items()):
            if isinstance(value, torch.Tensor):
                print(f"  {i+1:3d}. {key:50s} shape: {tuple(value.shape)}")
            elif isinstance(value, dict):
                print(f"  {i+1:3d}. {key:50s} [nested dict with {len(value)} keys]")
            else:
                print(f"  {i+1:3d}. {key:50s} [type: {type(value).__name__}]")
            if i >= 20:  # Chỉ hiển thị 20 đầu tiên
                print(f"  ... ({len(state_dict) - 20} more)")
                break
    else:
        print(f"\nState dict type: {type(state_dict)}")
    
    return state_dict


def map_pytorch_to_jax_params(pytorch_state_dict):
    """
    Map PyTorch parameter names sang JAX/Flax parameter names
    
    PyTorch ViT structure (từ vit.py):
    - to_patch_embedding.0: Rearrange (không có params)
    - to_patch_embedding.1: nn.Linear(patch_dim, dim) -> weight, bias
    - to_time_embedding: nn.Linear(t_dim, dim) -> weight, bias
    - pos_embedding: nn.Parameter
    - cls_token: nn.Parameter
    - transformer.layers[i].0.norm: LayerNorm -> weight, bias
    - transformer.layers[i].0.fn.to_qkv: Linear -> weight
    - transformer.layers[i].0.fn.to_out[0]: Linear -> weight, bias
    - transformer.layers[i].1.norm: LayerNorm -> weight, bias
    - transformer.layers[i].1.fn.net[0]: Linear -> weight, bias
    - transformer.layers[i].1.fn.net[3]: Linear -> weight, bias
    - mlp_head[0]: LayerNorm -> weight, bias
    - mlp_head[1]: Linear -> weight, bias
    
    JAX/Flax structure (từ vit_jax.py):
    - to_patch_embedding: Dense -> kernel, bias
    - to_time_embedding: Dense -> kernel, bias
    - pos_embedding: param
    - cls_token: param
    - transformer.TransformerBlock_{i}.LayerNorm_0: scale, bias
    - transformer.TransformerBlock_{i}.SelfAttention_0: query, key, value, out (mỗi có kernel, bias)
    - transformer.TransformerBlock_{i}.LayerNorm_1: scale, bias
    - transformer.TransformerBlock_{i}.FeedForward_0.Dense_0: kernel, bias
    - transformer.TransformerBlock_{i}.FeedForward_0.Dense_1: kernel, bias
    - cls_norm: scale, bias
    - cls_dense: kernel, bias
    """
    
    jax_params = {}
    
    # 1. Patch embedding
    if 'to_patch_embedding.1.weight' in pytorch_state_dict:
        # PyTorch: (dim, patch_dim), JAX: (patch_dim, dim)
        jax_params['to_patch_embedding'] = {
            'kernel': pytorch_state_dict['to_patch_embedding.1.weight'].T.numpy(),
            'bias': pytorch_state_dict.get('to_patch_embedding.1.bias', torch.zeros(pytorch_state_dict['to_patch_embedding.1.weight'].shape[0])).numpy()
        }
    
    # 2. Time embedding
    if 'to_time_embedding.weight' in pytorch_state_dict:
        jax_params['to_time_embedding'] = {
            'kernel': pytorch_state_dict['to_time_embedding.weight'].T.numpy(),
            'bias': pytorch_state_dict.get('to_time_embedding.bias', torch.zeros(pytorch_state_dict['to_time_embedding.weight'].shape[0])).numpy()
        }
    
    # 3. Positional embedding
    if 'pos_embedding' in pytorch_state_dict:
        jax_params['pos_embedding'] = pytorch_state_dict['pos_embedding'].numpy()
    
    # 4. CLS token
    if 'cls_token' in pytorch_state_dict:
        jax_params['cls_token'] = pytorch_state_dict['cls_token'].numpy()
    
    # 5. Transformer blocks
    depth = 0
    while f'transformer.layers.{depth}.0.norm.weight' in pytorch_state_dict:
        block_name = f'TransformerBlock_{depth}'
        
        # LayerNorm 0 (pre-attention)
        jax_params[f'transformer.{block_name}.LayerNorm_0'] = {
            'scale': pytorch_state_dict[f'transformer.layers.{depth}.0.norm.weight'].numpy(),
            'bias': pytorch_state_dict[f'transformer.layers.{depth}.0.norm.bias'].numpy()
        }
        
        # SelfAttention
        # PyTorch: to_qkv là một Linear(dim, inner_dim*3) không bias
        # JAX: SelfAttention tách thành query, key, value, out
        qkv_weight = pytorch_state_dict[f'transformer.layers.{depth}.0.fn.to_qkv.weight']  # (inner_dim*3, dim)
        inner_dim = qkv_weight.shape[0] // 3
        dim = qkv_weight.shape[1]
        heads = inner_dim // 64  # Giả sử dim_head=64, có thể cần điều chỉnh
        
        # Tách qkv_weight thành q, k, v
        qkv_weight = qkv_weight.numpy()  # (inner_dim*3, dim)
        q_weight = qkv_weight[:inner_dim, :].T  # (dim, inner_dim)
        k_weight = qkv_weight[inner_dim:2*inner_dim, :].T  # (dim, inner_dim)
        v_weight = qkv_weight[2*inner_dim:, :].T  # (dim, inner_dim)
        
        # JAX SelfAttention: query, key, value, out
        # Flax SelfAttention có cấu trúc: (num_heads, dim_head, dim) cho qkv
        # Nhưng thực tế Flax dùng qkv_features = heads * dim_head
        # Cần reshape để khớp với Flax format
        dim_head = inner_dim // heads
        
        # Reshape cho Flax: (heads, dim_head, dim)
        q_weight_reshaped = q_weight.reshape(dim, heads, dim_head).transpose(1, 2, 0)  # (heads, dim_head, dim)
        k_weight_reshaped = k_weight.reshape(dim, heads, dim_head).transpose(1, 2, 0)
        v_weight_reshaped = v_weight.reshape(dim, heads, dim_head).transpose(1, 2, 0)
        
        jax_params[f'transformer.{block_name}.SelfAttention_0'] = {
            'query': {'kernel': q_weight_reshaped},
            'key': {'kernel': k_weight_reshaped},
            'value': {'kernel': v_weight_reshaped},
        }
        
        # Output projection
        if f'transformer.layers.{depth}.0.fn.to_out.0.weight' in pytorch_state_dict:
            out_weight = pytorch_state_dict[f'transformer.layers.{depth}.0.fn.to_out.0.weight']  # (dim, inner_dim)
            out_bias = pytorch_state_dict.get(f'transformer.layers.{depth}.0.fn.to_out.0.bias', None)
            
            # Reshape output: (dim, inner_dim) -> (heads, dim_head, dim)
            out_weight_reshaped = out_weight.T.numpy().reshape(inner_dim, dim).reshape(heads, dim_head, dim)
            
            jax_params[f'transformer.{block_name}.SelfAttention_0']['out'] = {
                'kernel': out_weight_reshaped,
            }
            if out_bias is not None:
                jax_params[f'transformer.{block_name}.SelfAttention_0']['out']['bias'] = out_bias.numpy()
        
        # LayerNorm 1 (pre-ff)
        jax_params[f'transformer.{block_name}.LayerNorm_1'] = {
            'scale': pytorch_state_dict[f'transformer.layers.{depth}.1.norm.weight'].numpy(),
            'bias': pytorch_state_dict[f'transformer.layers.{depth}.1.norm.bias'].numpy()
        }
        
        # FeedForward
        # PyTorch: net[0] (dim -> hidden_dim), net[3] (hidden_dim -> dim)
        ff_hidden_dim = pytorch_state_dict[f'transformer.layers.{depth}.1.fn.net.0.weight'].shape[0]
        
        jax_params[f'transformer.{block_name}.FeedForward_0.Dense_0'] = {
            'kernel': pytorch_state_dict[f'transformer.layers.{depth}.1.fn.net.0.weight'].T.numpy(),
            'bias': pytorch_state_dict[f'transformer.layers.{depth}.1.fn.net.0.bias'].numpy()
        }
        
        jax_params[f'transformer.{block_name}.FeedForward_0.Dense_1'] = {
            'kernel': pytorch_state_dict[f'transformer.layers.{depth}.1.fn.net.3.weight'].T.numpy(),
            'bias': pytorch_state_dict[f'transformer.layers.{depth}.1.fn.net.3.bias'].numpy()
        }
        
        depth += 1
    
    # 6. Classification head
    if 'mlp_head.0.weight' in pytorch_state_dict:
        jax_params['cls_norm'] = {
            'scale': pytorch_state_dict['mlp_head.0.weight'].numpy(),
            'bias': pytorch_state_dict['mlp_head.0.bias'].numpy()
        }
    
    if 'mlp_head.1.weight' in pytorch_state_dict:
        jax_params['cls_dense'] = {
            'kernel': pytorch_state_dict['mlp_head.1.weight'].T.numpy(),
            'bias': pytorch_state_dict['mlp_head.1.bias'].numpy()
        }
    
    return jax_params, depth


def convert_timm_vit_weights(pytorch_state_dict, depth, dim, heads, dim_head):
    """
    Convert timm-style ViT weights (module.backbone.*) to JAX format
    """
    jax_params = {}
    
    # 1. Patch embedding: Conv2d -> Linear
    # PyTorch: Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
    # Shape: (dim, 3, patch_h, patch_w) -> need to convert to Linear
    if 'module.backbone.patch_embed.proj.weight' in pytorch_state_dict:
        conv_weight = pytorch_state_dict['module.backbone.patch_embed.proj.weight']  # (dim, 3, patch_h, patch_w)
        conv_bias = pytorch_state_dict.get('module.backbone.patch_embed.proj.bias', None)
        
        # Unfold Conv2d to Linear: (dim, 3, patch_h, patch_w) -> (dim, 3*patch_h*patch_w)
        patch_dim = conv_weight.shape[1] * conv_weight.shape[2] * conv_weight.shape[3]
        linear_weight = conv_weight.reshape(dim, patch_dim)  # (dim, patch_dim)
        
        jax_params['to_patch_embedding'] = {
            'kernel': linear_weight.T.numpy(),  # (patch_dim, dim) for JAX
            'bias': conv_bias.numpy() if conv_bias is not None else np.zeros(dim, dtype=linear_weight.numpy().dtype)
        }
    
    # 2. CLS token
    if 'module.backbone.cls_token' in pytorch_state_dict:
        jax_params['cls_token'] = pytorch_state_dict['module.backbone.cls_token'].numpy()
    
    # 3. Positional embedding
    if 'module.backbone.pos_embed' in pytorch_state_dict:
        jax_params['pos_embedding'] = pytorch_state_dict['module.backbone.pos_embed'].numpy()
    
    # 4. Transformer blocks
    for d in range(depth):
        block_name = f'TransformerBlock_{d}'
        prefix = f'module.backbone.blocks.{d}'
        
        # LayerNorm 1 (pre-attention)
        jax_params[f'transformer.{block_name}.LayerNorm_0'] = {
            'scale': pytorch_state_dict[f'{prefix}.norm1.weight'].numpy(),
            'bias': pytorch_state_dict[f'{prefix}.norm1.bias'].numpy()
        }
        
        # Attention: qkv với bias
        qkv_weight = pytorch_state_dict[f'{prefix}.attn.qkv.weight']  # (inner_dim*3, dim)
        qkv_bias = pytorch_state_dict.get(f'{prefix}.attn.qkv.bias', None)  # (inner_dim*3,)
        inner_dim = qkv_weight.shape[0] // 3
        
        # Tách qkv
        qkv_weight_np = qkv_weight.numpy()  # (inner_dim*3, dim)
        q_weight = qkv_weight_np[:inner_dim, :]  # (inner_dim, dim)
        k_weight = qkv_weight_np[inner_dim:2*inner_dim, :]  # (inner_dim, dim)
        v_weight = qkv_weight_np[2*inner_dim:, :]  # (inner_dim, dim)
        
        # Reshape cho Flax: (dim, heads, dim_head)
        q_weight_reshaped = q_weight.T.reshape(dim, heads, dim_head)
        k_weight_reshaped = k_weight.T.reshape(dim, heads, dim_head)
        v_weight_reshaped = v_weight.T.reshape(dim, heads, dim_head)
        
        # Handle bias
        if qkv_bias is not None:
            qkv_bias_np = qkv_bias.numpy()
            q_bias = qkv_bias_np[:inner_dim].reshape(heads, dim_head)
            k_bias = qkv_bias_np[inner_dim:2*inner_dim].reshape(heads, dim_head)
            v_bias = qkv_bias_np[2*inner_dim:].reshape(heads, dim_head)
        else:
            q_bias = np.zeros((heads, dim_head), dtype=q_weight_reshaped.dtype)
            k_bias = np.zeros((heads, dim_head), dtype=k_weight_reshaped.dtype)
            v_bias = np.zeros((heads, dim_head), dtype=v_weight_reshaped.dtype)
        
        jax_params[f'transformer.{block_name}.SelfAttention_0'] = {
            'query': {'kernel': q_weight_reshaped, 'bias': q_bias},
            'key': {'kernel': k_weight_reshaped, 'bias': k_bias},
            'value': {'kernel': v_weight_reshaped, 'bias': v_bias},
        }
        
        # Output projection
        proj_weight = pytorch_state_dict[f'{prefix}.attn.proj.weight']  # (dim, dim)
        proj_bias = pytorch_state_dict.get(f'{prefix}.attn.proj.bias', None)
        
        # Reshape: (dim, dim) -> (dim, heads, dim_head) -> transpose -> (heads, dim_head, dim)
        out_weight_reshaped = proj_weight.numpy().reshape(dim, heads, dim_head).transpose(1, 2, 0)
        
        jax_params[f'transformer.{block_name}.SelfAttention_0']['out'] = {
            'kernel': out_weight_reshaped,
            'bias': proj_bias.numpy() if proj_bias is not None else np.zeros(dim, dtype=out_weight_reshaped.dtype)
        }
        
        # LayerNorm 2 (pre-ff)
        jax_params[f'transformer.{block_name}.LayerNorm_1'] = {
            'scale': pytorch_state_dict[f'{prefix}.norm2.weight'].numpy(),
            'bias': pytorch_state_dict[f'{prefix}.norm2.bias'].numpy()
        }
        
        # FeedForward
        jax_params[f'transformer.{block_name}.FeedForward_0.Dense_0'] = {
            'kernel': pytorch_state_dict[f'{prefix}.mlp.fc1.weight'].T.numpy(),
            'bias': pytorch_state_dict[f'{prefix}.mlp.fc1.bias'].numpy()
        }
        
        jax_params[f'transformer.{block_name}.FeedForward_0.Dense_1'] = {
            'kernel': pytorch_state_dict[f'{prefix}.mlp.fc2.weight'].T.numpy(),
            'bias': pytorch_state_dict[f'{prefix}.mlp.fc2.bias'].numpy()
        }
    
    return jax_params


def convert_attention_weights_simple(pytorch_state_dict, depth, dim, heads, dim_head):
    """
    Convert attention weights với cách tiếp cận đơn giản hơn
    Flax SelfAttention sử dụng MultiHeadDotProductAttention với cấu trúc khác
    """
    jax_params = {}
    
    for d in range(depth):
        block_name = f'TransformerBlock_{d}'
        
        # LayerNorm 0
        jax_params[f'transformer.{block_name}.LayerNorm_0'] = {
            'scale': pytorch_state_dict[f'transformer.layers.{d}.0.norm.weight'].numpy(),
            'bias': pytorch_state_dict[f'transformer.layers.{d}.0.norm.bias'].numpy()
        }
        
        # Attention: Flax SelfAttention sử dụng qkv_features = heads * dim_head
        # và tự động tách thành q, k, v
        qkv_weight = pytorch_state_dict[f'transformer.layers.{d}.0.fn.to_qkv.weight']  # (inner_dim*3, dim)
        inner_dim = qkv_weight.shape[0] // 3
        
        # Flax SelfAttention cần: (dim, qkv_features) với qkv_features = heads * dim_head
        # Nhưng thực tế Flax dùng MultiHeadDotProductAttention với format khác
        # Cần kiểm tra lại cấu trúc của Flax SelfAttention
        
        # Tạm thời: Flax SelfAttention có thể nhận trực tiếp qkv_features
        # và tự động reshape. Nhưng cần đảm bảo format đúng
        
        # Cách đơn giản: giữ nguyên format (inner_dim*3, dim) và để Flax xử lý
        # Nhưng Flax không có cơ chế này, nên cần reshape
        
        # Kiểm tra: Flax SelfAttention thực tế dùng MultiHeadDotProductAttention
        # với kernel shape: (..., features, num_heads, head_dim)
        # Hoặc (features, num_heads, head_dim) cho query/key/value riêng
        
        # Tách qkv từ PyTorch: (inner_dim*3, dim)
        qkv_weight_np = qkv_weight.numpy()  # (inner_dim*3, dim)
        q_weight = qkv_weight_np[:inner_dim, :]  # (inner_dim, dim)
        k_weight = qkv_weight_np[inner_dim:2*inner_dim, :]  # (inner_dim, dim)
        v_weight = qkv_weight_np[2*inner_dim:, :]  # (inner_dim, dim)
        
        # PyTorch: (inner_dim, dim) -> transpose -> (dim, inner_dim)
        # Reshape: (dim, inner_dim) -> (dim, heads, dim_head)
        # inner_dim = heads * dim_head
        q_weight_reshaped = q_weight.T.reshape(dim, heads, dim_head)  # (dim, heads, dim_head)
        k_weight_reshaped = k_weight.T.reshape(dim, heads, dim_head)  # (dim, heads, dim_head)
        v_weight_reshaped = v_weight.T.reshape(dim, heads, dim_head)  # (dim, heads, dim_head)
        
        # Flax SelfAttention format:
        # query/key/value kernels: (features, num_heads, head_dim) = (dim, heads, dim_head)
        # query/key/value biases: (num_heads, head_dim) = (heads, dim_head) - PyTorch không có bias cho qkv
        jax_params[f'transformer.{block_name}.SelfAttention_0'] = {
            'query': {
                'kernel': q_weight_reshaped,
                'bias': np.zeros((heads, dim_head), dtype=q_weight_reshaped.dtype)
            },
            'key': {
                'kernel': k_weight_reshaped,
                'bias': np.zeros((heads, dim_head), dtype=k_weight_reshaped.dtype)
            },
            'value': {
                'kernel': v_weight_reshaped,
                'bias': np.zeros((heads, dim_head), dtype=v_weight_reshaped.dtype)
            },
        }
        
        # Output projection
        # PyTorch: (dim, inner_dim)
        # Flax: out kernel (num_heads, head_dim, features) = (heads, dim_head, dim)
        if f'transformer.layers.{d}.0.fn.to_out.0.weight' in pytorch_state_dict:
            out_weight = pytorch_state_dict[f'transformer.layers.{d}.0.fn.to_out.0.weight']  # (dim, inner_dim)
            out_bias = pytorch_state_dict.get(f'transformer.layers.{d}.0.fn.to_out.0.bias', None)
            
            # Reshape: (dim, inner_dim) -> (dim, heads, dim_head) -> transpose -> (heads, dim_head, dim)
            out_weight_reshaped = out_weight.numpy().reshape(dim, heads, dim_head).transpose(1, 2, 0)  # (heads, dim_head, dim)
            
            jax_params[f'transformer.{block_name}.SelfAttention_0']['out'] = {
                'kernel': out_weight_reshaped,
            }
            if out_bias is not None:
                jax_params[f'transformer.{block_name}.SelfAttention_0']['out']['bias'] = out_bias.numpy()
            else:
                jax_params[f'transformer.{block_name}.SelfAttention_0']['out']['bias'] = np.zeros(dim, dtype=out_weight_reshaped.dtype)
        
        # LayerNorm 1
        jax_params[f'transformer.{block_name}.LayerNorm_1'] = {
            'scale': pytorch_state_dict[f'transformer.layers.{d}.1.norm.weight'].numpy(),
            'bias': pytorch_state_dict[f'transformer.layers.{d}.1.norm.bias'].numpy()
        }
        
        # FeedForward
        jax_params[f'transformer.{block_name}.FeedForward_0.Dense_0'] = {
            'kernel': pytorch_state_dict[f'transformer.layers.{d}.1.fn.net.0.weight'].T.numpy(),
            'bias': pytorch_state_dict[f'transformer.layers.{d}.1.fn.net.0.bias'].numpy()
        }
        
        jax_params[f'transformer.{block_name}.FeedForward_0.Dense_1'] = {
            'kernel': pytorch_state_dict[f'transformer.layers.{d}.1.fn.net.3.weight'].T.numpy(),
            'bias': pytorch_state_dict[f'transformer.layers.{d}.1.fn.net.3.bias'].numpy()
        }
    
    return jax_params


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch ViT weights to JAX/Flax format')
    parser.add_argument('--pytorch_ckpt', type=str, required=True,
                        help='Path to PyTorch checkpoint file (.pth)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for JAX checkpoint (.npz or directory)')
    parser.add_argument('--inspect_only', action='store_true',
                        help='Only inspect the checkpoint, do not convert')
    
    args = parser.parse_args()
    
    # Inspect checkpoint
    pytorch_state_dict = inspect_pytorch_checkpoint(args.pytorch_ckpt)
    
    if args.inspect_only:
        print("\nInspection complete. Exiting.")
        return
    
    # Infer architecture from checkpoint
    # Kiểm tra xem đây là timm-style hay custom ViT
    is_timm_style = False
    if 'module.backbone.blocks.0.norm1.weight' in pytorch_state_dict:
        is_timm_style = True
        print("Detected timm-style ViT architecture")
    elif 'transformer.layers.0.0.norm.weight' in pytorch_state_dict:
        is_timm_style = False
        print("Detected custom ViT architecture")
    else:
        raise ValueError("Cannot determine ViT architecture from checkpoint")
    
    # Tìm depth
    depth = 0
    if is_timm_style:
        while f'module.backbone.blocks.{depth}.norm1.weight' in pytorch_state_dict:
            depth += 1
    else:
        while f'transformer.layers.{depth}.0.norm.weight' in pytorch_state_dict:
            depth += 1
    
    # Tìm dim từ cls_token hoặc pos_embedding
    if is_timm_style:
        if 'module.backbone.cls_token' in pytorch_state_dict:
            dim = pytorch_state_dict['module.backbone.cls_token'].shape[-1]
        elif 'module.backbone.pos_embed' in pytorch_state_dict:
            dim = pytorch_state_dict['module.backbone.pos_embed'].shape[-1]
        else:
            raise ValueError("Cannot infer 'dim' from checkpoint")
    else:
        if 'cls_token' in pytorch_state_dict:
            dim = pytorch_state_dict['cls_token'].shape[-1]
        elif 'pos_embedding' in pytorch_state_dict:
            dim = pytorch_state_dict['pos_embedding'].shape[-1]
        else:
            raise ValueError("Cannot infer 'dim' from checkpoint")
    
    # Tìm heads và dim_head từ attention weights
    if depth > 0:
        if is_timm_style:
            qkv_key = 'module.backbone.blocks.0.attn.qkv.weight'
        else:
            qkv_key = 'transformer.layers.0.0.fn.to_qkv.weight'
        
        if qkv_key in pytorch_state_dict:
            qkv_weight = pytorch_state_dict[qkv_key]
            inner_dim = qkv_weight.shape[0] // 3
            # Giả sử dim_head = 64 (thường dùng)
            dim_head = 64
            heads = inner_dim // dim_head
            if heads * dim_head != inner_dim:
                # Thử các giá trị dim_head khác
                for dh in [32, 64, 128]:
                    if inner_dim % dh == 0:
                        dim_head = dh
                        heads = inner_dim // dh
                        break
        else:
            heads = 12  # default
            dim_head = 64  # default
    else:
        heads = 12  # default
        dim_head = 64  # default
    
    print(f"\n{'='*70}")
    print("Inferred Architecture:")
    print(f"  Style: {'timm' if is_timm_style else 'custom'}")
    print(f"  Depth: {depth}")
    print(f"  Dim: {dim}")
    print(f"  Heads: {heads}")
    print(f"  Dim head: {dim_head}")
    print(f"{'='*70}\n")
    
    # Convert weights
    print("Converting weights...")
    if is_timm_style:
        jax_params = convert_timm_vit_weights(pytorch_state_dict, depth, dim, heads, dim_head)
        
        # Add time embedding if exists (might not be in timm ViT)
        if 'module.backbone.to_time_embedding.weight' in pytorch_state_dict:
            jax_params['to_time_embedding'] = {
                'kernel': pytorch_state_dict['module.backbone.to_time_embedding.weight'].T.numpy(),
                'bias': pytorch_state_dict.get('module.backbone.to_time_embedding.bias', torch.zeros(pytorch_state_dict['module.backbone.to_time_embedding.weight'].shape[0])).numpy()
            }
        else:
            # Create default time embedding layer (t_dim -> dim)
            t_dim = 256  # default
            jax_params['to_time_embedding'] = {
                'kernel': np.random.randn(t_dim, dim).astype(np.float32) * 0.02,  # small random init
                'bias': np.zeros(dim, dtype=np.float32)
            }
            print("Warning: No time embedding found, initialized with random weights")
        
        # Add classification head if exists
        if 'module.head.weight' in pytorch_state_dict:
            num_classes = pytorch_state_dict['module.head.weight'].shape[0]
            jax_params['cls_norm'] = {
                'scale': np.ones(dim, dtype=np.float32),
                'bias': np.zeros(dim, dtype=np.float32)
            }
            jax_params['cls_dense'] = {
                'kernel': pytorch_state_dict['module.head.weight'].T.numpy(),
                'bias': pytorch_state_dict.get('module.head.bias', torch.zeros(num_classes)).numpy()
            }
        else:
            # Create default classification head
            num_classes = 10  # CIFAR-10 default
            jax_params['cls_norm'] = {
                'scale': np.ones(dim, dtype=np.float32),
                'bias': np.zeros(dim, dtype=np.float32)
            }
            jax_params['cls_dense'] = {
                'kernel': np.random.randn(dim, num_classes).astype(np.float32) * 0.02,
                'bias': np.zeros(num_classes, dtype=np.float32)
            }
            print("Warning: No classification head found, initialized with random weights")
    else:
        jax_params = convert_attention_weights_simple(pytorch_state_dict, depth, dim, heads, dim_head)
        
        # Add other params for custom ViT
        if 'to_patch_embedding.1.weight' in pytorch_state_dict:
            jax_params['to_patch_embedding'] = {
                'kernel': pytorch_state_dict['to_patch_embedding.1.weight'].T.numpy(),
                'bias': pytorch_state_dict.get('to_patch_embedding.1.bias', torch.zeros(pytorch_state_dict['to_patch_embedding.1.weight'].shape[0])).numpy()
            }
        
        if 'to_time_embedding.weight' in pytorch_state_dict:
            jax_params['to_time_embedding'] = {
                'kernel': pytorch_state_dict['to_time_embedding.weight'].T.numpy(),
                'bias': pytorch_state_dict.get('to_time_embedding.bias', torch.zeros(pytorch_state_dict['to_time_embedding.weight'].shape[0])).numpy()
            }
        
        if 'pos_embedding' in pytorch_state_dict:
            jax_params['pos_embedding'] = pytorch_state_dict['pos_embedding'].numpy()
        
        if 'cls_token' in pytorch_state_dict:
            jax_params['cls_token'] = pytorch_state_dict['cls_token'].numpy()
        
        if 'mlp_head.0.weight' in pytorch_state_dict:
            jax_params['cls_norm'] = {
                'scale': pytorch_state_dict['mlp_head.0.weight'].numpy(),
                'bias': pytorch_state_dict['mlp_head.0.bias'].numpy()
            }
        
        if 'mlp_head.1.weight' in pytorch_state_dict:
            jax_params['cls_dense'] = {
                'kernel': pytorch_state_dict['mlp_head.1.weight'].T.numpy(),
                'bias': pytorch_state_dict['mlp_head.1.bias'].numpy()
            }
    
    # Save
    output_path = args.output or args.pytorch_ckpt.replace('.pth', '_jax.npz')
    print(f"\nSaving converted weights to: {output_path}")
    
    # Flatten nested dict để lưu npz
    def flatten_dict(d, parent_key='', sep='/'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    flat_params = flatten_dict(jax_params)
    np.savez(output_path, **flat_params)
    
    print(f"✅ Conversion complete!")
    print(f"\nTo load in JAX:")
    print(f"  from convert_vit_pytorch_to_jax import load_jax_weights")
    print(f"  params = load_jax_weights('{output_path}')")
    print(f"  # Then use params with your Flax ViT model")


def unflatten_dict(flat_dict, sep='/'):
    """Reconstruct nested dict from flattened dict"""
    result = {}
    for key, value in flat_dict.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def load_jax_weights(npz_path):
    """
    Load converted JAX weights from npz file and return nested dict structure
    compatible with Flax model parameters.
    
    Usage:
        params = load_jax_weights('vit_cifar10_patch4_input32_jax.npz')
        # Then use with model.apply(params, ...)
    """
    data = np.load(npz_path, allow_pickle=True)
    flat_dict = {k: v for k, v in data.items()}
    nested_dict = unflatten_dict(flat_dict)
    
    # Convert to proper Flax params structure
    # Flax expects params to be nested under 'params' key
    if 'params' not in nested_dict:
        # If not already nested, wrap it
        return {'params': nested_dict}
    return nested_dict


if __name__ == '__main__':
    main()

