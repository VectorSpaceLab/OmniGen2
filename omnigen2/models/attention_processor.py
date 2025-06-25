"""
OmniGen2 Attention Processor Module

Copyright 2025 BAAI, The OmniGen2 Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import warnings
import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from einops import repeat



from diffusers.models.attention_processor import Attention
from .embeddings import apply_rotary_emb




class OmniGen2AttnProcessor:
    """
    Processor for implementing scaled dot-product attention compatible with MPS and CUDA.
    
    This processor is optimized for PyTorch 2.0 and implements:
    - Scaled dot-product attention (compatible with MPS)
    - Rotary position embeddings (RoPE)
    - Query-Key normalization
    - Proportional attention scaling
    
    Args:
        None
        
    Raises:
        ImportError: If PyTorch version is less than 2.0
    """

    def __init__(self) -> None:
        """Initialize the attention processor."""
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "OmniGen2AttnProcessor requires PyTorch 2.0. "
                "Please upgrade PyTorch to version 2.0 or later."
            )
    

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        base_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Process attention computation with scaled dot-product attention.

        Args:
            attn: Attention module
            hidden_states: Hidden states tensor of shape (batch_size, seq_len, hidden_dim)
            encoder_hidden_states: Encoder hidden states tensor
            attention_mask: Optional attention mask tensor
            image_rotary_emb: Optional rotary embeddings for image tokens
            base_sequence_length: Optional base sequence length for proportional attention

        Returns:
            torch.Tensor: Processed hidden states after attention computation
        """
        batch_size, sequence_length, _ = hidden_states.shape

        # Get Query-Key-Value Pair
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        # Reshape tensors for attention computation
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # Apply Query-Key normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply Rotary Position Embeddings
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=False)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        # Calculate attention scale
        if base_sequence_length is not None:
            softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
        else:
            softmax_scale = attn.scale

        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            # Handle different mask shapes properly
            if attention_mask.dim() == 2:  # (batch, seq_len)
                attention_mask = attention_mask.view(batch_size, 1, 1, -1)
            elif attention_mask.dim() == 3:  # (batch, seq_len, seq_len) 
                attention_mask = attention_mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)
            elif attention_mask.dim() == 4:  # Already correct shape
                pass
            else:
                # Flatten and reshape for unexpected dimensions
                attention_mask = attention_mask.view(batch_size, 1, 1, -1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # explicitly repeat key and value to match query length, otherwise using enable_gqa=True results in MATH backend of sdpa in our test of pytorch2.6
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)
        
        # Fix attention mask to match actual query/key sequence lengths after processing
        if attention_mask is not None:
            actual_query_seq_len = query.size(-2)
            actual_key_seq_len = key.size(-2)
            
            # Check if mask dimensions are compatible with MPS
            if query.device.type == 'mps':
                # On MPS, disable mask if dimensions don't match to avoid crashes
                if (attention_mask.size(-1) != actual_key_seq_len or 
                    attention_mask.size(-2) != actual_query_seq_len):
                    print(f"Warning: Attention mask shape mismatch on MPS, disabling mask for stability")
                    print(f"  Mask shape: {attention_mask.shape}, Expected: (*, *, {actual_query_seq_len}, {actual_key_seq_len})")
                    attention_mask = None

        # Validate tensor shapes for MPS compatibility
        if query.numel() == 0 or key.numel() == 0 or value.numel() == 0:
            # Handle empty tensors - create zero output with correct shape
            hidden_states = torch.zeros(
                (batch_size, sequence_length, attn.heads * head_dim),
                dtype=query.dtype,
                device=query.device
            )
        else:
            # Check for MPS and use fallback for problematic cases
            if query.device.type == 'mps' and (query.size(-2) == 0 or key.size(-2) == 0):
                # Use manual attention computation for MPS when shapes are problematic
                scores = torch.matmul(query, key.transpose(-2, -1)) * softmax_scale
                if attention_mask is not None:
                    scores = scores + attention_mask
                attn_weights = F.softmax(scores, dim=-1)
                hidden_states = torch.matmul(attn_weights, value)
            else:
                try:
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value, attn_mask=attention_mask, scale=softmax_scale
                    )
                except RuntimeError as e:
                    if "INTERNAL ASSERT FAILED" in str(e) and query.device.type == 'mps':
                        # Fallback to manual attention for MPS errors
                        try:
                            scores = torch.matmul(query, key.transpose(-2, -1)) * softmax_scale
                            if attention_mask is not None:
                                scores = scores + attention_mask
                            attn_weights = F.softmax(scores, dim=-1)
                            hidden_states = torch.matmul(attn_weights, value)
                        except RuntimeError as e2:
                            # If manual attention also fails on MPS, try CPU fallback
                            print(f"MPS attention failed, falling back to CPU for this operation")
                            query_cpu = query.cpu()
                            key_cpu = key.cpu()
                            value_cpu = value.cpu()
                            mask_cpu = attention_mask.cpu() if attention_mask is not None else None
                            
                            scores = torch.matmul(query_cpu, key_cpu.transpose(-2, -1)) * softmax_scale
                            if mask_cpu is not None:
                                scores = scores + mask_cpu
                            attn_weights = F.softmax(scores, dim=-1)
                            hidden_states = torch.matmul(attn_weights, value_cpu).to(query.device)
                    else:
                        raise e
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.type_as(query)

        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states