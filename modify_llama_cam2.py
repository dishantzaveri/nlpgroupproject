import math
import torch
from torch import nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaRotaryEmbedding, eager_attention_forward
from transformers.models.llama.modeling_llama import LlamaAttention
from typing import Callable, List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

__all__ = ['convert_kvcache_llama_cam', 'LlamaAttention_cam']

def local_cam_mask(value_states, attn_weights, start_budget, recent_budget):
    seq_length = attn_weights.shape[-1]
    for token_index in range(start_budget + recent_budget, seq_length):
        if torch.isnan(attn_weights).any():
            print(f"🚨 NaNs detected in attn_weights at token {token_index}")
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        if torch.isinf(attn_weights).any():
            print(f"🚨 Infs detected in attn_weights at token {token_index}")
            attn_weights = torch.clamp(attn_weights, min=-1e4, max=1e4)
        attn_score = torch.mean(attn_weights[:, :, :token_index, :token_index], dim=-2)
        
        eps = 1e-6  # small epsilon to avoid division by zero
        mean_attn = torch.max(torch.cat(
            (attn_score[0,:,:start_budget], attn_score[0,:,token_index-recent_budget:token_index]), dim=-1
        ), dim=-1)[0].clamp(min=eps)
        if torch.any(mean_attn == 0):
            print("⚠️ Zero detected in mean_attn! Skipping this token.")

        if torch.any(torch.isnan(attn_score)):
            print("NaNs detected in attn score.")
        merge_prob = (attn_score[0,:,token_index-recent_budget] / mean_attn).clamp(min=0.0, max=1.0)
        merge_mask = torch.bernoulli(merge_prob)
        score1 = value_states[:, :, token_index - recent_budget, ...].clone() * merge_mask.unsqueeze(-1) / recent_budget
        value_states[:, :, token_index - recent_budget + 1:token_index - recent_budget + recent_budget + 1, :] += score1.unsqueeze(2)
    return value_states

class LlamaAttention_cam(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.start_budget_ratio = getattr(config, "start_ratio", 0.3)
        self.recent_budget_ratio = getattr(config, "recent_ratio", 0.2)
        self.merge_token = getattr(config, "merge_token", True)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        print("Hi! Attention cam layer", self.layer_idx)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
    # === 🔁 CAM-specific logic inserted here ===

        start_budget = math.ceil(self.start_budget_ratio * attn_weights.shape[-1])
        recent_budget = math.ceil(self.recent_budget_ratio * attn_weights.shape[-1])

        ones = torch.ones_like(attn_weights, dtype=torch.bool)
        ones = torch.triu(ones, diagonal=-recent_budget)
        mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
        mask_bottom[:, :, :, :start_budget] = True
        mask_bottom = torch.logical_or(mask_bottom, ones)
        mask_bottom = torch.tril(mask_bottom, diagonal=0)

        attn_weights[~mask_bottom] = torch.min(attn_weights)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        if self.merge_token:
            value_states = local_cam_mask(value_states, attn_weights, start_budget, recent_budget)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        # === 🔁 CAM logic ends here ===

        return attn_output, attn_weights

def convert_kvcache_llama_cam(model, config, layer_idx=0):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # Recurse into children, incrementing layer index when appropriate
            model._modules[name], layer_idx = convert_kvcache_llama_cam(module, config, layer_idx)

        if isinstance(module, LlamaAttention):
            # Replace LlamaAttention with CAM version
            cam_module = LlamaAttention_cam(config, layer_idx=layer_idx)
            model._modules[name] = cam_module
            layer_idx += 1  # Increment after assigning

            # Move to same device
            target_device = next(module.parameters()).device
            for param in cam_module.parameters():
                param.data = param.data.to(target_device)
            for buffer in cam_module.buffers():
                buffer.data = buffer.data.to(target_device)

            # Convert to half precision
            model._modules[name].half()

    return model, layer_idx