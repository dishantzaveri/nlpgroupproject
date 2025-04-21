import torch
from torch import nn

import math
from typing import Optional, Tuple
import torch
from torch import nn
import math

from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Config, Qwen2RotaryEmbedding

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from transformers.cache_utils import Cache 
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config 
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
from typing import Optional, Tuple
import copy


def local_cam_mask(value_states,attn_score,start_budget,recent_budget):
    merge_budget = 32
    #(1,nhead,k-token-1)
    token_index=attn_score.shape[-1]
    attn_score = attn_score.squeeze(0).squeeze(1)
    mean_attn=torch.mean(attn_score[:,token_index-recent_budget+2:token_index+1],dim=-1)
    merge_prob=attn_score[:,token_index-recent_budget+1]/mean_attn
    if torch.isnan(merge_prob).any(): merge_prob[torch.isnan(merge_prob)] = 0
    if torch.isinf(merge_prob).any(): merge_prob[torch.isnan(merge_prob)] = 1
    merge_mask = torch.bernoulli(merge_prob.clamp(min=0,max=1))
    score1=value_states[:,:,token_index-recent_budget+1,...].clone()*merge_mask.unsqueeze(-1)/merge_budget
    value_states[:,:,token_index-recent_budget+2:token_index-recent_budget+merge_budget+2,:]+=score1.unsqueeze(2)
    return value_states


class Qwen2AttentionCam(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.max_position_embeddings = config.max_position_embeddings

        self.rotary_emb = Qwen2RotaryEmbedding(config)

        self.start_budget_ratio = config.start_ratio
        self.recent_budget_ratio = config.recent_ratio
        self.merge_token = config.merge_token

        self.start_budget_ratio = config.start_ratio
        self.recent_budget_ratio = config.recent_ratio
        self.attention_masks_next = None 
        self.start_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None
        self.layer_index = None
        self.input_length = []
        self.cache_budget_records = []
    
    def _reset_masks(self):
        self.attention_masks_next = None 
        self.start_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache = False,
        output_attentions=False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
            # key_states = torch.cat([past_key_value[0], key_states], dim=2)
            # value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # past_key_value = (key_states, value_states) if use_cache else None

        key = repeat_kv(key_states, self.num_key_value_groups)
        if value_states.shape[1]<query_states.shape[1]:
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        
        if self.previous_scores == None:
            self.start_budget = int(self.start_budget_ratio * attn_weights.shape[-1])
            self.recent_budget = int(self.recent_budget_ratio * attn_weights.shape[-1])
            self.cache_budget = self.start_budget + self.recent_budget
            self.cache_budget_records.append(self.cache_budget)
            self.input_length.append(attn_weights.shape[-1])
            self.previous_scores = 1
        

        if self.attention_masks_next is not None:
            attn_weights = attn_weights * self.attention_masks_next + (1 - self.attention_masks_next) * torch.finfo(attn_weights.dtype).min
            
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        dtype_attn_weights = attn_weights.dtype
        attn_weights_devices = attn_weights.device
        attn_mask = torch.zeros(attn_weights.shape[1], attn_weights.shape[-1]+1).to(dtype_attn_weights).to(attn_weights_devices)

        if attn_weights.shape[-1] > self.cache_budget:
            # activate most recent k-cache
            if not self.recent_budget == 0:
                attn_mask[:, attn_weights.shape[-1]+1-self.recent_budget:attn_weights.shape[-1]+1] = 1
            if not self.start_budget == 0:
                attn_mask[:,:self.start_budget]=1

        attn_output = torch.matmul(attn_weights, value_states)

        if self.attention_masks_next is not None:
            value_states = local_cam_mask(value_states,attn_weights,self.start_budget,self.recent_budget)
            past_key_value=(key_states,value_states) if use_cache else None

        self.attention_masks_next = attn_mask.unsqueeze(0).unsqueeze(2)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights


def convert_kvcache_qwen_cam(model, config, layer_idx=None):
    for name, module in reversed(model._modules.items()):
        if name.isnumeric():
            layer_idx = int(name)
        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_qwen_cam(module, config, layer_idx)

        if isinstance(module, Qwen2Attention):
            orig_state_dict = copy.deepcopy(module.state_dict())
            model._modules[name] = Qwen2AttentionCam(config, layer_idx=layer_idx)
            
            target_device = next(module.parameters()).device
            for param in model._modules[name].parameters():
                param.data = param.data.to(target_device)
            for buffer in model._modules[name].buffers():
                buffer.data = buffer.data.to(target_device)
            model._modules[name].half()
            model._modules[name].load_state_dict(orig_state_dict)
    return model
