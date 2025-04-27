# utils_lm_eval/modify_llama_cam.py

import math
import torch
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    LlamaAttention,
    apply_rotary_pos_emb,
)

__all__ = ["convert_kvcache_llama_cam", "LlamaAttention_cam"]

def local_cam_mask(value_states: torch.Tensor, A_bar: torch.Tensor, start_budget: int, recent_budget: int) -> torch.Tensor:
    """
    Implements CaM merging over a local window of `recent_budget` tokens:
      - value_states: [B, H, K, D]
      - A_bar: cumulative attention scores [B, H, K]
    Returns a new V tensor with the evicted slot merged into the suffix window.
    """
    B, H, K, D = value_states.shape
    m = recent_budget
    # No merge if window invalid
    if m <= 0 or m >= K:
        return value_states

    # Eviction index: the slot just before the kept suffix
    evict_idx = K - m - 1
    evict_idx = max(0, min(evict_idx, K - 1))

    # Suffix window to merge into: indices [K-m .. K-1]
    window_start = max(0, K - m)
    window_end = K

    # Compute average attention over the window
    window = A_bar[..., window_start:window_end]  # [B, H, m]
    avg_window = window.mean(dim=-1).clamp_min(1e-6)  # [B, H]

    # Attention on the evicted slot
    attn_e = A_bar[..., evict_idx]  # [B, H]

    # Probabilistic merge mask
    prob = (attn_e / avg_window).clamp(0.0, 1.0)
    prob = torch.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
    mask = torch.bernoulli(prob)  # [B, H]

    # Gather and scale the evicted V
    v_e = value_states[..., evict_idx, :].clone()  # [B, H, D]
    v_e = v_e * mask.unsqueeze(-1) / m              # [B, H, D]

    # Out-of-place merge into each of the m suffix slots
    V_new = value_states.clone()
    V_new[..., window_start:window_end, :] += v_e.unsqueeze(-2)

    return V_new


class LlamaAttention_cam(nn.Module):
    """
    LLaMA Multi-Head Attention with Cache Merging (CaM):
      - Accumulates past attention scores (A_bar)
      - Merges evicted cache entries into a local suffix window
      - Enforces a keep-mask of start & recent budgets on the next step
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Rotary embeddings
        self.rotary_emb = LlamaRotaryEmbedding(config)

        # CaM hyperparameters
        self.start_ratio = getattr(config, "start_ratio", 0.0)
        self.recent_ratio = getattr(config, "recent_ratio", 0.0)

        # State
        self.A_bar = None
        self.attention_masks_next = None

    def _shape(self, x: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_value: tuple = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ):
        # Sanitizing inputs
        hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e4, neginf=-1e4)
        B, T, _ = hidden_states.shape

        # QKV projections
        Q = self._shape(self.q_proj(hidden_states), T, B)
        K = self._shape(self.k_proj(hidden_states), T, B)
        V = self._shape(self.v_proj(hidden_states), T, B)

        # Rotary positional embeddings
        kv_len = K.size(-2)
        if position_ids is None:
            pos_ids = torch.arange(kv_len, device=K.device)
            pos_ids = pos_ids.unsqueeze(0).expand(B, kv_len)
        else:
            pos_ids = position_ids
        cos, sin = self.rotary_emb(V, pos_ids)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # Appending past cache only when it's really a (K,V) tuple
        if isinstance(past_key_value, tuple) and len(past_key_value) == 2:
            K = torch.cat([past_key_value[0], K], dim=2)
            V = torch.cat([past_key_value[1], V], dim=2)

        # Raw attention scores
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask

        # Applying previous-step keep-mask
        if self.attention_masks_next is not None:
            MIN = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~self.attention_masks_next, MIN)

        # Softmax
        MIN = torch.finfo(scores.dtype).min
        scores = torch.nan_to_num(scores, nan=MIN, posinf=MIN, neginf=MIN)
        attn_weights = torch.softmax(scores, dim=-1)

        # Accumulating last-token attention (pad/trim A_bar to match new K if needed)
        last_attn = attn_weights[..., -1, :].detach()  # [B, H, K_curr]
        if self.A_bar is None:
            # first step: just copy
            self.A_bar = last_attn.clone()
        else:
            prev_K = self.A_bar.size(2)
            curr_K = last_attn.size(2)
            if curr_K > prev_K:
                # pad new slots with zeros
                pad = torch.zeros(
                    self.A_bar.size(0),
                    self.A_bar.size(1),
                    curr_K - prev_K,
                    device=self.A_bar.device,
                    dtype=self.A_bar.dtype
                )
                A_bar = torch.cat([self.A_bar, pad], dim=2)
            elif curr_K < prev_K:
                # trim old entries
                A_bar = self.A_bar[..., :curr_K].clone()
            else:
                A_bar = self.A_bar
            # now safe to accumulate
            self.A_bar = A_bar + last_attn

        # Building keep-mask for the NEXT step
        k_len = K.size(-2)
        sb = int(self.start_ratio * k_len)
        rb = int(self.recent_ratio * k_len)
        sb = max(0, min(sb, k_len))
        rb = max(0, min(rb, k_len - sb))

        keep = torch.zeros(B, self.num_heads, 1, k_len, dtype=torch.bool, device=K.device)
        if sb > 0:
            keep[..., 0, :sb] = True
        if rb > 0:
            keep[..., 0, -rb:] = True
        self.attention_masks_next = keep

        # Cache Merging: merge evicted slot into suffix window
        V = local_cam_mask(V, self.A_bar, sb, rb)

        # Attention output
        # attn_output = attn_weights @ V  # [B, H, T, head_dim]
        attn_output = torch.einsum('bhtk,bhkd->bhtd', attn_weights, V)

        # Final linear projection
        out = attn_output.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        out = self.o_proj(out)

        # Prepare new_kv for caching
        if not output_attentions:
            attn_weights = None
        return out, attn_weights


def convert_kvcache_llama_cam(model: nn.Module, config: LlamaConfig) -> nn.Module:
    """
    Recursively replace every LlamaAttention with LlamaAttention_cam.
    """
    for name, child in list(model._modules.items()):
        if len(list(child.children())) > 0:
            model._modules[name] = convert_kvcache_llama_cam(child, config)
        if isinstance(child, LlamaAttention):
            cam_layer = LlamaAttention_cam(config)
            device = next(child.parameters()).device
            cam_layer.to(device).half()
            model._modules[name] = cam_layer
    return model
