import math
import torch
from torch import nn

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

__all__ = ["convert_minigpt4_llama_cam", "MiniGPT4AttentionCam"]


def local_cam_mask(value_states, attn_scores, start_budget, recent_budget):
    B, H, Q, K = attn_scores.shape
    mb = recent_budget
    if mb <= 0 or mb >= K:
        return value_states

    last_attn = attn_scores[..., -1, :]
    evict_idx = K - mb
    win_start = evict_idx + 1
    win_len = K - win_start
    if win_len <= 0:
        return value_states

    window = last_attn[..., win_start:K]
    mean_attn = window.mean(dim=-1) + 1e-6
    attn_e = last_attn[..., evict_idx]
    prob = (attn_e / mean_attn).clamp(0.0, 1.0)
    prob = torch.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
    mask = torch.bernoulli(prob)
    v_e = value_states[..., evict_idx, :].clone()
    v_e = v_e * mask.unsqueeze(-1) / mb
    vs = value_states.clone()
    start = evict_idx + 1
    end = min(K, start + mb)
    vs[..., start:end, :] += v_e.unsqueeze(-2)
    return vs


class MiniGPT4AttentionCam(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(config)
        self.start_ratio = getattr(config, "start_ratio", 1.0)
        self.recent_ratio = getattr(config, "recent_ratio", 1.0)

    def _shape(self, x, seq_len, bsz):
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs
    ):
        hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e4, neginf=-1e4)
        B, T, _ = hidden_states.shape

        Q = self._shape(self.q_proj(hidden_states), T, B)
        K = self._shape(self.k_proj(hidden_states), T, B)
        V = self._shape(self.v_proj(hidden_states), T, B)

        kv_len = V.size(-2)
        pos_ids = torch.arange(kv_len, device=V.device)
        pos_ids = pos_ids.unsqueeze(0).expand(B, kv_len)
        cos, sin = self.rotary_emb(V, pos_ids)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if isinstance(past_key_value, (tuple, list)):
            K = torch.cat([past_key_value[0], K], dim=2)
            V = torch.cat([past_key_value[1], V], dim=2)
        new_kv = (K, V) if use_cache else None

        k_len = K.size(-2)
        sb = int(self.start_ratio * k_len)
        rb = int(self.recent_ratio * k_len)
        sb = max(0, min(sb, k_len))
        rb = max(0, min(rb, k_len - sb))

        cam_mask = torch.zeros(B, self.num_heads, 1, k_len, dtype=torch.bool, device=Q.device)
        if sb > 0:
            cam_mask[..., 0, :sb] = True
        if rb > 0:
            cam_mask[..., 0, -rb:] = True

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        scores = scores.masked_fill(~cam_mask, torch.finfo(scores.dtype).min)
        scores = torch.nan_to_num(scores, nan=torch.finfo(scores.dtype).min)
        scores = torch.softmax(scores, dim=-1)

        if T > 0:
            V = local_cam_mask(V, scores, sb, rb)
            new_kv = (K, V) if use_cache else None

        out = scores @ V
        out = out.transpose(1, 2).reshape(B, T, -1)
        out = self.o_proj(out)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        return out, new_kv


def convert_minigpt4_llama_cam(model: nn.Module, config):
    """
    Replace LLaMA-style attention with MiniGPT4AttentionCam across the model.
    """
    from transformers.models.llama.modeling_llama import LlamaAttention

    for name, child in model.named_children():
        if isinstance(child, LlamaAttention):
            new_mod = MiniGPT4AttentionCam(config)
            new_mod.load_state_dict(child.state_dict(), strict=False)
            model._modules[name] = new_mod.to(next(child.parameters()).device)
        else:
            convert_minigpt4_llama_cam(child, config)
    return model
