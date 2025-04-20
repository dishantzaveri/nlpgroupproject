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


def local_cam_mask(value_states, attn_scores, start_budget, recent_budget):
    """
    Out‑of‑place CaM merge for the last generated token's V:
      - value_states: [B, H, K, D]
      - attn_scores:  [B, H, Q, K]
    Returns a fresh V tensor (never writes out‑of‑bounds).
    """
    B, H, Q, K = attn_scores.shape
    mb = recent_budget

    # If budget is zero or >= full cache, nothing to merge
    if mb <= 0 or mb >= K:
        return value_states

    # last token’s attention: [B,H,K]
    last_attn = attn_scores[..., -1, :]

    # eviction index
    evict_idx = K - mb

    # compute window length
    win_start = evict_idx + 1
    win_len   = K - win_start
    if win_len <= 0:
        return value_states

    # mean over the surviving window
    window    = last_attn[..., win_start : K]       # [B,H,win_len]
    mean_attn = window.mean(dim=-1)                 # [B,H]
    mean_attn = mean_attn + 1e-6                    # avoid div by zero

    # merge probability & mask
    attn_e = last_attn[..., evict_idx]              # [B,H]
    prob   = (attn_e / mean_attn).clamp(0.0, 1.0)    
    prob   = torch.nan_to_num(                      # sanitize
        prob, nan=0.0, posinf=1.0, neginf=0.0
    )
    mask   = torch.bernoulli(prob)                  # [B,H]

    # gather & scale the evicted V
    v_e = value_states[..., evict_idx, :].clone()   # [B,H,D]
    v_e = v_e * mask.unsqueeze(-1) / mb             # [B,H,D]

    # out‑of‑place update: clone then scatter
    vs = value_states.clone()                       # [B,H,K,D]
    start = evict_idx + 1
    end   = min(K, start + mb)
    vs[..., start:end, :] += v_e.unsqueeze(-2)      # broadcast

    return vs


class LlamaAttention_cam(nn.Module):
    """
    Chunked, streaming CaM‐wrapped LlamaAttention:
      • rebuilds keep‐mask every forward to match growing KV cache
      • merges evicted cache entries back into neighbors (out‑of‑place)
      • sanitizes hidden_states, scores, and outputs
      • returns exactly (attn_output, new_kv)
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.step = 0
        self.hidden_size = config.hidden_size
        self.num_heads   = config.num_attention_heads
        self.head_dim    = self.hidden_size // self.num_heads

        # projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # rotary
        self.rotary_emb = LlamaRotaryEmbedding(config)

        # CaM ratios
        self.start_ratio  = config.start_ratio
        self.recent_ratio = config.recent_ratio

    def _shape(self, x, seq_len, bsz):
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states,        # [B, T, hidden_size]
        attention_mask=None,  # [B, 1, T, K]
        position_ids=None,
        past_key_value=None,  # tuple(K_cache, V_cache)
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs
    ):
        # 0) sanitize hidden_states
        hidden_states = torch.nan_to_num(
            hidden_states, nan=0.0, posinf=1e4, neginf=-1e4
        )

        B, T, _ = hidden_states.shape

        # 1) project QKV
        Q = self._shape(self.q_proj(hidden_states), T, B)
        K = self._shape(self.k_proj(hidden_states), T, B)
        V = self._shape(self.v_proj(hidden_states), T, B)

        # 2) rotary embeddings
        kv_len = V.size(-2)
        pos_ids = torch.arange(kv_len, device=V.device)
        pos_ids = pos_ids.unsqueeze(0).expand(B, kv_len)
        cos, sin = self.rotary_emb(V, pos_ids)
        Q, K    = apply_rotary_pos_emb(Q, K, cos, sin)

        # 3) append past cache
        if isinstance(past_key_value, (tuple, list)):
            K = torch.cat([past_key_value[0], K], dim=2)
            V = torch.cat([past_key_value[1], V], dim=2)

        new_kv = (K, V) if use_cache else None

        # 4) rebuild tiny streaming mask [B,H,1,K]
        MIN   = torch.finfo(Q.dtype).min
        k_len = K.size(-2)
        sb    = int(self.start_ratio * k_len)
        rb    = int(self.recent_ratio * k_len)
        sb    = max(0, min(sb, k_len))
        rb    = max(0, min(rb, k_len - sb))

        sm = torch.zeros(B, self.num_heads, 1, k_len,
                         dtype=torch.bool, device=Q.device)
        if sb > 0:
            sm[..., 0, :sb]   = True
        if rb > 0:
            sm[..., 0, -rb:]  = True

        # 5) chunked attention + CaM merge only on the final chunk
        out_chunks = []
        chunk_size = 256

        for i in range(0, T, chunk_size):
            j = min(i + chunk_size, T)
            Qi = Q[:, :, i:j, :]

            # raw scores
            scores = (Qi @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # masks
            if attention_mask is not None:
                scores = scores + attention_mask[:, :, i:j, :]
            scores = scores.masked_fill(~sm, MIN)

            # sanitize & softmax
            scores = torch.nan_to_num(
                scores, nan=MIN, posinf=MIN, neginf=MIN
            )
            scores = torch.softmax(scores, dim=-1)

            # CaM merge on last slice
            if j == T:
                V = local_cam_mask(V, scores, sb, rb)
                new_kv = (K, V) if use_cache else None

            out_chunks.append(scores @ V)

        # 6) stitch and project out
        out = torch.cat(out_chunks, dim=2)               # [B,H,T,hd]
        out = out.transpose(1, 2).reshape(B, T, -1)      # [B,T,hidden_size]
        out = self.o_proj(out)

        # final sanitize
        out = torch.nan_to_num(
            out, nan=0.0, posinf=1e4, neginf=-1e4
        )
        
        self.step += 1
        # K is of shape [B, H, K_len, head_dim]
        B      = K.shape[0]
        kv_len = K.shape[2]
        # each value vector is fp16 => 2 bytes
        kv_bytes = B * self.num_heads * self.head_dim * kv_len * 2
        print(f"[CaM] Step {self.step}: KV cache length = {kv_len}, "
              f"approx {kv_bytes/2**20:.1f} MB")  

        return out, new_kv


def convert_kvcache_llama_cam(model: nn.Module, config):
    """
    Recursively replace every LlamaAttention module
    with our CaM version.
    """
    for name, child in list(model._modules.items()):
        if len(list(child.children())) > 0:
            model._modules[name] = convert_kvcache_llama_cam(child, config)
        if isinstance(child, LlamaAttention):
            cam = LlamaAttention_cam(config)
            dev = next(child.parameters()).device
            cam.to(dev).half()
            model._modules[name] = cam
    return model
