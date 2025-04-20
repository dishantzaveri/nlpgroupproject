import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class MiniGPT4AttentionCam(nn.Module):
    def __init__(self, hidden_size, num_heads, max_position_embeddings, rope_theta=10000, dropout=0.0,
                 start_ratio=1.0, recent_ratio=1.0, merge_token=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.start_ratio = start_ratio
        self.recent_ratio = recent_ratio
        self.merge_token = merge_token
        self.use_cam = start_ratio < 1.0 or recent_ratio < 1.0

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        bsz, seq_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.apply_rope(q, k, position_ids)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_past = (k, v) if use_cache else None

        kv_len = k.size(2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # CaM mask application
        if self.use_cam and kv_len > 1:
            start_budget = math.ceil(self.start_ratio * kv_len)
            recent_budget = math.ceil(self.recent_ratio * kv_len)

            cam_mask = torch.zeros(kv_len, dtype=torch.bool, device=attn_weights.device)
            cam_mask[:start_budget] = True
            cam_mask[-recent_budget:] = True

            cam_mask = cam_mask.view(1, 1, 1, kv_len)
            attn_weights = torch.where(
                cam_mask,
                attn_weights,
                torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Optional: Token merging in value (not implemented here, placeholder)
        if self.merge_token:
            # Add value merging logic here if needed
            pass

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        output = self.out_proj(attn_output)

        return output, new_past

    def apply_rope(self, q, k, position_ids):
        if position_ids is None:
            position_ids = torch.arange(q.shape[2], device=q.device).unsqueeze(0)

        theta = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, device=q.device).float() / self.head_dim))
        sinusoid = torch.einsum("i,j->ij", position_ids.squeeze(0).float(), theta)
        sin, cos = sinusoid.sin(), sinusoid.cos()

        def apply(x):
            x1, x2 = x[..., 0::2], x[..., 1::2]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return apply(q), apply(k)
