# workers/cam/attention.py
import math
import torch
from torch import nn
from transformers.models.llava_next.modeling_llava_next import LlavaNextAttention

class LlavaNextAttention_CaM(LlavaNextAttention):
    def __init__(self, config):
        super().__init__(config)
        # Cache configuration
        self.start_ratio = config.start_ratio
        self.recent_ratio = config.recent_ratio
        self.merge_budget = 32  # Fixed merging window size
        
        # State tracking
        self.cache_budget = None
        self.attention_masks_next = None
        self.current_seq_len = 0

    def _local_cam_mask(self, value_states, attn_weights):
        """Merge evicted cache entries into subsequent tokens"""
        bsz, num_heads, seq_len, _ = value_states.size()
        
        # Calculate merging probability
        attn_scores = attn_weights.squeeze(0).squeeze(1)  # [bsz, num_heads, seq_len]
        token_idx = seq_len - self.recent_ratio
        
        # Handle initial sequence lengths
        if token_idx < 0:
            return value_states

        # Calculate attention score ratios
        with torch.no_grad():
            recent_scores = attn_scores[:, :, -self.recent_ratio+1:]
            mean_attn = torch.mean(recent_scores, dim=-1, keepdim=True)
            merge_prob = attn_scores[:, :, token_idx] / mean_attn.squeeze(-1)
            
            # Sanitize probability values
            merge_prob = torch.nan_to_num(merge_prob, nan=0.0, posinf=1.0, neginf=0.0)
            merge_mask = torch.bernoulli(merge_prob.clamp(min=0, max=1))

        # Distribute evicted token's value states
        evicted_values = value_states[:, :, token_idx, :].clone()
        merge_contribution = evicted_values * merge_mask.unsqueeze(-1) / self.merge_budget
        
        # Add to subsequent tokens
        start_idx = token_idx + 1
        end_idx = min(token_idx + self.merge_budget + 1, seq_len)
        
        if start_idx < end_idx:
            value_states[:, :, start_idx:end_idx, :] += merge_contribution.unsqueeze(2)

        return value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Original attention computation
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # RoPE embeddings
        query_states = self.rotary_emb(query_states, position_ids)
        key_states = self.rotary_emb(key_states, position_ids)

        # Merge with past KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
            self.current_seq_len = key_states.size(1)

        # Initialize cache budget
        if self.cache_budget is None and past_key_value is not None:
            self.cache_budget = int(self.start_ratio * self.current_seq_len) + \
                               int(self.recent_ratio * self.current_seq_len)

        # Perform cache merging when exceeding budget
        if past_key_value is not None and self.current_seq_len > self.cache_budget:
            # Compute attention weights
            attn_weights = torch.matmul(
                query_states,
                key_states.transpose(1, 2)
            ) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, 
                    torch.tensor(torch.finfo(attn_weights.dtype).min
                )

            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            
            # Apply cache merging
            value_states = self._local_cam_mask(value_states, attn_weights)
            
            # Update past key-value states
            past_key_value = (key_states, value_states) if use_cache else None

        # Continue with standard attention
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value