import torch
from torch import nn
from transformers import LlavaOnevisionForConditionalGeneration, AutoConfig, LlavaOnevisionProcessor
from PIL import Image
import os

import math
from typing import Optional, Tuple
import torch
import math

from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Config, Qwen2RotaryEmbedding, Qwen2MLP, Qwen2RMSNorm, Qwen2DecoderLayer, Qwen2PreTrainedModel, Qwen2Model, BaseModelOutputWithPast, AttentionMaskConverter, StaticCache, SlidingWindowCache
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from transformers.cache_utils import Cache # Assuming Cache object is used
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config # Assuming config path
from transformers.utils import is_flash_attn_2_available
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv, eager_attention_forward, ALL_ATTENTION_FUNCTIONS
from typing import Callable, Optional, Tuple
import copy
from transformers.utils import logging
from transformers import SinkCache

logger = logging.get_logger(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
os.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True"

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    "models/llava-onevision-qwen2-7b-ov-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="cache_dir"
)


# def local_cam_mask(value_states,attn_weights,start_budget,recent_budget):
#     seq_length = attn_weights.shape[-1]
#     padding_length = 0
#     #merge_budget = math.ceil(recent_budget/2)
#     merge_budget = recent_budget
#     for token_index in range(start_budget+padding_length+recent_budget, seq_length):
#         attn_score = torch.mean(attn_weights[:,:, :token_index,:token_index], dim=-2)
#         mean_attn = torch.max(torch.cat((attn_score[0,:,:start_budget],attn_score[0,:,token_index-recent_budget:token_index]),dim=-1),dim=-1)[0]
#         merge_prob = attn_score[0,:,token_index-recent_budget]/mean_attn
#         merge_mask = torch.bernoulli(merge_prob.clamp(min=0,max=1))
#         score1=value_states[:,:,token_index-recent_budget, ...].clone()*merge_mask.unsqueeze(-1)/merge_budget
#         value_states[:,:,token_index-recent_budget+1:token_index-recent_budget+merge_budget+1,:]+=score1.unsqueeze(2)
#     return value_states

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

def unrepeat_kv(repeated_states: torch.Tensor, num_key_value_heads: int, num_attention_heads: int) -> torch.Tensor:
    """Reverses the effect of repeat_kv."""
    batch, n_attn_heads, slen, head_dim = repeated_states.shape
    if n_attn_heads == num_key_value_heads:
        return repeated_states
    if n_attn_heads % num_key_value_heads == 0:
        n_rep = num_attention_heads // num_key_value_heads
        reshaped_states = repeated_states.view(batch, num_key_value_heads, n_rep, slen, head_dim)
        original_states = reshaped_states[:, :, 0, :, :]
        return original_states
    else:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")


# def local_cam_mask(
#     value_states: torch.Tensor,      # Full value tensor [B, H, kv_seq_len, D]
#     attn_score_slice: torch.Tensor,  # SLICED scores [B, H, slice_width] (assuming q_len=1 was squeezed)
#     kv_seq_len: int,                 # FULL original sequence length
#     start_budget: int,               # Unused in this function, but kept for signature?
#     recent_budget: int
# ) -> torch.Tensor:
#     """
#     Applies Cache Merging based on sliced attention scores.

#     Args:
#         value_states: The full value tensor after repeat_kv [B, H, kv_seq_len, D].
#                       This tensor will be modified IN-PLACE (implicitly).
#         attn_score_slice: The post-softmax attention scores corresponding ONLY to the
#                           keys relevant for CaM calculation (typically near the end).
#                           Expected shape [B, H, slice_width] assuming q_len=1.
#         kv_seq_len: The total sequence length of keys/values.
#         start_budget: The start budget size (currently unused here).
#         recent_budget: The recent budget size, determines which tokens are involved.

#     Returns:
#         The modified value_states tensor.
#     """
#     merge_budget = 32 # Or get from config
#     bsz, num_heads, _, head_dim = value_states.shape

#     # --- Input Validation ---
#     if attn_score_slice.dim() != 3 or attn_score_slice.shape[0] != bsz or attn_score_slice.shape[1] != num_heads:
#          raise ValueError(f"Unexpected attn_score_slice shape: {attn_score_slice.shape}. Expected [B, H, slice_width]")
#     if recent_budget < 2:
#          print("Warning: recent_budget < 2, CaM logic might be unstable.")
#          return value_states # Skip CaM if budget is too small

#     # --- Determine Absolute Indices ---
#     # Absolute index of the token whose score is used for merge_prob
#     merge_prob_abs_idx = kv_seq_len - recent_budget + 1
#     # Absolute index of the token *being merged from*
#     merged_from_abs_idx = merge_prob_abs_idx # Same token in this logic

#     # Absolute start index for the destination range where merging occurs
#     merge_dest_start_abs_idx = kv_seq_len - recent_budget + 2
#     # Absolute end index for the destination range (exclusive)
#     merge_dest_end_abs_idx = merge_dest_start_abs_idx + merge_budget
#     # Clamp destination end index to sequence length
#     merge_dest_end_abs_idx = min(merge_dest_end_abs_idx, kv_seq_len)

#     # Check if destination range is valid
#     if merge_dest_start_abs_idx >= merge_dest_end_abs_idx:
#         # print(f"Warning: CaM destination range is empty or invalid. Start: {merge_dest_start_abs_idx}, End: {merge_dest_end_abs_idx}")
#         return value_states # Cannot merge

#     # --- Calculate Indices Relative to the Slice ---
#     slice_width = attn_score_slice.shape[-1]
#     # Start index (absolute) of the keys represented in the slice
#     slice_abs_start_idx = kv_seq_len - slice_width

#     # Relative index within the slice corresponding to merge_prob_abs_idx
#     merge_prob_rel_idx = merge_prob_abs_idx - slice_abs_start_idx

#     # Relative start index for calculating mean_attn
#     mean_attn_start_rel_idx = (kv_seq_len - recent_budget + 2) - slice_abs_start_idx
#     # Relative end index for calculating mean_attn (exclusive)
#     mean_attn_end_rel_idx = kv_seq_len - slice_abs_start_idx

#     # --- Boundary Checks for Slice Indexing ---
#     if not (0 <= merge_prob_rel_idx < slice_width):
#         print(f"Warning: Calculated merge_prob_rel_idx ({merge_prob_rel_idx}) is out of bounds for slice width {slice_width}. Skipping CaM.")
#         return value_states
#     if not (0 <= mean_attn_start_rel_idx < mean_attn_end_rel_idx <= slice_width):
#          print(f"Warning: Calculated mean_attn indices ({mean_attn_start_rel_idx}:{mean_attn_end_rel_idx}) are out of bounds for slice width {slice_width}. Skipping CaM.")
#          return value_states


#     # --- Calculate Merge Probability ---
#     # attn_score_slice shape: [B, H, slice_width]
#     # Calculate mean over the last relevant scores IN THE SLICE
#     mean_attn = torch.mean(attn_score_slice[:, :, mean_attn_start_rel_idx:mean_attn_end_rel_idx], dim=-1) # Shape: [B, H]

#     # Get the specific score needed for the numerator using RELATIVE index
#     merge_prob_numerator_score = attn_score_slice[:, :, merge_prob_rel_idx] # Shape: [B, H]

#     # Avoid division by zero or very small numbers
#     mean_attn = torch.where(mean_attn == 0, torch.tensor(1e-6, device=mean_attn.device, dtype=mean_attn.dtype), mean_attn)
#     merge_prob = merge_prob_numerator_score / mean_attn # Shape: [B, H]

#     # Handle NaN/Inf
#     if torch.isnan(merge_prob).any(): merge_prob[torch.isnan(merge_prob)] = 0
#     if torch.isinf(merge_prob).any(): merge_prob[torch.isinf(merge_prob)] = 1 # Clamp large ratios to 1

#     # --- Perform Merging ---
#     merge_mask = torch.bernoulli(merge_prob.clamp(min=0, max=1)) # Shape: [B, H]

#     # Select the value state vector to be merged *using ABSOLUTE index*
#     # Shape before clone: [B, H, D]
#     value_to_merge = value_states[:, :, merged_from_abs_idx, :].clone()

#     # Calculate the merged value component
#     # merge_mask needs expansion: [B, H, 1] to multiply with [B, H, D]
#     merged_component = value_to_merge * merge_mask.unsqueeze(-1) / merge_budget # Shape: [B, H, D]

#     # Add the merged component to the destination range *using ABSOLUTE indices*
#     # merged_component needs expansion: [B, H, 1, D] to add to [B, H, range_len, D]
#     value_states[:, :, merge_dest_start_abs_idx:merge_dest_end_abs_idx, :] += merged_component.unsqueeze(2)

#     return value_states

# class Qwen2AttentionCam(nn.Module):
#     """
#     Multi-headed attention from 'Attention Is All You Need' paper, modified with
#     Cache Merging (CaM).
#     """

#     def __init__(self, config: Qwen2Config, layer_idx: int = None):
#         super().__init__()
#         self.config = config
#         self.layer_idx = layer_idx
#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         self.head_dim = self.hidden_size // self.num_heads
#         self.num_key_value_heads = config.num_key_value_heads
#         self.num_key_value_groups = self.num_heads // self.num_key_value_heads
#         self.max_position_embeddings = config.max_position_embeddings
#         self.rope_theta = config.rope_theta
#         self.is_causal = True
#         self.attention_dropout = config.attention_dropout

#         self.q_proj = nn.Linear(self.hidden_size, self.num_heads  self.head_dim, bias=True)
#         self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads  self.head_dim, bias=True)
#         self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads  self.head_dim, bias=True)
#         self.o_proj = nn.Linear(self.num_heads  self.head_dim, self.hidden_size, bias=False)

#         # --- CaM Specific Parameters ---
#         self.start_budget_ratio = getattr(config, "start_ratio", 1.0)
#         self.recent_budget_ratio = getattr(config, "recent_ratio", 1.0)
#         self.merge_token = getattr(config, "merge_token", False)

#         self.use_cam = self.start_budget_ratio < 1.0 or self.recent_budget_ratio < 1.0
#         # --- End CaM Specific Parameters ---


#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#         cache_position: Optional[torch.LongTensor] = None, 
#         kwargs
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

#         bsz, q_len, _ = hidden_states.size()

#         # Standard QKV projections
#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)

#         query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#         key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#         value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

#         kv_seq_len = key_states.shape[-2]
#         if past_key_value is not None:
#             if cache_position is None:
#                  raise ValueError("cache_position is required when past_key_value is not None")
#             kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

 
#         # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

#         if past_key_value is not None:
#             # Update KV cache using the Cache object
#             # This handles concatenation and potential eviction implicitly if using DynamicCache etc.
#             # We need cos/sin for RoPE cache update if applicable
#             # Let's assume RoPE is applied before caching K/V, as is common.
#             # If RoPE is applied after cache retrieval, logic needs adjustment.

#             # Apply RoPE before updating cache
#             # We need position_ids to apply RoPE correctly during generation
#             if position_ids is None and q_len != 1:
#                  # This may happen during prompt processing
#                  position_ids = torch.arange(kv_seq_len - q_len, kv_seq_len, dtype=torch.long, device=hidden_states.device)
#                  position_ids = position_ids.unsqueeze(0).view(-1, q_len)
#             elif position_ids is None and q_len == 1:
#                  # Generation phase, position is the current length
#                  position_ids = torch.tensor([[kv_seq_len - 1]], dtype=torch.long, device=hidden_states.device)

#             # Placeholder: Instantiate or get RoPE embedding class
#             # rotary_emb = Qwen2RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)
#             # cos, sin = rotary_emb(value_states, seq_len=kv_seq_len) # Might need adjustment based on RoPE class API
#             # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos[:, :, kv_seq_len-q_len:kv_seq_len, :], sin[:, :, kv_seq_len-q_len:kv_seq_len, :])
#             #  Actual RoPE application needs careful integration matching Qwen2 
#             # Simulating it being done:
#             # query_states, key_states = apply_rotary_pos_emb(...) # Assume this happens correctly

#             cache_kwargs = {"cache_position": cache_position} # Add cos/sin if RoPE cache needs them
#             key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
#             # After update, key_states/value_states contain the full sequence from the cache
#             kv_seq_len = key_states.shape[-2] # Update kv_seq_len to full length

#         # --- CaM Logic Branch ---
#         if self.use_cam and q_len > 0 : # Only apply CaM if enabled and there's a query
#             # Force Eager computation for CaM
#             # Handle GQA by repeating K/V states
#             key_states_repeated = repeat_kv(key_states, self.num_key_value_groups)
#             value_states_repeated = repeat_kv(value_states, self.num_key_value_groups)

#             # Calculate scaling factor
#             scaling_factor = self.head_dim-0.5

#             # Calculate Attention Scores
#             attn_weights = torch.matmul(query_states, key_states_repeated.transpose(2, 3))  scaling_factor

#             if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
#                  raise ValueError(
#                      f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
#                      f" {attn_weights.size()}"
#                  )

#             # Apply original attention mask (causal/padding)
#             if attention_mask is not None:
#                 if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
#                      # Try broadcasting if mask shape is different (e.g., (bsz, kv_seq_len))
#                      # This depends on the exact format of the mask Qwen2 expects
#                      # Assuming it's compatible or needs adjustment here
#                      try:
#                          # Common shape is (bsz, 1, q_len, kv_seq_len)
#                          # Or sometimes (1, 1, q_len, kv_seq_len) for causal
#                          # Or (bsz, 1, 1, kv_seq_len) for padding
#                          attn_weights = attn_weights + attention_mask
#                      except RuntimeError as e:
#                            raise ValueError(
#                                f"Attention mask shape {attention_mask.size()} cannot be broadcast to attention weights "
#                                f"shape {attn_weights.size()}. Ensure mask is compatible."
#                            ) from e
#                 # Masked positions should be negative infinity
#                 attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))


#             # --- CaM Step 1: Attention Score Masking (Heavy Hitter + Recent) ---
#             start_budget = math.ceil(self.start_budget_ratio  kv_seq_len)
#             recent_budget = math.ceil(self.recent_budget_ratio  kv_seq_len)
#             # Ensure budgets are valid
#             start_budget = max(0, min(start_budget, kv_seq_len))
#             recent_budget = max(0, min(recent_budget, kv_seq_len))

#             if kv_seq_len > start_budget + recent_budget: # Only apply CaM mask if there are tokens in the middle
#                 # Create mask for allowed indices (True = keep)
#                 cam_mask = torch.zeros_like(attn_weights[0, 0, 0, :], dtype=torch.bool) # Shape [kv_seq_len]
#                 cam_mask[:start_budget] = True
#                 cam_mask[-recent_budget:] = True

#                 # Expand mask to match attention weights shape (excluding batch and head dims for now)
#                 # Mask applies along the key/value sequence length dimension
#                 cam_mask_expanded = cam_mask.view(1, 1, 1, kv_seq_len).expand(bsz, self.num_heads, q_len, kv_seq_len)

#                 # Apply CaM mask: set scores for non-allowed keys to -inf
#                 # Combine with causality implicitly if attention_mask already handles it.
#                 # We apply CaM mask after the original mask.
#                 attn_weights = torch.where(
#                     cam_mask_expanded,
#                     attn_weights,
#                     torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
#                 )
#             # --- End CaM Step 1 ---

#             # Softmax
#             attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=query_states.dtype) #.to(query_states.dtype)

#             # Dropout (optional, apply after softmax)
#             # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training) # If dropout needed here

#             # --- CaM Step 2: Value Merging (Conditional) ---
#             if self.merge_token and kv_seq_len > start_budget + recent_budget:
#                 # Pass the repeated value states to local_cam_mask as it operates per-head like attn_weights
#                 # Note: local_cam_mask modifies value_states_repeated in-place or returns modified one
#                 value_states_repeated = local_cam_mask(value_states_repeated, attn_weights, start_budget, recent_budget)
#             # --- End CaM Step 2 ---

#             # Final Attention Output Calculation
#             attn_output = torch.matmul(attn_weights, value_states_repeated)

#             # Restore original attention weights if needed for output
#             if not output_attentions:
#                 attn_weights = None

#         # --- Original Logic Branch (No CaM or q_len is 0) ---
#         else:
#             # Use the standard attention interface (potentially optimized)
#             sliding_window = None
#             if (
#                 self.config.use_sliding_window
#                 and getattr(self.config, "sliding_window", None) is not None
#                 # Check layer index condition if needed
#                 # and self.layer_idx >= self.config.max_window_layers # Example condition
#             ):
#                 sliding_window = self.config.sliding_window

#             # Determine attention implementation (copied from original Qwen2)
#             # Note: _attn_implementation might require specific setup during model loading
#             attn_implementation = self.config._attn_implementation \
#                 if hasattr(self.config, "_attn_implementation") else "eager" # Default to eager if not set

#             if attn_implementation == "flash_attention_2":
#                  if not is_flash_attn_2_available():
#                       raise ImportError("Flash Attention 2 requested but not available.")
#                  from transformers.models.qwen2.modeling_qwen2 import flash_attention_forward # Import specific forwarder
#                  attention_interface = flash_attention_forward
#             elif attn_implementation == "sdpa":
#                  from transformers.models.qwen2.modeling_qwen2 import sdpa_attention_forward # Import specific forwarder
#                  if output_attentions:
#                      # Fallback to eager if output_attentions is needed
#                      from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward
#                      attention_interface = eager_attention_forward
#                  else:
#                       attention_interface = sdpa_attention_forward
#             else: # Default or "eager"
#                  from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward # Import specific forwarder
#                  attention_interface = eager_attention_forward

#             # Call the selected attention function
#             # It needs the scaling factor separate from config in some implementations
#             attn_output, attn_weights = attention_interface(
#                 query_states,
#                 key_states,
#                 value_states,
#                 attention_mask=attention_mask,
#                 num_heads=self.num_heads, # Pass necessary args
#                 head_dim=self.head_dim,
#                 dropout=self.attention_dropout if self.training else 0.0,
#                 use_cache=use_cache,
#                 layer_idx=self.layer_idx,
#                 sliding_window=sliding_window,
#                 output_attentions=output_attentions,
#                 # Pass other kwargs if the interface expects them
#                 kwargs,
#             )
#         # --- End Original Logic Branch ---


#         # Reshape and project output
#         attn_output = attn_output.transpose(1, 2).contiguous()
#         attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
#         attn_output = self.o_proj(attn_output)

#         # Return value format expected by HF models
#         # The past_key_value to return is the one updated by cache.update()
#         # CaM modifies attention pattern and values, not the cache structure itself.
#         # past_key_value_to_return = past_key_value if use_cache else None

#         return attn_output, attn_weights


# class Qwen2AttentionCam(nn.Module):
#     def __init__(self, config: Qwen2Config, layer_idx: int = None):
#         super().__init__()
#         self.config = config
#         self.layer_idx = layer_idx
#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         self.head_dim = self.hidden_size // self.num_heads
#         self.num_key_value_heads = config.num_key_value_heads
#         self.num_key_value_groups = self.num_heads // self.num_key_value_heads
#         self.max_position_embeddings = config.max_position_embeddings
#         self.rope_theta = config.rope_theta
#         self.is_causal = True
#         self.scaling = self.head_dim-0.5
#         self.attention_dropout = config.attention_dropout

#         self.q_proj = nn.Linear(self.hidden_size, self.num_heads, self.head_dim, bias=True)
#         self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads, self.head_dim, bias=True)
#         self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads, self.head_dim, bias=True)
#         self.o_proj = nn.Linear(self.num_heads, self.head_dim, self.hidden_size, bias=False)

#         # --- CaM Specific Parameters ---
#         self.start_budget_ratio = getattr(config, "start_ratio", 1.0)
#         self.recent_budget_ratio = getattr(config, "recent_ratio", 1.0)
#         self.merge_token = getattr(config, "merge_token", False)

#         self.use_cam = self.start_budget_ratio < 1.0 or self.recent_budget_ratio < 1.0
#         # --- End CaM Specific Parameters ---
    
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         position_embeddings: Tuple[torch.Tensor, torch.Tensor],
#         attention_mask: Optional[torch.Tensor] = None,
#         past_key_value: Optional[Cache] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = False,
#         **kwargs
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         bsz, q_len, _ = hidden_states.size()

#         # # Standard QKV projections
#         # query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#         # key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#         # value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

#         input_shape = hidden_states.shape[:-1]
#         hidden_shape = (input_shape, -1, self.head_dim)

#         query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
#         key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
#         value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

#         # Apply rotary position embeddings
#         cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

#         # Handle past_key_value (cache)
#         # kv_seq_len = key_states.shape[-2]
#         if past_key_value is not None:
#             # cached_key_states, cached_value_states = past_key_value[0]

#             # Ensure cached_key_states and cached_value_states are valid
#             # if not isinstance(cached_key_states, torch.Tensor) or not isinstance(cached_value_states, torch.Tensor):
#             #     raise ValueError("past_key_value must contain tensors for cached key and value states.")

#             # kv_seq_len += cached_key_states.shape[-2]

#             # Concatenate cached and current key/value states
#             # key_states = torch.cat([cached_key_states, key_states], dim=2)
#             # value_states = torch.cat([cached_value_states, value_states], dim=2)
#             # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
#             # kv_seq_len += past_key_value[0][0].shape[-2]
#             # key_states = torch.cat([past_key_value[0][0], key_states], dim=2)
#             # value_states = torch.cat([past_key_value[0][1], value_states], dim=2)
#             # kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

#         # if past_key_value is not None:
#             # sin and cos are specific to RoPE models; cache_position needed for the static cache
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

#         sliding_window = None
#         if (
#             self.config.use_sliding_window
#             and getattr(self.config, "sliding_window", None) is not None
#             and self.layer_idx >= self.config.max_window_layers
#         ):
#             sliding_window = self.config.sliding_window

#         attention_interface: Callable = eager_attention_forward
#         if self.config._attn_implementation != "eager":
#             if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
#                 logger.warning_once(
#                     "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
#                     'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
#                 )
#             else:
#                 attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

#         # kwargs['output_attentions']=True
#         # key_states = repeat_kv(key_states, self.num_key_value_groups)
#         # value_states = repeat_kv(value_states, self.num_key_value_groups)
#         attn_output, attn_weights = attention_interface(
#             self,
#             query_states,
#             key_states,
#             value_states,
#             attention_mask,
#             dropout=0.0 if not self.training else self.attention_dropout,
#             scaling=self.scaling,
#             sliding_window=sliding_window,  # main diff with Llama
#             # output_attentions=True,
#             **kwargs,
#         )

#         attn_output = attn_output.reshape(input_shape, -1).contiguous()
#         attn_output = self.o_proj(attn_output)
#         return attn_output, attn_weights

#         ## CaM
#         ### Heavy + Recent
#         start_budget = math.ceil(self.start_budget_ratio, attn_weights.shape[-1])
#         recent_budget = math.ceil(self.recent_budget_ratio, attn_weights.shape[-1])

#         ones = torch.ones_like(attn_weights, dtype=torch.bool)
#         ones = torch.triu(ones, diagonal=-recent_budget)
#         mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
#         mask_bottom[:, :, :, :start_budget] = True
#         mask_bottom = torch.logical_or(mask_bottom, ones)
#         mask_bottom = torch.tril(mask_bottom, diagonal=0)  
              
#         attn_weights[~mask_bottom] = torch.min(attention_mask)
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
#         value_states_ = repeat_kv(value_states, self.num_key_value_groups)
#         if self.merge_token==True:
#             value_states_ = local_cam_mask(value_states_,attn_weights,start_budget,recent_budget) # Default: No padding applied to input
#         attn_output = torch.matmul(attn_weights, value_states_)
#         if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
#                 f" {attn_output.size()}"
#             )

#         attn_output = attn_output.transpose(1, 2)
#         attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

#         attn_output = self.o_proj(attn_output)

#         # attn_output = attn_output.reshape(input_shape, -1).contiguous()
#         # attn_output = self.o_proj(attn_output)


#         # Apply CaM logic (as in your implementation)
#         # ...
#         # past_key_value = (key_states, value_states) if use_cache else None

#         return attn_output, attn_weights

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

        self.head_chunk_size = 4 
    
    def _reset_masks(self):
        self.attention_masks_next = None 
        self.start_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None
        self.input_length = []
        self.cache_budget_records = []

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
            if value_states.shape[1]<query_states.shape[1]:
                value_states = repeat_kv(value_states, self.num_key_value_groups)
            key_states, value = past_key_value.update(key_states, value_states, self.layer_idx)
            # key_states = torch.cat([past_key_value[0], key_states], dim=2)
            # value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # past_key_value = (key_states, value_states) if use_cache else None

        # key = repeat_kv(key_states, self.num_key_value_groups)
        # # if value_states.shape[1]<query_states.shape[1]:
        # value = repeat_kv(value_states, self.num_key_value_groups)
        key = repeat_kv(key_states, self.num_key_value_groups)
        if value_states.shape[1]<query_states.shape[1]:
            value = repeat_kv(value_states, self.num_key_value_groups)
        del value_states

        # kv_seq_len = key.shape[-2]
        if self.previous_scores is None: # Or some other logic to run once
            self.start_budget = int(self.start_budget_ratio * kv_seq_len)
            self.recent_budget = int(self.recent_budget_ratio * kv_seq_len)
            self.cache_budget = self.start_budget + self.recent_budget
            self.cache_budget_records.append(self.cache_budget) # Track budget history
            self.input_length.append(kv_seq_len) # Track input length history
            self.previous_scores = 1 

        attn_output = torch.zeros_like(query_states)
        sliced_attn_weights_chunks = []
        cam_slice_start_idx = kv_seq_len - self.recent_budget + 1
        cam_slice_end_idx = kv_seq_len
        # Ensure start index is valid
        cam_slice_start_idx = max(0, cam_slice_start_idx)

        # Check if slicing is possible/needed
        can_slice_for_cam = self.merge_token and self.attention_masks_next is not None and (cam_slice_end_idx > cam_slice_start_idx)

        for h_start in range(0, self.num_heads, self.head_chunk_size):
            h_end = min(h_start + self.head_chunk_size, self.num_heads)

            # --- Slice Tensors ---
            q_chunk = query_states[:, h_start:h_end, :, :]
            k_chunk = key[:, h_start:h_end, :, :]
            v_chunk = value[:, h_start:h_end, :, :]

            # --- Calculate Scores ---
            attn_weights_chunk_raw = torch.matmul(q_chunk, k_chunk.transpose(2, 3)) * self.scaling

            # --- Apply Primary Mask ---
            if attention_mask is not None:
                 # Check mask shape compatibility
                 if attention_mask.shape[-1] != kv_seq_len:
                      raise ValueError(f"attention_mask shape mismatch: {attention_mask.shape} vs kv_seq_len {kv_seq_len}")
                 attn_weights_chunk_raw = attn_weights_chunk_raw + attention_mask

            # --- Apply Secondary Mask ---
            if self.attention_masks_next is not None:
                mask_next_full = self.attention_masks_next
                if mask_next_full.shape[-1] == kv_seq_len : # Check size
                    mask_next_chunk = mask_next_full[:, h_start:h_end, :, :kv_seq_len]
                    attn_weights_chunk_raw = attn_weights_chunk_raw * mask_next_chunk + (1 - mask_next_chunk) * torch.finfo(attn_weights_chunk_raw.dtype).min
                # else: Handle mismatch

            # --- Softmax ---
            attn_weights_chunk_softmax = nn.functional.softmax(attn_weights_chunk_raw, dim=-1, dtype=torch.float32).to(q_chunk.dtype)

            # --- Extract Slice for CaM (if needed) ---
            if can_slice_for_cam:
                 # Slice along the key dimension (last dimension)
                 attn_weights_slice = attn_weights_chunk_softmax[:, :, :, cam_slice_start_idx:cam_slice_end_idx]
                 sliced_attn_weights_chunks.append(attn_weights_slice.cpu()) # Move slice to CPU maybe? Or store on GPU?

            # --- Calculate Output Chunk ---
            attn_output_chunk = torch.matmul(attn_weights_chunk_softmax, v_chunk)

            # --- Accumulate Output ---
            attn_output[:, h_start:h_end, :, :] = attn_output_chunk

            # del q_chunk, k_chunk, v_chunk, attn_weights_chunk_raw, attn_weights_chunk_softmax, attn_output_chunk


        # --- Post-Loop Processing ---

        # --- Reconstruct *Sliced* attn_weights for CaM ---
        full_sliced_attn_weights = None
        if can_slice_for_cam and sliced_attn_weights_chunks:
             try:
                  # Concatenate the slices along the head dimension
                  full_sliced_attn_weights = torch.cat(sliced_attn_weights_chunks, dim=1).to(query_states.device) # Move back to GPU if needed
                  del sliced_attn_weights_chunks # Free chunk list memory
             except RuntimeError as e:
                  print(f"Error concatenating sliced attn_weights chunks: {e}")
                  can_slice_for_cam = False # Disable CaM if reconstruction failed


        # --- CaM Merging (Using Sliced Weights, Operating on Full Value) ---
        # Check all conditions again, especially if slicing/concat worked
        if can_slice_for_cam and full_sliced_attn_weights is not None:
            print(f"Layer {self.layer_idx}: Applying CaM with sliced attn_weights.")
            # Pass the SLICED scores, but the FULL value tensor to be modified
            # local_cam_mask needs to know how to interpret the slice relative to original indices
            # We pass the necessary slice, the original function should work IF
            # its indexing logic for value_states is absolute and correct.
            # We need to ensure local_cam_mask receives the score slice corresponding
            # to the *current* query (q_len=1 assumption). Let's assume q_len=1 here.
            if q_len != 1:
                 print("Warning: CaM logic assumes q_len=1 for score slicing.")
            # Pass scores slice for the first (and assumed only) query token
            # value = local_cam_mask(value, full_sliced_attn_weights[:, :, 0, :], self.start_budget, self.recent_budget)
            value = local_cam_mask(
                value,                           # Full value tensor [B, H, kv_seq_len, D]
                full_sliced_attn_weights[:,:,0,:], # Sliced scores for q=0 [B, H, slice_width]
                kv_seq_len,                      # Pass the full sequence length
                self.start_budget,               # Pass budget info
                self.recent_budget               # Pass budget info
            )
            key_states, value_states = past_key_value.update(key_states, value, self.layer_idx)


        # --- Create Mask for *NEXT* Step ---
        # ... (Next mask creation remains the same) ...
        dtype_attn_weights = query_states.dtype
        attn_weights_devices = query_states.device
        next_mask = torch.zeros(self.num_heads, kv_seq_len, dtype=dtype_attn_weights, device=attn_weights_devices)
        if kv_seq_len > self.cache_budget:
            if self.recent_budget > 0: next_mask[:, -self.recent_budget:] = 1
            if self.start_budget > 0: next_mask[:, :self.start_budget] = 1
        else:
            next_mask[:, :] = 1
        self.attention_masks_next = next_mask.unsqueeze(0).unsqueeze(2)


        # --- Reshape and Output Projection ---
        # ... (Reshape and o_proj remain the same) ...
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)


        return attn_output, None




        # for h_start in range(0, self.num_heads, self.head_chunk_size):
        #     h_end = min(h_start + self.head_chunk_size, self.num_heads)
        #     num_heads_in_chunk = h_end - h_start

        #     # --- Slice Tensors for the Current Head Chunk ---
        #     q_chunk = query_states[:, h_start:h_end, :, :]
        #     k_chunk = key[:, h_start:h_end, :, :]
        #     v_chunk = value[:, h_start:h_end, :, :]

        #     # --- Calculate Attention Scores for Chunk ---
        #     # Shape: [bsz, num_heads_in_chunk, q_len, kv_seq_len]
        #     attn_weights_chunk = torch.matmul(q_chunk, k_chunk.transpose(2, 3)) * self.scaling

        #     # --- Apply Primary Attention Mask ---
        #     # attention_mask shape: [bsz, 1, q_len, kv_seq_len], broadcasts over head dim
        #     if attention_mask is not None:
        #         attn_weights_chunk = attn_weights_chunk + attention_mask
        #         attn_weights_chunk = torch.max(attn_weights_chunk, torch.tensor(torch.finfo(attn_weights_chunk.dtype).min))

        #     # --- Apply Secondary Custom Mask (self.attention_masks_next) ---
        #     if self.attention_masks_next is not None:
        #          # Original mask shape: [1, num_heads, 1, kv_seq_len+1], needs slicing & check length
        #          mask_next_full = self.attention_masks_next
        #          if mask_next_full.shape[-1] == kv_seq_len + 1: # Check if length matches expectation
        #              mask_next_chunk = mask_next_full[:, h_start:h_end, :, :kv_seq_len] # Slice heads and trim last dim if needed
        #              # Apply the mask (original logic was multiplicative + min value)
        #              attn_weights_chunk = attn_weights_chunk * mask_next_chunk + (1 - mask_next_chunk) * torch.finfo(attn_weights_chunk.dtype).min
        #          else:
        #               # Handle potential size mismatch if needed
        #               print(f"Warning: attention_masks_next shape {mask_next_full.shape} mismatch with kv_seq_len {kv_seq_len}")


        #     # --- Softmax ---
        #     attn_weights_chunk = nn.functional.softmax(attn_weights_chunk, dim=-1, dtype=torch.float32).to(q_chunk.dtype)

        #     # --- Calculate Output for Chunk ---
        #     # Shape: [bsz, num_heads_in_chunk, q_len, head_dim]
        #     attn_output_chunk = torch.matmul(attn_weights_chunk, v_chunk)

        #     # --- Place Chunk Output into Full Output Tensor ---
        #     attn_output[:, h_start:h_end, :, :] = attn_output_chunk

        #     # --- Clean up chunk tensors (optional, Python GC handles it) ---
        #     # del q_chunk, k_chunk, v_chunk, attn_weights_chunk, attn_output_chunk
        #     # torch.cuda.empty_cache() # Avoid excessive calls inside loop

        # # --- Post-Attention Processing ---

        # # --- CaM Merging (COMMENTED OUT - Requires Adaptation) ---
        # # The original CaM logic (`local_cam_mask`) required the full, post-softmax
        # # attn_weights matrix, which we avoided creating. It also modified value_states.
        # # Integrating CaM with chunking requires redesigning `local_cam_mask`
        # # or calculating/storing attention scores differently.
        # # if self.merge_token and self.attention_masks_next is not None:
        # #     print(f"Layer {self.layer_idx}: CaM requires adaptation for chunking - SKIPPING")
        #     # TODO: Adapt local_cam_mask
        #     # 1. It needs attn_weights. Maybe compute/store attn_weights_chunk and concat? (Memory issue?)
        #     # 2. Or redesign local_cam_mask to not need specific scores?
        #     # 3. Modifying v_chunk inside loop might not update original `value` if it's a copy.
        #     # value_states_merged = local_cam_mask(value, ??attn_weights??, self.start_budget, self.recent_budget)
        #     # Update cache IF CaM modified value_states and if caching is enabled
        #     # if use_cache and past_key_value is not None:
        #     #    past_key_value.key_cache[self.layer_idx] = key_states # Key wasn't modified by CaM
        #     #    past_key_value.value_cache[self.layer_idx] = ??value_states_merged?? # Need the correctly merged value state reference

        # # --- Create Mask for *NEXT* Step ---
        # # Create based on budgets and the *full* kv_seq_len
        # dtype_attn_weights = query_states.dtype # Use query dtype as reference
        # attn_weights_devices = query_states.device
        # # Note: Original size was kv_seq_len + 1. Ensure this is correct.
        # # Let's assume it should be kv_seq_len for masking the *current* keys/values in the *next* step. Adjust if needed.
        # next_mask = torch.zeros(self.num_heads, kv_seq_len, dtype=dtype_attn_weights, device=attn_weights_devices)

        # if kv_seq_len > self.cache_budget: # Only apply budget mask if cache exceeds budget
        #     # Activate most recent k-cache
        #     if self.recent_budget > 0:
        #         next_mask[:, -self.recent_budget:] = 1 # Mask for last 'recent_budget' tokens
        #     # Activate start budget
        #     if self.start_budget > 0:
        #         next_mask[:, :self.start_budget] = 1 # Mask for first 'start_budget' tokens
        # else:
        #     # If within budget, allow attention to all tokens
        #      next_mask[:, :] = 1


        # # Reshape for broadcasting: [1, num_heads, 1, kv_seq_len]
        # self.attention_masks_next = next_mask.unsqueeze(0).unsqueeze(2)


        # # --- Reshape and Output Projection ---
        # attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # attn_output = self.o_proj(attn_output)

        # # return attn_output, None


        # attn_weights = torch.matmul(query_states, key.transpose(2, 3)) / math.sqrt(self.head_dim)
        # if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
        #         f" {attn_weights.size()}"
        #     )
        # if attention_mask is not None:
        #     if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
        #         raise ValueError(
        #             f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
        #         )
        #     attn_weights = attn_weights + attention_mask
        #     attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        
        # if self.previous_scores == None:
        #     self.start_budget = int(self.start_budget_ratio * attn_weights.shape[-1])
        #     self.recent_budget = int(self.recent_budget_ratio * attn_weights.shape[-1])
        #     self.cache_budget = self.start_budget + self.recent_budget
        #     self.cache_budget_records.append(self.cache_budget)
        #     self.input_length.append(attn_weights.shape[-1])
        #     self.previous_scores = 1
        

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


class Qwen2DecoderLayerCam(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2AttentionCam(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Qwen2ModelCam(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayerCam(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # if use_cache and past_key_values is None:
        #     past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        # if using_sliding_window_cache or using_static_cache:
        #     target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        # else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2Config,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class Qwen2AttentionCam_orig(nn.Module):
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


class Qwen2AttentionCam_my_take(nn.Module):
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
        
        if self.previous_scores == None:
            self.start_budget = int(self.start_budget_ratio * query_states.shape[-2])
            self.recent_budget = int(self.recent_budget_ratio * query_states.shape[-2])
            self.cache_budget = self.start_budget + self.recent_budget
            self.cache_budget_records.append(self.cache_budget)
            self.input_length.append(query_states.shape[-2])
            self.previous_scores = 1

        # from here I will start a for loop to split my attention weight calculations
        # attn_weights_chunk contain rows of attention weights
        attn_output = []
        attn_weights = []
        attn_mask = []
        
        size = 2**11
        for start in range(0, query_states.shape[-2], size):
            attn_weights_chunk = torch.matmul(query_states, key.transpose(2, 3)[:,:,start:start+size]) / math.sqrt(self.head_dim)
            # removing rows that are not needed
            attn_weights_chunk = attn_weights_chunk[:,:, start:, :]
            if attention_mask is not None:
                attn_weights_chunk = attn_weights_chunk + attention_mask[:,:,start:,start+size]
                attn_weights_chunk = torch.max(attn_weights_chunk, torch.tensor(torch.finfo(attn_weights_chunk.dtype).min))
            
            if self.attention_masks_next is not None:
                #TODO: check attention_mask_next shapes
                attn_weights_chunk = attn_weights_chunk * self.attention_masks_next + (1 - self.attention_masks_next) * torch.finfo(attn_weights_chunk.dtype).min
            
            attn_weights_chunk = nn.functional.softmax(attn_weights_chunk, dim=-1, dtype=torch.float32).to(query_states.dtype)

            dtype_attn_weights = attn_weights_chunk.dtype
            attn_weights_devices = attn_weights_chunk.device
            attn_mask.append(torch.zeros(attn_weights_chunk.shape[1], attn_weights_chunk.shape[-1]+1).to(dtype_attn_weights).to(attn_weights_devices))

            attn_output.append(torch.matmul(attn_weights, value_states))

        
        

            
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

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from typing import Optional, Tuple

# Assuming:
# - Qwen2Config, Qwen2RotaryEmbedding, apply_rotary_pos_emb, repeat_kv, Cache (DynamicCache), local_cam_mask are defined
# - local_cam_mask is the MODIFIED version accepting kv_seq_len and sliced scores.
# - past_key_value.update(k, v, layer_idx) updates the cache internally and
#   returns the FULL sequence key and value states FOR THAT LAYER, shapes:
#   [bsz, num_key_value_heads, full_kv_seq_len, head_dim]

class Qwen2AttentionCam(nn.Module):
    """
    Chunked Attention with CaM integration attempt.
    Assumes DynamicCache updates internally and returns full K/V state.
    CaM modifies the value state which *should* persist in DynamicCache.
    """
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        if self.num_heads % self.num_key_value_heads != 0:
             raise ValueError("num_heads must be divisible by num_key_value_heads")
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout # Currently unused with chunking logic

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(config)

        # Configurable Ratios/Flags
        self.start_budget_ratio = config.start_ratio
        self.recent_budget_ratio = config.recent_ratio
        self.merge_token = config.merge_token # Enable/disable CaM

        # State variables
        self.attention_masks_next = None
        self.start_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None # Flag to initialize budgets
        self.input_length = []
        self.cache_budget_records = []

        # Chunking config
        self.head_chunk_size = 4 # Make this configurable if needed

    def _reset_masks(self):
        self.attention_masks_next = None
        self.start_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None
        self.input_length = []
        self.cache_budget_records = []

    # _shape not needed, reshape done in forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor]=None, # No longer used if rotary uses position_ids
        attention_mask: Optional[torch.Tensor] = None, # Expects [bsz, 1, q_len, kv_seq_len]
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None, # Assumed DynamicCache
        cache_position: Optional[torch.LongTensor] = None, # Might be used by cache.update
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        if output_attentions:
             # Cannot reliably output full attention weights with this chunking + CaM
             print("Warning: output_attentions=True is disabled for chunked CaM attention.")
             output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        # --- Projections ---
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Project K and V for the *current* tokens
        key_states_current = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states_current = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # --- Determine Total Sequence Length for RoPE ---
        # This needs the length *after* the current token is added to cache
        current_kv_seq_len = key_states_current.shape[2] # Should be q_len
        past_kv_len = 0
        if past_key_value is not None:
            past_kv_len = past_key_value.get_seq_length(self.layer_idx)
        # Total length the cache WILL have after update
        total_kv_seq_len = past_kv_len + current_kv_seq_len

        # --- Rotary Embeddings ---
        # Apply RoPE to the current query and key states
        # RoPE needs positions corresponding to the *final* sequence length
        cos, sin = self.rotary_emb(value_states_current, position_ids) # Use final length
        query_states, key_states_current_rotated = apply_rotary_pos_emb(query_states, key_states_current, cos, sin, position_ids)

        # --- KV Cache Update ---
        # Update the cache with the rotated K and original projected V for the current tokens
        # Assumes cache.update returns the FULL sequence K/V state *before* repeat_kv
        # And that the cache object manages persistence internally
        full_key_states = key_states_current_rotated # If no cache
        full_value_states = value_states_current   # If no cache
        if past_key_value is not None:
            if use_cache:
                # Pass current rotated K, original projected V
                full_key_states, full_value_states = past_key_value.update(
                    key_states_current_rotated,
                    value_states_current, # Pass non-rotated V to cache
                    self.layer_idx,
                    {"cache_position": cache_position} # Pass cache_position if needed by update
                )
            else: # If use_cache is False but cache object exists (e.g. reuse in generate)
                 full_key_states = key_states_current_rotated
                 full_value_states = value_states_current


        # --- Repeat KV states ---
        # Apply repeat_kv to the *full* sequence K/V obtained from cache
        key = repeat_kv(full_key_states, self.num_key_value_groups)
        value = repeat_kv(full_value_states, self.num_key_value_groups)

        # Final kv_seq_len after cache update and repeat
        kv_seq_len = key.shape[-2] # Should be same as total_kv_seq_len now

        # --- Calculate Budgets (Only Once or on Length Change) ---
        # Use final kv_seq_len
        # Note: Original code initialized budgets based on attn_weights shape AFTER first mask.
        # Here we use kv_seq_len directly.
        if self.previous_scores is None: # Initialize budgets
            self.start_budget = int(self.start_budget_ratio * kv_seq_len)
            self.recent_budget = int(self.recent_budget_ratio * kv_seq_len)
            # Add boundary checks for budgets
            self.recent_budget = max(2, min(self.recent_budget, kv_seq_len)) # Ensure at least 2 for CaM slicing
            self.start_budget = max(0, min(self.start_budget, kv_seq_len))
            self.cache_budget = self.start_budget + self.recent_budget
            self.cache_budget_records.append(self.cache_budget)
            self.input_length.append(kv_seq_len)
            self.previous_scores = 1
        elif kv_seq_len != self.input_length[-1]: # Update if length changed (optional, maybe budget is fixed)
             print(f"Warning: kv_seq_len changed ({self.input_length[-1]} -> {kv_seq_len}), consider updating budgets.")
             # Optionally recalculate budgets here if needed
             self.input_length.append(kv_seq_len)


        # --- Attention Calculation (Chunked) ---
        attn_output = torch.zeros_like(query_states) # [B, H, q_len, D]
        sliced_attn_weights_chunks = []

        # Define CaM slice indices (relative to full kv_seq_len)
        cam_slice_start_idx = max(0, kv_seq_len - self.recent_budget + 1)
        cam_slice_end_idx = kv_seq_len
        can_slice_for_cam = self.merge_token and (cam_slice_end_idx > cam_slice_start_idx)

        # Ensure primary attention mask has correct shape [B, 1, Q_len, K_len]
        if attention_mask is not None:
            expected_mask_shape = (bsz, 1, q_len, kv_seq_len)
            if attention_mask.shape != expected_mask_shape:
                 # Maybe resize or error? For now, error.
                 raise ValueError(f"attention_mask shape error. Expected {expected_mask_shape}, got {attention_mask.shape}")

        # Ensure secondary attention mask has correct shape [B, H, 1, K_len]
        # Correcting the presumed off-by-one error from original code.
        current_secondary_mask = None
        if self.attention_masks_next is not None:
             expected_sec_mask_shape = (bsz, self.num_heads, 1, kv_seq_len)
             if self.attention_masks_next.shape == expected_sec_mask_shape:
                  current_secondary_mask = self.attention_masks_next
             else:
                  # Handle mismatch - maybe slice if it was kv_seq_len+1?
                  if self.attention_masks_next.shape == (bsz, self.num_heads, 1, kv_seq_len + 1):
                       print("Warning: Slicing secondary mask from kv_seq_len+1 to kv_seq_len")
                       current_secondary_mask = self.attention_masks_next[:, :, :, :kv_seq_len]
                  else:
                       raise ValueError(f"secondary attention mask shape error. Expected {expected_sec_mask_shape} or [..., {kv_seq_len+1}], got {self.attention_masks_next.shape}")


        # --- Prepare Masks ---
        # Primary Mask
        if attention_mask is not None:
            expected_mask_shape = (bsz, 1, q_len, kv_seq_len) # Use current kv_seq_len
            if attention_mask.shape != expected_mask_shape:
                 raise ValueError(f"Primary attention_mask shape mismatch: Expected {expected_mask_shape}, got {attention_mask.shape}")

        # Secondary Mask (Applying the one from the *previous* step)
        current_secondary_mask = None
        if self.attention_masks_next is not None:
            # The mask from the PREVIOUS step dictates attention to PAST keys.
            # Its length should match the number of keys it was created for.
            mask_len = self.attention_masks_next.shape[-1]

            # Compare the mask length to the *current* kv_seq_len.
            # The mask should cover *all but the newest token*.
            if mask_len == kv_seq_len:
                 # This is the expected case: mask covers all keys for current step.
                 current_secondary_mask = self.attention_masks_next
                 print(f"DEBUG: Applying secondary mask with matching length {mask_len}") # Debug print
            elif mask_len == kv_seq_len - q_len:
                 # Mask is from previous step, doesn't cover current query's key/value yet.
                 # This is likely the scenario during generation (q_len=1).
                 # We need to apply it only to the past keys.
                 # Pad the mask to match the current kv_seq_len.
                 print(f"DEBUG: Padding secondary mask from {mask_len} to {kv_seq_len}") # Debug print
                 pad_width = kv_seq_len - mask_len
                 # Assume we allow attention to the newest token(s) by default (pad with 1)
                 padding = torch.ones(
                     (bsz, self.num_heads, q_len, pad_width), # Pad shape [B, H, Q, New_K_count]
                     dtype=self.attention_masks_next.dtype,
                     device=self.attention_masks_next.device
                 )
                 # We need to reshape the stored mask before padding if q_len > 1
                 # Stored mask shape [1, H, 1, K_past_len] needs to become [B, H, Q, K_past_len] ?
                 # Let's assume q_len=1 for simplicity here, as is typical for generation.
                 if q_len == 1:
                    current_secondary_mask = torch.cat(
                        (self.attention_masks_next, padding), dim=-1 # Concatenate along the key dimension
                    )
                 else:
                      # Handling prefill (q_len > 1) with this secondary mask requires
                      # careful consideration of how the mask should apply to multiple queries.
                      # For now, raise an error or disable secondary masking during prefill.
                      print(f"Warning: Secondary mask application logic assumes q_len=1, but got {q_len}. Disabling secondary mask for this step.")
                      current_secondary_mask = None

            else:
                 # Length mismatch is unexpected.
                 raise ValueError(
                     f"Secondary attention mask length mismatch. Mask has length {mask_len}, "
                     f"but current kv_seq_len is {kv_seq_len}. Expected mask length {kv_seq_len} or {kv_seq_len - q_len}."
                 )

        for h_start in range(0, self.num_heads, self.head_chunk_size):
            h_end = min(h_start + self.head_chunk_size, self.num_heads)

            # Slice Q, K, V for the current chunk of heads
            q_chunk = query_states[:, h_start:h_end, :, :]
            k_chunk = key[:, h_start:h_end, :, :]
            v_chunk = value[:, h_start:h_end, :, :] # Use repeated value

            # Calculate raw scores: [B, chunk_H, Q_len, K_len]
            attn_weights_chunk_raw = torch.matmul(q_chunk, k_chunk.transpose(2, 3)) * self.scaling

            # Apply primary mask (broadcasts heads)
            if attention_mask is not None:
                attn_weights_chunk_raw = attn_weights_chunk_raw + attention_mask

            # Apply secondary mask (slice heads)
            if current_secondary_mask is not None:
                mask_next_chunk = current_secondary_mask[:, h_start:h_end, :, :]
                # Apply multiplicative + min value logic from original
                attn_weights_chunk_raw = attn_weights_chunk_raw * mask_next_chunk + \
                                         (1 - mask_next_chunk) * torch.finfo(attn_weights_chunk_raw.dtype).min

            # Softmax
            attn_weights_chunk_softmax = nn.functional.softmax(attn_weights_chunk_raw, dim=-1, dtype=torch.float32).to(q_chunk.dtype)

            # Extract slice for CaM
            if can_slice_for_cam:
                attn_weights_slice = attn_weights_chunk_softmax[..., :, cam_slice_start_idx:cam_slice_end_idx]
                sliced_attn_weights_chunks.append(attn_weights_slice.cpu()) # Store slice (on CPU?)

            # Calculate output chunk: [B, chunk_H, Q_len, D]
            attn_output_chunk = torch.matmul(attn_weights_chunk_softmax, v_chunk)
            attn_output[:, h_start:h_end, :, :] = attn_output_chunk

            # del q_chunk, k_chunk, v_chunk, attn_weights_chunk_raw, attn_weights_chunk_softmax, attn_output_chunk


        # --- Post-Loop Processing ---

        # Reconstruct *Sliced* attn_weights for CaM
        full_sliced_attn_weights = None
        value_modified_by_cam = False
        if can_slice_for_cam and sliced_attn_weights_chunks:
            try:
                full_sliced_attn_weights = torch.cat(sliced_attn_weights_chunks, dim=1).to(query_states.device)
                del sliced_attn_weights_chunks
            except RuntimeError as e:
                print(f"Error concatenating sliced attn_weights chunks: {e}")
                can_slice_for_cam = False

        # Apply CaM if conditions met
        if can_slice_for_cam and full_sliced_attn_weights is not None:
            print(f"Layer {self.layer_idx}: Applying CaM.")
            if q_len != 1:
                print("Warning: CaM logic assumes q_len=1 for score slicing.")

            # Call the MODIFIED local_cam_mask
            # It needs to modify the 'value' tensor IN-PLACE.
            # Assuming local_cam_mask handles the in-place modification correctly.
            value = local_cam_mask(
                value,                           # Full REPEATED value tensor [B, H, kv_seq_len, D]
                full_sliced_attn_weights[:,:,0,:], # Sliced scores for q=0 [B, H, slice_width]
                kv_seq_len,                      # Full sequence length
                self.start_budget,
                self.recent_budget
            )
            modified_full_value_states = unrepeat_kv(
                value, # Pass the modified repeated tensor
                self.num_key_value_heads,
                self.num_heads
            )
            # 2. Access cache internals and overwrite the value state
            # WARNING: Relies on internal attribute name 'value_cache'
            if hasattr(past_key_value, 'value_cache') and isinstance(past_key_value.value_cache, list):
                    # Ensure tensor is on the correct device as the cache expects
                    past_key_value.value_cache[self.layer_idx] = modified_full_value_states.to(past_key_value.value_cache[self.layer_idx].device)
                    print(f"Layer {self.layer_idx}: Explicitly updated DynamicCache value state after CaM.")
            else:
                    print(f"Layer {self.layer_idx}: Warning: Could not access past_key_value.value_cache[{self.layer_idx}] to update after CaM.")

            # full_key_states, full_value_states = past_key_value.update(
            #         key,
            #         value, # Pass non-rotated V to cache
            #         self.layer_idx,
            #         {"cache_position": cache_position})
            # value_modified_by_cam = True
            # CRITICAL ASSUMPTION: Modifying 'value' here correctly affects the state held by DynamicCache.
            # This depends on DynamicCache using references and repeat_kv potentially creating views.
            # If CaM fails to persist, this assumption is likely wrong.


        # --- Create Mask for *NEXT* Step (Corrected Size) ---
        next_mask = torch.zeros(self.num_heads, kv_seq_len, dtype=query_states.dtype, device=query_states.device)
        if kv_seq_len > self.cache_budget:
            if self.recent_budget > 0: next_mask[:, -self.recent_budget:] = 1
            if self.start_budget > 0: next_mask[:, :self.start_budget] = 1
        else:
            next_mask[:, :] = 1
        # Store reshaped mask for the next step's secondary masking
        self.attention_masks_next = next_mask.unsqueeze(0).unsqueeze(2) # Shape [1, H, 1, kv_seq_len]


        # --- Reshape and Output Projection ---
        attn_output = attn_output.transpose(1, 2).contiguous() # [B, q_len, H, D]
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size) # [B, q_len, D_model]
        attn_output = self.o_proj(attn_output)

        # Return None for weights and cache tuple (cache managed internally)
        return attn_output, None


def convert_kvcache_qwen_cam(model, config, layer_idx=None):
    for name, module in reversed(model._modules.items()):
        if name.isnumeric():
            layer_idx = int(name)
        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_qwen_cam(module, config, layer_idx)

        if isinstance(module, Qwen2Attention):
            orig_state_dict = copy.deepcopy(module.state_dict())
            model._modules[name] = Qwen2AttentionCam_my_take(config, layer_idx=layer_idx)
            
            target_device = next(module.parameters()).device
            for param in model._modules[name].parameters():
                param.data = param.data.to(target_device)
            for buffer in model._modules[name].buffers():
                buffer.data = buffer.data.to(target_device)
            model._modules[name].half()
            model._modules[name].load_state_dict(orig_state_dict)
    return model

# checkpoint = copy.deepcopy(model.language_model.state_dict())
config = model.language_model.config
config.start_ratio = 0.01
config.recent_ratio = 0.1
config.merge_token = True

# model.language_model = convert_kvcache_qwen_cam(model.language_model, config)
# model.language_model.load_state_dict(checkpoint)
# del checkpoint
torch.cuda.empty_cache()

# model.language_model

processor = LlavaOnevisionProcessor.from_pretrained("models/llava-onevision-qwen2-7b-ov-hf", use_fast=True)


def apply_chat_template(conv, add_generation_prompt=True):
    """
    Creates a properly formatted prompt for the LLaVA OneVision model.

    Args:
        conv: The conversation dictionary containing roles and content.
        add_generation_prompt: Whether to add the assistant's generation prompt.

    Returns:
        A string containing the formatted prompt.
    """
    prompt = ""
    for message in conv:
        role = message["role"]
        content = message["content"]

        # Add the role-specific start token
        prompt += f"<|im_start|>{role}\n"

        # Add the content
        for item in content:
            if item["type"] == "text":
                prompt += item["text"] + "\n"
            elif item["type"] == "image":
                prompt += "<image>\n"

        # Add the end token
        prompt += "<|im_end|>"

    # Add the assistant's generation prompt if required
    if add_generation_prompt:
        prompt += "<|im_start|>assistant"

    return prompt


image_max_count = 20
image_dir = "datasets/MileBench/ActionLocalization/images/0KZYF"
images = [Image.open(os.path.join(image_dir,image_path)).convert('RGB')
                 for image_path in os.listdir(image_dir)][:image_max_count]

conv = [
    {
                    "role": "user",
                    "content": [
                        {"type": "text", 
                         "text": "Describe what is happening in these images?"}
                    ],
                }
            ]
conv[0]['content'].extend([{"type": "image"} for i in range(image_max_count)])

prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
# prompt = "user \nDescribe the event in these images? "+"<|im_start|><image><|im_end|> "*image_max_count + "assistant\n"
inputs = processor(images=images, text=prompt, return_tensors='pt').to('cuda', torch.float16)


with torch.no_grad():
    model.eval()
    past_key_values=SinkCache(9999999, num_sink_tokens=8)
    output = model.generate(**inputs, output_attentions=False, use_cache=True, max_new_tokens=999999, past_key_values=past_key_values, temperature = 1.0)
    print('here')

text = processor.decode(output[0], skip_special_tokens=True)
print(text)
# output = model.generate(inputs, max_new_tokens=1, use_cache=True)
# print(processor.decode(output[0], skip_special_tokens=True))
print('finished')