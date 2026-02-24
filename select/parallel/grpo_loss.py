"""
GRPO Loss Functions and Advantage Computation
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from time import time
from accelerate import Accelerator

def compute_grpo_advantages(
    rewards: torch.Tensor, 
    group_indices: np.ndarray, 
    norm_adv_by_std: bool = True,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Compute GRPO advantages using group-relative normalization.
    
    Args:
        rewards: Tensor of shape (batch_size,) containing rewards
        group_indices: Array of group indices for each sample
        norm_adv_by_std: Whether to normalize advantages by std
        epsilon: Small constant for numerical stability
        
    Returns:
        advantages: Tensor of shape (batch_size,) containing advantages
    """
    advantages = torch.zeros_like(rewards)
    
    # Group rewards by index
    id2rewards = defaultdict(list)
    id2indices = defaultdict(list)
    
    for i, group_id in enumerate(group_indices):
        id2rewards[group_id].append(rewards[i])
        id2indices[group_id].append(i)
    
    # Compute advantages for each group
    for group_id in id2rewards:
        group_rewards = torch.stack(id2rewards[group_id])
        group_mean = torch.mean(group_rewards)
        # group_mean = (group_mean * 1 + 0.5) / 2
        
        if norm_adv_by_std and len(group_rewards) > 1:
            group_std = torch.std(group_rewards)
            group_advantages = (group_rewards - group_mean) / (group_std + epsilon)
        else:
            group_advantages = group_rewards - group_mean
        
        # Assign advantages back to original indices
        for i, idx in enumerate(id2indices[group_id]):
            advantages[idx] = group_advantages[i]
    # print("Advantages", advantages)
    return advantages


def compute_policy_loss(
    old_log_probs: torch.Tensor, 
    new_log_probs: torch.Tensor, 
    advantages: torch.Tensor, 
    response_mask: torch.Tensor,
    clip_ratio: float = 0.2,
    accelerator: Accelerator = None
) -> torch.Tensor:
    """
    Compute PPO policy loss with clipping.
    
    Args:
        old_log_probs: Old log probabilities
        new_log_probs: New log probabilities
        advantages: Advantage estimates
        response_mask: Mask for response tokens
        clip_ratio: PPO clipping ratio
        
    Returns:
        policy_loss: Computed policy loss
    """
    # print("Compute policy loss", old_log_probs.shape, new_log_probs.shape, advantages.shape, response_mask.shape)
    start_time = time()
    # new_log_probs.register_hook(lambda grad: print(f"new_log_probs grad norm {torch.norm(grad)}"))
    # Compute ratio
    ratio = torch.exp(new_log_probs - old_log_probs)
    # ratio = ratio[response_mask == 1]
    # advantages = advantages[response_mask == 1]
    
    # Compute surrogate losses
    # print("Ratio Statistics: ", ratio[response_mask == 1].min(), ratio[response_mask == 1].max(), ratio[response_mask == 1].mean())
    # print(ratio)
    # print(new_log_probs)
    # print(old_log_probs)
    # sum = response_mask.sum()
    # print('sum', sum)
    # print('response_mask', response_mask[])
    # sum = accelerator.reduce(sum, reduction="sum")
    # if accelerator.is_main_process:
    #     print('sumfa ', sum)
    # exit()
    surr1 = ratio * advantages
    # surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    # print('rsum',response_mask.sum(dim=-1))
    
    # Take minimum (pessimistic bound)
    policy_loss = -surr1
    
    # Mask and average
    masked_loss = policy_loss * response_mask
    # masked_loss.register_hook(lambda grad: print(f"masked_loss grad norm {torch.norm(grad)}"))
    # if self.accelerator.is_main_process:
        # print("Compute policy loss time", time() - start_time)
    # exit()
    # if accelerator.is_main_process:
        # print('Advantages', advantages[0, 0])
    # print('fafa', masked_loss.sum() / response_mask.sum())
    return masked_loss.sum() / response_mask.sum()


def compute_kl_loss(
    old_log_probs: torch.Tensor, 
    new_log_probs: torch.Tensor, 
    response_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL divergence loss between old and new policies.
    
    Args:
        old_log_probs: Old log probabilities
        new_log_probs: New log probabilities
        response_mask: Mask for response tokens
        
    Returns:
        kl_loss: KL divergence loss
    """
    # print("Compute kl loss", old_log_probs.shape, new_log_probs.shape, response_mask.shape)
    start_time = time()
    kl_div = old_log_probs - new_log_probs
    masked_kl = kl_div * response_mask
    # print("Compute kl loss time", time() - start_time)
    return masked_kl.sum() / response_mask.sum()


# def compute_log_probs(
#     model: torch.nn.Module,
#     input_ids: torch.Tensor,      # (B, S)
#     attention_mask: torch.Tensor, # (B, S)
#     response_mask: torch.Tensor,  # (B, S)
#     chunk_size: int = 512,        # # of time steps per chunk
#     dtype: Optional[torch.dtype] = None,
# ) -> torch.Tensor:
#     """
#     Compute log-probs by:
#       1) one forward to get hidden_states[-1]: (B, S, H)
#       2) chunking along S so we only ever do (B, L, V) with L ≤ chunk_size.
    
#     Returns:
#       masked_log_probs: (B, S-1) zeroed where response_mask==0
#     """
#     device = next(model.parameters()).device
#     B, S = input_ids.shape
#     start_time = time()

#     # 1) Single forward to get hidden states
#     outputs = model(
#         input_ids=input_ids.to(device),
#         attention_mask=attention_mask.to(device),
#         output_hidden_states=True,
#         use_cache=False
#     )
#     # last hidden layer: (B, S, H)
#     last_hidden = outputs.hidden_states[-1]

#     # 2) Prepare for next-token scoring
#     hidden = last_hidden[:, :-1, :]         # (B, S-1, H)
#     labels = input_ids[:, 1:].to(device)    # (B, S-1)
#     mask   = response_mask.to(device) # (B, S-1)

#     B, T, H = hidden.shape
#     lm_head: torch.nn.Linear = model.get_output_embeddings()
#     weight = lm_head.weight                 # (V, H)
#     bias   = lm_head.bias                   # (V,) or None
#     model_device = input_ids.device
#     device = 'cuda:7'
#     hidden = hidden.to(device)

#     # Preallocate result
#     masked_log_probs = torch.zeros(B, T, device=device, dtype=torch.float32)

#     # 3) Chunk along the sequence dimension
#     for start in range(0, T, chunk_size):
#         end = min(start + chunk_size, T)
#         hidden_chunk = hidden[:, start:end, :]   # (B, L, H)
#         labels_chunk = labels[:, start:end]      # (B, L)
#         mask_chunk   = mask[:, start:end]        # (B, L)
#         # print("hidden_chunk", hidden_chunk.shape, labels_chunk.shape, mask_chunk.shape)

#         # flatten batch & time: (B*L, H)
#         hidden_flat = hidden_chunk.reshape(-1, H)
#         labels_flat = labels_chunk.reshape(-1)

#         # 4) project to logits for only true labels
#         #    we could do hidden_flat @ weight.T + bias, but to get full distribution:
#         logits_chunk = hidden_flat @ weight.T    # (B*L, V)
#         if bias is not None:
#             logits_chunk = logits_chunk + bias.unsqueeze(0)

#         if dtype is not None:
#             logits_chunk = logits_chunk.to(dtype)

#         # 5) compute log_softmax and gather
#         logp = F.log_softmax(logits_chunk, dim=-1)             # (B*L, V)
#         gathered = torch.gather(logp.to(model_device), 1, labels_flat.unsqueeze(-1))  # (B*L, 1)
#         gathered = gathered.squeeze(1).view(B, end-start)      # (B, L)

#         # 6) apply mask and write to output
#         print("gathered", gathered.shape, mask_chunk.shape)
#         masked_log_probs[:, start:end] = gathered * mask_chunk
#     print("Compute log probs time", time() - start_time)
#     return masked_log_probs.to(model_device)

def compute_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor, 
    attention_mask: torch.Tensor, 
    response_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute log probabilities for given sequences.
    
    Args:
        model: Language model
        input_ids: Input token IDs
        attention_mask: Attention mask
        response_mask: Response mask
        
    Returns:
        log_probs: Log probabilities for response tokens
    """
    # print("Compute log probs", input_ids.shape, attention_mask.shape, response_mask.shape)
    start_time = time()
    # Get model device
    model_device = next(model.parameters()).device
    
    # Move tensors to model device
    input_ids = input_ids.to(model_device)
    attention_mask = attention_mask.to(model_device)
    response_mask = response_mask.to(model_device)
    
    # with torch.no_grad():
    # device = 'cuda:7'
    outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits#.to(device)
    logits = outputs
    
    # Shift logits and input_ids for next token prediction
    # seq_batch_size = 2048
    shift_logits = logits[..., :-1, :]#.contiguous()
    shift_labels = input_ids[..., 1:]#.to(device)#.contiguous()
    
    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Mask to only include response tokens
    # response_mask = response_mask[..., 1:].contiguous()
    masked_log_probs = gathered_log_probs * response_mask#.to(device)
    # print("Compute log probs time", time() - start_time)
    return masked_log_probs#.to(model_device)