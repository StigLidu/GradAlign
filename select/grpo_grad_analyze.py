"""
Lightweight GRPO (Group Relative Policy Optimization) Gradient Analyzer
Based on the verl implementation and the paper: https://arxiv.org/pdf/2402.03300

Modified to analyze gradients instead of training.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW, Adam
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import numpy as np
import random
from time import time
import os
import json
from collections import defaultdict
from grpo_utils import print_gpu_memory

from grpo_loss import (
    compute_grpo_advantages, 
    compute_policy_loss, 
    compute_kl_loss, 
    compute_log_probs
)
from grpo_utils import (
    collate_batch,
    create_response_mask,
    print_training_stats,
    save_model_checkpoint,
    load_model_checkpoint,
    check_model_for_nan
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GRPOGradientAnalyzer:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = None,
        device: str = "cuda",
        learning_rate: float = 1e-6,
        kl_loss_coef: float = 0.001,
        clip_ratio: float = 0.2,
        norm_adv_by_std: bool = True,
        max_length: int = 1024,
        n_samples_per_prompt: int = 5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 32,
        epsilon: float = 1e-6,
    ):
        """
        Initialize GRPO gradient analyzer.
        
        Args:
            model_path: Path to the model to analyze
            tokenizer_path: Path to tokenizer (defaults to model_path)
            device: Device to use for analysis
            learning_rate: Learning rate for optimizer (not used for analysis)
            kl_loss_coef: Coefficient for KL divergence loss
            clip_ratio: PPO clipping ratio
            norm_adv_by_std: Whether to normalize advantages by std
            max_length: Maximum sequence length
            n_samples_per_prompt: Number of samples per prompt (group size)
            ppo_epochs: Number of PPO epochs per update (not used for analysis)
            mini_batch_size: Mini batch size for analysis
            epsilon: Small constant for numerical stability
        """
        self.device = device
        self.learning_rate = learning_rate
        self.kl_loss_coef = kl_loss_coef
        self.clip_ratio = clip_ratio
        self.norm_adv_by_std = norm_adv_by_std
        self.max_length = max_length
        self.n_samples_per_prompt = n_samples_per_prompt
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.epsilon = epsilon
        self.dtype = torch.float32
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map="auto"
        )
        self.model.train()  # Set to train mode for gradient computations
        self.ignored_tasks = []
        
        # # Create reference model (frozen copy)
        # self.ref_model = AutoModelForCausalLM.from_pretrained(
        #     model_path,
        #     torch_dtype=self.dtype,
        #     device_map="auto"
        # )
        # self.ref_model.eval()
        # for param in self.ref_model.parameters():
        #     param.requires_grad = False
        
        # Setup optimizer (not used for gradient analysis but needed for structure)
        # self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        logger.info(f"Initialized GRPO gradient analyzer with model: {model_path}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def load_responses_from_disk(self, responses_dir: str, max_samples: int = None) -> Tuple[List[str], List[str], List[int]]:
        """Load pre-generated responses from disk.
        
        Args:
            responses_dir: Directory containing responses.json file
            
        Returns:
            Tuple of (prompts, responses, group_indices)
        """
        responses_file = os.path.join(responses_dir, 'responses.json')
        if not os.path.exists(responses_file):
            raise FileNotFoundError(f"Responses file not found: {responses_file}")
        
        print(f"Loading responses from: {responses_file}")
        with open(responses_file, 'r') as f:
            responses_data = json.load(f)
        
        prompts = []
        responses = []
        group_indices = []
        
        for item in responses_data[:max_samples]:
            prompts.append(item['prompt'])
            responses.append(item['response'])
            group_indices.append(item['group_id'])
        
        print(f"Loaded {len(responses)} responses from {len(set(group_indices))} groups")
        
        # Load metadata for verification
        metadata_file = os.path.join(responses_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"Response generation metadata:")
            print(f"  Model: {metadata.get('model_path', 'Unknown')}")
            print(f"  Temperature: {metadata.get('temperature', 'Unknown')}")
            print(f"  Samples per prompt: {metadata.get('n_samples', 'Unknown')}")
            print(f"  Generated at: {metadata.get('timestamp', 'Unknown')}")
        
        return prompts, responses, group_indices
    
    def prepare_data(self, prompts: List[str], responses: List[str], 
                    rewards: List[float], group_indices: List[int]) -> List[Dict[str, torch.Tensor]]:
        """Prepare data for gradient analysis."""
        # Convert to tensors
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        group_indices_array = np.array(group_indices)
        print('Start to compute advantages')
        print_gpu_memory()
        
        # Compute advantages
        # print('\n\n\nrewards_tensor\n\n\n', rewards_tensor)
        advantages = compute_grpo_advantages(
            rewards_tensor, 
            group_indices_array, 
            self.norm_adv_by_std,
            self.epsilon
        ).detach()
        
        # Prepare input sequences
        batch_data = []
        print('Start to append batch_data')
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            full_text = prompt + response
            
            # Tokenize
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length
            )
            
            # Create response mask
            response_mask = create_response_mask(self.tokenizer, prompt, full_text)
            if response_mask.shape[1] > inputs['input_ids'].shape[1] - 1:
                response_mask = response_mask[:, :inputs['input_ids'].shape[1] - 1]
            if abs(advantages[i]) < 1e-3:
                self.ignored_tasks.append(group_indices[i])
                continue
            
            batch_data.append({
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'response_mask': response_mask.squeeze(0),
                'advantages': advantages[i],
                'group_id': group_indices[i]
            })
        print_gpu_memory()
        
        # Process in mini-batches to compute log probs
        for i in range(0, len(batch_data), self.mini_batch_size):
            mini_batch = batch_data[i:i + self.mini_batch_size]
            
            # Pad batch using same collate function 
            print('Start to collate batch')
            print_gpu_memory()
            padded_batch = collate_batch(mini_batch, self.tokenizer.pad_token_id)
            print_gpu_memory()
            
            # Get model device (handles device_map="auto" case)
            model_device = next(self.model.parameters()).device
            
            # Move to device for batch computation
            input_ids = padded_batch['input_ids'].to(model_device)
            attention_mask = padded_batch['attention_mask'].to(model_device)
            response_mask = padded_batch['response_mask'].to(model_device)
            
            # Compute log probs in batch
            # with torch.no_grad():
            #     print('Start to compute log probs')
            #     print_gpu_memory()
            #     batch_old_log_probs = compute_log_probs(
            #         self.model, input_ids, attention_mask, response_mask
            #     ).cpu()
            #     print_gpu_memory()
                # batch_ref_log_probs = compute_log_probs(
                #     self.ref_model, input_ids, attention_mask, response_mask
                # ).cpu()
            
            # Add log probs back to the original batch_data
            # for j, data_idx in enumerate(range(i, min(i + self.mini_batch_size, len(batch_data)))):
            #     length = batch_data[data_idx]['input_ids'].shape[0] - 1
            #     batch_data[data_idx]['old_log_probs'] = batch_old_log_probs[j][:length]
                # batch_data[data_idx]['ref_log_probs'] = batch_ref_log_probs[j][:length]
                # print('fafa', data_idx, batch_data[data_idx]['old_log_probs'].shape, batch_data[data_idx]['input_ids'].shape)
        
        return batch_data
    
    def compute_loss_and_gradients(self, batch_data: List[Dict[str, torch.Tensor]]) -> Tuple[float, Dict[str, torch.Tensor]]:
        """Compute loss and gradients for a batch of data.
        
        Returns:
            Tuple of (loss_value, gradients_dict)
        """
        # Clear any existing gradients
        # self.optimizer.zero_grad()
        self.model.zero_grad()
        # for param in self.model.parameters():
        #     if param.grad is not None:
        #         param.grad.zero_()
        #         param.requires_grad = True
        print("Compute loss and gradients", len(batch_data))
        
        total_loss = 0.0
        num_batches = 0
        
        # Process data in mini-batches
        for i in range(0, len(batch_data), self.mini_batch_size):
            mini_batch = batch_data[i:i + self.mini_batch_size]
            
            # Pad batch
            padded_batch = collate_batch(mini_batch, self.tokenizer.pad_token_id)
            
            # Get model device
            model_device = next(self.model.parameters()).device
            
            input_ids = padded_batch['input_ids'].to(model_device)
            attention_mask = padded_batch['attention_mask'].to(model_device)
            response_mask = padded_batch['response_mask'].to(model_device)
            # old_log_probs = padded_batch['old_log_probs'].to(model_device)
            # ref_log_probs = padded_batch['ref_log_probs'].to(model_device)
            advantages = padded_batch['advantages'].to(model_device)
            
            # print('Advantages', advantages)
            
            # Compute new log probabilities
            new_log_probs = compute_log_probs(
                self.model, input_ids, attention_mask, response_mask
            )[..., 1:]
            old_log_probs = new_log_probs.detach()
            
            # Adjust tensors for shifted sequences
            response_mask = response_mask[..., 1:]
            print('response_mask', input_ids.shape, response_mask.shape)
            # old_log_probs = old_log_probs[..., 1:]
            # ref_log_probs = ref_log_probs[..., 1:]
            
            # Broadcast advantages to token level
            advantages_expanded = advantages.unsqueeze(-1).expand_as(response_mask)
            
            # Compute losses
            # print('shape', old_log_probs.shape, new_log_probs.shape, advantages_expanded.shape, response_mask.shape, input_ids.shape, attention_mask.shape)
            policy_loss = compute_policy_loss(
                old_log_probs, new_log_probs, advantages_expanded, response_mask, self.clip_ratio
            )
            # kl_loss = compute_kl_loss(ref_log_probs, new_log_probs, response_mask)
            
            # Total loss for this batch
            batch_loss = policy_loss #+ self.kl_loss_coef * kl_loss
            
            # Accumulate loss (will be averaged later)
            total_loss += batch_loss.item()
            num_batches += 1
            
            # Backward pass to accumulate gradients
            start_time = time()
            batch_loss.backward()
            print('Backward pass time', time() - start_time)
        
        # Average the loss
        avg_loss = total_loss / num_batches
        print('Total loss', total_loss)
        
        # Extract gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone().detach()
        
        return avg_loss, gradients
    
    def compute_cosine_similarity(self, grad1: Dict[str, torch.Tensor], grad2: Dict[str, torch.Tensor]) -> float:
        """Compute cosine similarity between two gradient dictionaries."""
        # Flatten all gradients into single vectors
        vec1 = []
        vec2 = []
        _start = time()
        sum = 0
        norm1 = 0
        norm2 = 0
        contribution = []
        for name in grad1.keys():
            if name in grad2:
                # _start1 = time()
                v1 = grad1[name].flatten().to('cuda:7')
                # print('move1', time() - _start1)
                v2 = grad2[name].flatten().to('cuda:7')
                sumv = torch.sum(v1 * v2)
                # print('move', time() - _start1)
                print(f"grad of {name} g1 {torch.sum(v1 * v1)} g2 {torch.sum(v2 * v2)} g12 {sumv} shape {v1.shape} {v2.shape}")
                contribution.append((abs(sumv.item()), sumv.item(), name))
                # if "lm_head.weight" in name:
                #     continue
                sum += sumv
                norm1 += torch.sum(v1 ** 2)
                norm2 += torch.sum(v2 ** 2)
                # vec1.append(grad1[name].flatten().to('cuda:6'))
                # vec2.append(grad2[name].flatten().to('cuda:6'))
        # if not vec1:
        #     return 0.0
        contribution.sort(key=lambda x: x[0], reverse=True)
        print('Contribution', contribution)
        
        # Concatenate all gradient vectors
        # vec1 = torch.cat(vec1)
        # vec2 = torch.cat(vec2)

        # grad_norm1 = torch.norm(vec1, p=2)
        # grad_norm2 = torch.norm(vec2, p=2)
        grad_norm1 = torch.sqrt(norm1)
        grad_norm2 = torch.sqrt(norm2)
        print('grad_norm1', grad_norm1, 'grad_norm2', grad_norm2)
        print('Cosine similarity time', time() - _start)
        
        # Compute cosine similarity
        cosine_sim = sum #torch.dot(vec1, vec2)
        # cosine_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0), dim=1, eps=0.1)
        return cosine_sim.item()
    
    def analyze_gradients(self, train_data: List[Dict], val_data: List[Dict]) -> Dict[str, float]:
        """Analyze gradients between validation data and train groups.
        
        Args:
            train_data: Training data with group information
            val_data: Validation data
            
        Returns:
            Dictionary mapping group_id -> cosine_similarity
        """
        print("Computing validation gradients...")
        val_loss, val_gradients = self.compute_loss_and_gradients(val_data)
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                val_gradients[name] = param.grad.detach().to('cuda:7')
        print(f"Validation loss: {val_loss:.4f}")
        
        # Group training data by group_id
        train_groups = defaultdict(list)
        for item in train_data:
            train_groups[item['group_id']].append(item)
        
        print(f"Found {len(train_groups)} training groups")
        
        # Compute gradients for each training group
        similarities = {}
        for group_id, group_data in tqdm(train_groups.items(), desc="Analyzing train groups"):
            print(f"\n\n\nProcessing group {group_id} ({len(group_data)} samples)...\n\n\n")
            group_loss, group_gradients = self.compute_loss_and_gradients(group_data)
            
            # Compute cosine similarity with validation gradients
            similarity = self.compute_cosine_similarity(val_gradients, group_gradients)
            similarities[group_id] = similarity
            
            print(f"Group {group_id}: loss={group_loss:.4f}, similarity={similarity:.4f}")
        
        
        for i in self.ignored_tasks:
            if not i in similarities:
                similarities[i] = 0.0
        return similarities
    
    # Keep existing methods for compatibility
    def rollout(self, prompts: List[str], responses: List[str], 
                rewards: List[float], group_indices: List[int]) -> List[Dict[str, torch.Tensor]]:
        """Alias for prepare_data for backward compatibility."""
        return self.prepare_data(prompts, responses, rewards, group_indices)
    
    def save_model(self, save_path: str):
        """Save the model."""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")


# Alias for backward compatibility
GRPOTrainer = GRPOGradientAnalyzer 