"""
Lightweight GRPO (Group Relative Policy Optimization) Trainer
Based on the verl implementation and the paper: https://arxiv.org/pdf/2402.03300

Modified to work with pre-generated responses from separate inference script.
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
import os
import json

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


class GRPOTrainer:
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
        Initialize GRPO trainer.
        
        Args:
            model_path: Path to the model to train
            tokenizer_path: Path to tokenizer (defaults to model_path)
            device: Device to use for training
            learning_rate: Learning rate for optimizer
            kl_loss_coef: Coefficient for KL divergence loss
            clip_ratio: PPO clipping ratio
            norm_adv_by_std: Whether to normalize advantages by std
            max_length: Maximum sequence length
            n_samples_per_prompt: Number of samples per prompt (group size)
            ppo_epochs: Number of PPO epochs per update
            mini_batch_size: Mini batch size for training
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
        self.dtype = torch.bfloat16
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map="auto"
        )
        self.model.train()
        
        # Create reference model (frozen copy)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map="auto"
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Setup optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        logger.info(f"Initialized GRPO trainer with model: {model_path}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def load_responses_from_disk(self, responses_dir: str) -> Tuple[List[str], List[str], List[int]]:
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
        
        for item in responses_data:
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
    
    def rollout(self, prompts: List[str], responses: List[str], 
                rewards: List[float], group_indices: List[int]) -> List[Dict[str, torch.Tensor]]:
        """Perform rollout and prepare training data for GRPO."""
        # Convert to tensors
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        group_indices_array = np.array(group_indices)
        
        # Compute advantages
        advantages = compute_grpo_advantages(
            rewards_tensor, 
            group_indices_array, 
            self.norm_adv_by_std,
            self.epsilon
        )
        
        # Prepare input sequences (same as before but without log probs)
        batch_data = []
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
            
            batch_data.append({
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'response_mask': response_mask.squeeze(0),
                'advantages': advantages[i]
            })
        
        # Process in mini-batches like train_step
        for i in range(0, len(batch_data), self.mini_batch_size):
            mini_batch = batch_data[i:i + self.mini_batch_size]
            
            # Pad batch using same collate function as train_step
            padded_batch = collate_batch(mini_batch, self.tokenizer.pad_token_id)
            
            # Get model device (handles device_map="auto" case)
            model_device = next(self.model.parameters()).device
            
            # Move to device for batch computation
            input_ids = padded_batch['input_ids'].to(model_device)
            attention_mask = padded_batch['attention_mask'].to(model_device)
            response_mask = padded_batch['response_mask'].to(model_device)
            
            # Compute log probs in batch
            with torch.no_grad():
                batch_old_log_probs = compute_log_probs(
                    self.model, input_ids, attention_mask, response_mask
                ).cpu()
                
                batch_ref_log_probs = compute_log_probs(
                    self.ref_model, input_ids, attention_mask, response_mask
                ).cpu()
            
            # Add log probs back to the original batch_data
            for j, data_idx in enumerate(range(i, min(i + self.mini_batch_size, len(batch_data)))):
                batch_data[data_idx]['old_log_probs'] = batch_old_log_probs[j]
                batch_data[data_idx]['ref_log_probs'] = batch_ref_log_probs[j]
        
        return batch_data
    
    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step."""
        # Get model device (handles device_map="auto" case)
        model_device = next(self.model.parameters()).device
        
        input_ids = batch_data['input_ids'].to(model_device)
        attention_mask = batch_data['attention_mask'].to(model_device)
        response_mask = batch_data['response_mask'].to(model_device)
        old_log_probs = batch_data['old_log_probs'].to(model_device)
        ref_log_probs = batch_data['ref_log_probs'].to(model_device)
        advantages = batch_data['advantages'].to(model_device)
        
        # Compute new log probabilities
        new_log_probs = compute_log_probs(
            self.model, input_ids, attention_mask, response_mask
        )[..., 1:]#.contiguous()
        
        # Adjust tensors for shifted sequences
        # print("response_mask: ", response_mask.shape, "old_log_probs: ", old_log_probs.shape, "ref_log_probs: ", ref_log_probs.shape)
        response_mask = response_mask[..., 1:]#.contiguous()
        old_log_probs = old_log_probs[..., 1:]#.contiguous()
        ref_log_probs = ref_log_probs[..., 1:]#.contiguous()
        
        # Broadcast advantages to token level
        torch.set_printoptions(threshold=1000000, precision=4)
        advantages_expanded = advantages.unsqueeze(-1).expand_as(response_mask)
        # print(advantages_expanded)
        
        # Compute losses
        # ratio = torch.exp(new_log_probs - old_log_probs)
        # print(self.tokenizer.decode(input_ids[0][1:][response_mask[0] == 1], skip_special_tokens=False))
        # for i in range(0, input_ids.shape[1], 5):
        #     print(self.tokenizer.decode(input_ids[0, i:i+5], skip_special_tokens=False), ratio[0, i:i+5], new_log_probs[0, i:i+5], old_log_probs[0, i:i+5], response_mask[0, i:i+5])
        policy_loss = compute_policy_loss(
            old_log_probs, new_log_probs, advantages_expanded, response_mask, self.clip_ratio
        )
        kl_loss = compute_kl_loss(ref_log_probs, new_log_probs, response_mask)
        
        # Total loss
        total_loss = policy_loss + self.kl_loss_coef * kl_loss
        print("Total loss: ", total_loss.item(), "policy_loss: ", policy_loss.item(), "kl_loss: ", kl_loss.item())
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        # print('fafa', check_model_for_nan(self.model))
        # norm = 0
        # for name, param in self.model.named_parameters():
        #     if torch.isnan(param.grad).any():
        #         print(f"NaN found in grad: {name}")
        # for param in self.model.parameters():
        #     norm += (param.grad.cpu() ** 2).sum()
        # norm = norm ** 0.5
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        print("Norm: ", norm)
        self.optimizer.step()
        # if check_model_for_nan(self.model):
        #     print("NaN found in model")
        
        return {
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def train(self, train_data: List[Dict], num_epochs: int = 1, save_path: str = None):
        """Train the model using GRPO."""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Shuffle data
            random.shuffle(train_data)
            
            epoch_losses = []
            
            # Create mini-batches
            for i in tqdm(range(0, len(train_data), self.mini_batch_size), desc="Training"):
                batch = train_data[i:i + self.mini_batch_size]
                
                # Pad batch
                batch_data = collate_batch(batch, self.tokenizer.pad_token_id)
                
                # Multiple PPO epochs on this batch
                for _ in range(self.ppo_epochs):
                    losses = self.train_step(batch_data)
                    epoch_losses.append(losses)
            
            # Print epoch statistics
            print_training_stats(epoch_losses, epoch + 1)
            
            # Save checkpoint
            if save_path:
                save_model_checkpoint(self.model, self.tokenizer, self.optimizer, epoch + 1, save_path)
    
    def save_model(self, save_path: str):
        """Save the trained model."""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load a trained model."""
        self.model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=self.dtype,
            device_map={"": self.device}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        
        # Re-initialize vLLM with the new model (if enabled)
        # This part is removed as vLLM is no longer used.
        # if self.use_vllm:
        #     # Use data parallelism: tensor_parallel_size=1 means each GPU gets full model
        #     num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            
        #     print(f"Re-initializing vLLM with data parallelism (tensor_parallel_size={self.tensor_parallel_size})")
        #     print(f"Available GPUs: {num_gpus}")
            
        #     self.vllm_model = LLM(
        #         model=load_path,
        #         tokenizer=load_path,
        #         dtype=self.dtype,
        #         max_model_len=self.max_length,
        #         gpu_memory_utilization=0.8,
        #         tensor_parallel_size=self.tensor_parallel_size,
        #         max_num_seqs=256,
        #     )
        # else:
        #     self.vllm_model = None
        
        logger.info(f"Model loaded from {load_path}") 