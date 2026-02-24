"""
Utility functions for GRPO training
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import re
from utils import compare_answers, extract_answer

def check_model_for_nan(model):
    """Check if any model parameters contain NaN values."""
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN found in parameter: {name}")
            return True
    return False

def collate_batch(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching training data.
    
    Args:
        batch: List of training samples
        pad_token_id: Token ID for padding
        
    Returns:
        collated: Collated batch data
    """
    # Find max length in batch
    max_len = max(item['input_ids'].shape[0] for item in batch)
    
    # Pad sequences
    collated = {}
    for key in batch[0].keys():
        if key == 'advantages':
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            padded_tensors = []
            for item in batch:
                tensor = item[key]
                if type(tensor) == int:
                    padded_tensors.append(torch.tensor(tensor))
                    continue
                pad_length = max_len - tensor.shape[0]
                # print("Pad length: ", pad_length, key)
                if pad_length > 0:
                    if key in ['input_ids', 'attention_mask']:
                        padded = F.pad(tensor, (0, pad_length), value=pad_token_id)
                    else:
                        if pad_length > 1:
                            padded = F.pad(tensor, (0, pad_length - 1), value=0)
                        else:
                            padded = tensor
                else:
                    padded = tensor
                padded_tensors.append(padded)
            collated[key] = torch.stack(padded_tensors)
    
    return collated


def create_response_mask(tokenizer, prompt: str, full_text: str) -> torch.Tensor:
    """
    Create a mask for response tokens (excluding prompt tokens).
    
    Args:
        tokenizer: Tokenizer instance
        prompt: Original prompt text
        full_text: Full text including prompt and response
        
    Returns:
        response_mask: Mask tensor where 1 indicates response tokens
    """
    # Tokenize full text
    full_tokens = tokenizer(full_text, return_tensors="pt", padding=False)
    
    # Tokenize prompt only
    prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=False)
    prompt_length = prompt_tokens['input_ids'].shape[1]
    
    # Create mask
    response_mask = torch.zeros_like(full_tokens['input_ids'])
    response_mask[0, prompt_length:] = 1
    response_mask[full_tokens['input_ids'] == tokenizer.pad_token_id] = 0
    # print("debug", tokenizer.decode(full_tokens['input_ids'][0, ], skip_special_tokens=False))
    
    return response_mask[..., 1:]


def simple_reward_function(prompt: str, response: str) -> float:
    """
    Simple rule-based reward function for general responses.
    
    Args:
        prompt: Input prompt
        response: Generated response
        
    Returns:
        reward: Computed reward score (0-1 scale)
    """
    response = response.strip()
    
    # Rule 1: Must have minimum length
    if len(response) < 5:
        return 0.1
    
    # Rule 2: Word count bonus (capped at 20 words)
    word_count = min(len(response.split()), 20)
    reward = word_count * 0.03
    
    # Rule 3: Contains punctuation (shows structure)
    if any(p in response for p in '.!?'):
        reward += 0.2
    
    # Rule 4: Not just repetition
    words = response.split()
    unique_words = set(words)
    if len(unique_words) > len(words) * 0.7:  # At least 70% unique words
        reward += 0.2
    
    return min(1.0, reward)  # Cap at 1.0


def math_reward_function(prompt: str, response: str, expected_answer: str = None) -> float:
    """
    Rule-based reward function for mathematical responses that extracts boxed answers.
    
    Args:
        prompt: Input prompt
        response: Generated response
        expected_answer: Expected correct answer (optional)
        
    Returns:
        reward: Computed reward score (0-1 scale)
    """
    response = response.strip()
    
    # Extract the predicted answer using utility function
    predicted_answer = extract_answer(response)
    
    # If we have an expected answer, compare using utility function
    # print('predicted_answer', predicted_answer, 'expected_answer', expected_answer)
    if expected_answer and predicted_answer:
        is_correct = compare_answers(predicted_answer, expected_answer)
        return 1.0 if is_correct else 0.0
    
    return 0.0

def print_training_stats(losses: List[Dict[str, float]], epoch: int):
    """
    Print training statistics.
    
    Args:
        losses: List of loss dictionaries
        epoch: Current epoch number
    """
    if not losses:
        return
    
    avg_losses = {}
    for key in losses[0].keys():
        avg_losses[key] = sum(loss[key] for loss in losses) / len(losses)
    
    print(f"Epoch {epoch} - " + 
          ", ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()]))


def save_model_checkpoint(model, tokenizer, optimizer, epoch: int, save_path: str):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        optimizer: Optimizer to save
        epoch: Current epoch
        save_path: Path to save checkpoint
    """
    import os
    
    checkpoint_path = f"{save_path}/checkpoint_epoch_{epoch}"
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    
    # Save optimizer state
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"{checkpoint_path}/optimizer.pt")
    
    print(f"Saved checkpoint to {checkpoint_path}")



def create_prompt(problem: str, tokenizer) -> str:
    """
    Create a prompt for the model to solve the math problem using the model's chat template
    """
    
    # Create messages in the format expected by the chat template
    messages = [
        {
            "role": "user", 
            "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{problem}"
        }
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return formatted_prompt

def create_math_prompt_simple(problem: str) -> str:
    """Create a simple prompt template for math problems."""
    return f"{problem}\n\nPlease solve this step by step and put your final answer in \\boxed{{}}."


def parse_math_data_with_answers(data_item: dict) -> tuple:
    """Parse math data item to extract prompt and expected answer."""
    if isinstance(data_item, dict):
        problem = data_item.get('problem', data_item.get('prompt', ''))
        answer = data_item.get('answer', data_item.get('expected', ''))
        prompt = create_math_prompt_simple(problem)
        return prompt, answer
    else:
        # Fallback for string inputs
        return str(data_item), None


def print_gpu_memory():
    """
    Print the used GPU memory for GPU 0.
    """
    return
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return
    
    
    # Get memory info for GPU 0
    cnt = torch.cuda.device_count()
    cnt = 1
    for i in range(cnt):
        device = torch.device(f'cuda:{i}')
        memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)  # Convert to GB
        memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)   # Convert to GB
        # memory_cached = torch.cuda.memory_cached(device) / (1024**3)       # Convert to GB
    
        print(f"GPU {i} Memory Usage:")
        print(f"  Allocated: {memory_allocated:.2f} GB")
        print(f"  Reserved:  {memory_reserved:.2f} GB") 
        # print(f"  Cached:    {memory_cached:.2f} GB")


def load_model_checkpoint(model, tokenizer, optimizer, checkpoint_path: str) -> int:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load into
        tokenizer: Tokenizer to load into
        optimizer: Optimizer to load into
        checkpoint_path: Path to checkpoint
        
    Returns:
        epoch: Loaded epoch number
    """
    import os
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load optimizer state
    optimizer_path = f"{checkpoint_path}/optimizer.pt"
    if os.path.exists(optimizer_path):
        checkpoint = torch.load(optimizer_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    else:
        epoch = 0
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    return epoch 