#!/usr/bin/env python3
"""
Main training script for GRPO (Group Relative Policy Optimization)
Modified to work with pre-generated responses from separate inference.
"""

import argparse
import json
import os
import sys
import torch
from grpo_trainer import GRPOTrainer
from grpo_utils import (
    simple_reward_function, 
    math_reward_function, 
    create_math_prompt_simple,
    parse_math_data_with_answers
)


def compute_rewards(prompts, responses, reward_function, expected_answers=None):
    """Compute rewards for prompt-response pairs."""
    rewards = []
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        if reward_function == math_reward_function and expected_answers:
            reward = reward_function(prompt, response, expected_answers[i])
        else:
            reward = reward_function(prompt, response)
        rewards.append(reward)
    return rewards


def main():
    parser = argparse.ArgumentParser(description="GRPO Training Script")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the model to train")
    parser.add_argument("--responses_dir", type=str, required=True,
                       help="Directory containing pre-generated responses")
    parser.add_argument("--output_path", type=str, default="~/data_selection/data/chkpt/grpo",
                       help="Output directory for checkpoints")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                       help="Learning rate")
    parser.add_argument("--kl_loss_coef", type=float, default=0.001,
                       help="KL loss coefficient")
    parser.add_argument("--clip_ratio", type=float, default=0.2,
                       help="PPO clip ratio")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--mini_batch_size", type=int, default=8,
                       help="Mini batch size")
    parser.add_argument("--ppo_epochs", type=int, default=4,
                       help="Number of PPO epochs per update")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--reward_function", type=str, default="simple",
                       choices=["simple", "math"], help="Reward function to use")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist!")
        sys.exit(1)
    
    # Check if responses directory exists
    if not os.path.exists(args.responses_dir):
        print(f"Error: Responses directory {args.responses_dir} does not exist!")
        sys.exit(1)
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Select reward function
    if args.reward_function == "math":
        reward_fn = math_reward_function
    else:
        reward_fn = simple_reward_function
    
    # Initialize trainer
    print(f"Loading model from: {args.model_path}")
    trainer = GRPOTrainer(
        model_path=args.model_path,
        device=device,
        learning_rate=args.learning_rate,
        kl_loss_coef=args.kl_loss_coef,
        clip_ratio=args.clip_ratio,
        norm_adv_by_std=True,
        max_length=args.max_length,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
    )
    
    # Load pre-generated responses
    print(f"Loading responses from: {args.responses_dir}")
    prompts, responses, group_indices = trainer.load_responses_from_disk(args.responses_dir)
    
    # Load expected answers if using math reward function
    expected_answers = None
    if args.reward_function == "math":
        # Load original data to get expected answers
        responses_file = os.path.join(args.responses_dir, 'responses.json')
        with open(responses_file, 'r') as f:
            responses_data = json.load(f)
        expected_answers = [item.get('expected_answer', '') for item in responses_data]
    
    # Compute rewards
    print("Computing rewards...")
    rewards = compute_rewards(prompts, responses, reward_fn, expected_answers)
    
    print(f"Generated {len(responses)} responses from {len(set(group_indices))} groups")
    print(f"Average reward: {sum(rewards)/len(rewards):.3f}")
    print(f"Reward range: [{min(rewards):.3f}, {max(rewards):.3f}]")
    
    # Prepare training data using rollout
    print("Preparing training data...")
    train_data = trainer.rollout(prompts, responses, rewards, group_indices)
    
    print(f"Prepared {len(train_data)} training samples")
    
    # Train model
    print(f"\nStarting training for {args.epochs} epochs...")
    trainer.train(train_data, num_epochs=args.epochs, save_path=args.output_path)
    
    print(f"\nTraining completed! Model saved to: {args.output_path}")


if __name__ == "__main__":
    main() 