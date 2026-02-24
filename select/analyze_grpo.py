#!/usr/bin/env python3
"""
Main gradient analysis script for GRPO (Group Relative Policy Optimization)
Modified to analyze gradients between validation and training groups instead of training.
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from grpo_grad_analyze import GRPOGradientAnalyzer
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
        print(expected_answers[i])
        reward = reward_function(prompt, response, expected_answers[i])
        rewards.append(reward)
    return rewards


def load_dataset_responses(responses_dir: str, dataset_name: str, max_samples: int = None):
    """Load responses for a specific dataset (train or val)."""
    responses_file = os.path.join(responses_dir, f'responses.json')
    if not os.path.exists(responses_file):
        raise FileNotFoundError(f"Responses file not found: {responses_file}")
    
    print(f"Loading {dataset_name} responses from: {responses_file}")
    with open(responses_file, 'r') as f:
        responses_data = json.load(f)
    
    prompts = []
    responses = []
    group_indices = []
    expected_answers = []
    
    for item in responses_data[:max_samples]:
        prompts.append(item['prompt'])
        responses.append(item['response'])
        group_indices.append(item['group_id'])
        expected_answers.append(item['expected_answer'])
    
    print(f"Loaded {len(responses)} {dataset_name} responses from {len(set(group_indices))} groups")
    return prompts, responses, group_indices, expected_answers


def save_similarity_results(similarities: dict, output_path: str):
    """Save cosine similarity results to a JSON file."""
    results = {
        'cosine_similarities': similarities,
        'summary': {
            'num_groups': len(similarities),
            'mean_similarity': np.mean(list(similarities.values())),
            'std_similarity': np.std(list(similarities.values())),
            'max_similarity': max(similarities.values()),
            'min_similarity': min(similarities.values())
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSimilarity results saved to: {output_path}")
    print(f"Summary statistics:")
    print(f"  Number of groups: {results['summary']['num_groups']}")
    print(f"  Mean similarity: {results['summary']['mean_similarity']:.4f}")
    print(f"  Std similarity: {results['summary']['std_similarity']:.4f}")
    print(f"  Range: [{results['summary']['min_similarity']:.4f}, {results['summary']['max_similarity']:.4f}]")


def main():
    parser = argparse.ArgumentParser(description="GRPO Gradient Analysis Script")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the model to analyze")
    parser.add_argument("--train_responses_dir", type=str, required=True,
                       help="Directory containing pre-generated train responses")
    parser.add_argument("--val_responses_dir", type=str, required=True,
                       help="Directory containing pre-generated validation responses")
    parser.add_argument("--output_path", type=str, default="~/data_selection/data/analysis/gradient_similarities.json",
                       help="Output file for similarity results")
    parser.add_argument("--kl_loss_coef", type=float, default=0.001,
                       help="KL loss coefficient")
    parser.add_argument("--clip_ratio", type=float, default=0.2,
                       help="PPO clip ratio")
    parser.add_argument("--mini_batch_size", type=int, default=8,
                       help="Mini batch size for gradient computation")
    parser.add_argument("--max_length", type=int, default=8192,
                       help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to load")
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist!")
        sys.exit(1)
    
    # Check if response directories exist
    if not os.path.exists(args.train_responses_dir):
        print(f"Error: Train responses directory {args.train_responses_dir} does not exist!")
        sys.exit(1)
    
    if not os.path.exists(args.val_responses_dir):
        print(f"Error: Validation responses directory {args.val_responses_dir} does not exist!")
        sys.exit(1)
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Select reward function
    reward_fn = math_reward_function
    
    # Initialize gradient analyzer
    print(f"Loading model from: {args.model_path}")
    analyzer = GRPOGradientAnalyzer(
        model_path=args.model_path,
        device=device,
        kl_loss_coef=args.kl_loss_coef,
        clip_ratio=args.clip_ratio,
        norm_adv_by_std=False,
        max_length=args.max_length,
        mini_batch_size=args.mini_batch_size,
    )
    
    # Load training responses
    print(f"\nLoading training responses from: {args.train_responses_dir}")
    train_prompts, train_responses, train_group_indices, train_expected_answers = load_dataset_responses(
        args.train_responses_dir, "train", args.max_samples
    )
    
    # Load validation responses
    print(f"\nLoading validation responses from: {args.val_responses_dir}")
    val_prompts, val_responses, val_group_indices, val_expected_answers = load_dataset_responses(
        args.val_responses_dir, "val", args.max_samples
    )
    
    # Compute rewards for training data
    print("\nComputing rewards for training data...")
    train_rewards = compute_rewards(
        train_prompts, train_responses, reward_fn, 
        train_expected_answers
    )
    print(f"Train: {len(train_responses)} responses from {len(set(train_group_indices))} groups")
    print(f"Train reward - Mean: {np.mean(train_rewards):.3f}, Range: [{min(train_rewards):.3f}, {max(train_rewards):.3f}]")
    
    # Compute rewards for validation data  
    print("\nComputing rewards for validation data...")
    val_rewards = compute_rewards(
        val_prompts, val_responses, reward_fn,
        val_expected_answers
    )
    print(f"Val: {len(val_responses)} responses from {len(set(val_group_indices))} groups")
    print(f"Val reward - Mean: {np.mean(val_rewards):.3f}, Range: [{min(val_rewards):.3f}, {max(val_rewards):.3f}]")
    
    # Prepare training data
    print("\nPreparing training data...")
    train_data = analyzer.prepare_data(train_prompts, train_responses, train_rewards, train_group_indices)
    print(f"Prepared {len(train_data)} training samples")
    
    # Prepare validation data
    print("\nPreparing validation data...")
    val_data = analyzer.prepare_data(val_prompts, val_responses, val_rewards, val_group_indices)
    print(f"Prepared {len(val_data)} validation samples")
    
    # Perform gradient analysis
    print(f"\n{'='*50}")
    print("STARTING GRADIENT ANALYSIS")
    print(f"{'='*50}")
    
    similarities = analyzer.analyze_gradients(train_data, val_data)
    
    # Save results
    save_similarity_results(similarities, args.output_path)
    
    # Print top and bottom similarities
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 most similar training groups to validation:")
    for i, (group_id, similarity) in enumerate(sorted_similarities[:10]):
        print(f"  {i+1}. Group {group_id}: {similarity:.4f}")
    
    print(f"\nBottom 10 least similar training groups to validation:")
    for i, (group_id, similarity) in enumerate(sorted_similarities[-10:]):
        print(f"  {i+1}. Group {group_id}: {similarity:.4f}")
    
    print(f"\nGradient analysis completed! Results saved to: {args.output_path}")


if __name__ == "__main__":
    main() 