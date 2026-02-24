#!/usr/bin/env python3
"""
Main gradient analysis script for GRPO (Group Relative Policy Optimization) - Parallel Version
Modified to analyze gradients between validation and training groups using multiple GPUs.
Uses Accelerate + FSDP for distributed processing.
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from grpo_grad_analyze_parallel import GRPOGradientAnalyzerParallel
from grpo_loss import compute_grpo_advantages
from accelerate import Accelerator
from typing import Any, Dict, List, Optional, Tuple


def load_already_processed_groups(output_path: str, accelerator: Accelerator) -> set:
    """Load already processed group IDs from the JSONL output file."""
    processed_groups = set()
    
    # if not accelerator.is_main_process:
    #     return processed_groups
    
    if os.path.exists(output_path):
        if accelerator.is_main_process:
            print(f"Found existing results file: {output_path}")
            print("Loading already processed groups...")
        
        try:
            with open(output_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        result = json.loads(line.strip())
                        if 'group_id' in result:
                            processed_groups.add(result['group_id'])
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed line {line_num}: {e}")
                        
            print(f"Found {len(processed_groups)} already processed groups")
            print(f"Processed groups: {sorted(list(processed_groups))}")
        except Exception as e:
            print(f"Error reading existing results file: {e}")
            print("Starting fresh analysis...")
            processed_groups = set()
    else:
        print(f"No existing results file found at {output_path}")
        print("Starting fresh analysis...")
    
    return processed_groups


def save_final_summary(similarities: dict, output_path: str, accelerator: Accelerator):
    """Save a final summary of all results."""
    if not accelerator.is_main_process or not similarities:
        return
    
    summary_path = output_path.replace('.jsonl', '_summary.json')
    
    results = {
        'summary': {
            'num_groups': len(similarities),
            'mean_similarity': np.mean(list(similarities.values())),
            'std_similarity': np.std(list(similarities.values())),
            'max_similarity': max(similarities.values()),
            'min_similarity': min(similarities.values())
        },
        'all_similarities': similarities
    }
    
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFinal summary saved to: {summary_path}")
    print(f"Summary statistics:")
    print(f"  Number of groups: {results['summary']['num_groups']}")
    print(f"  Mean similarity: {results['summary']['mean_similarity']:.4f}")
    print(f"  Std similarity: {results['summary']['std_similarity']:.4f}")
    print(f"  Range: [{results['summary']['min_similarity']:.4f}, {results['summary']['max_similarity']:.4f}]")


def load_ground_truth_mapping(problem_set_path: str, accelerator: Accelerator):
    """Load ground truth mapping from problem set file."""
    if not os.path.exists(problem_set_path):
        raise FileNotFoundError(f"Problem set file not found: {problem_set_path}")
    
    if accelerator.is_main_process:
        print(f"Loading ground truth from problem set: {problem_set_path}")
    
    ground_truth_map = {}
    with open(problem_set_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            index = entry['extra_info']['index']
            ground_truth = entry['reward_model']['ground_truth']
            ground_truth_map[index] = ground_truth
    
    if accelerator.is_main_process:
        print(f"Loaded ground truth for {len(ground_truth_map)} problems")
    
    return ground_truth_map


def load_dataset_responses(
    responses_dir: str,
    dataset_name: str,
    accelerator: Accelerator,
    ground_truth_map: Optional[Dict[int, Any]] = None,
    max_num_samples: Optional[int] = None,
) -> Tuple[List[Any], List[str], List[int], List[str], List[bool]]:
    """Load responses for a specific dataset (train or val)."""
    responses_file = os.path.join(responses_dir, f'responses_sorted.json')
    if not os.path.exists(responses_file):
        raise FileNotFoundError(f"Responses file not found: {responses_file}")
    
    if accelerator.is_main_process:
        print(f"Loading {dataset_name} responses from: {responses_file}")
    
    with open(responses_file, 'r') as f:
        responses_data = json.load(f)
    
    prompts = []
    responses = []
    group_indices = []
    expected_answers = []
    passed_flags: List[bool] = []
    
    skipped_count = 0
    iterable = responses_data if max_num_samples is None else responses_data[:max_num_samples]
    for item in iterable:
        group_id = item['group_id']
        
        # Skip items with missing ground truth (e.g., problems removed by flip_label.py)
        if ground_truth_map and group_id not in ground_truth_map:
            skipped_count += 1
            continue
        
        prompts.append(item['prompt'])
        responses.append(item['response'])
        group_indices.append(group_id)
        # Get ground truth from problem set instead of stale response file
        expected_answers.append(ground_truth_map[group_id] if ground_truth_map else item['expected_answer'])
        passed_value = item.get('passed')
        if passed_value is None:
            passed_value = item.get('is_correct', False)
        passed_flags.append(bool(passed_value))
    
    if accelerator.is_main_process:
        print(f"Loaded {len(responses)} {dataset_name} responses from {len(set(group_indices))} groups")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} responses with missing expected_answer (likely removed by flip_label.py)")
    
    return prompts, responses, group_indices, expected_answers, passed_flags


def main():
    parser = argparse.ArgumentParser(description="GRPO Gradient Analysis Script - Parallel Version")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the model to analyze")
    parser.add_argument("--train_responses_dir", type=str, required=True,
                       help="Directory containing pre-generated train responses")
    parser.add_argument("--val_responses_dir", type=str, required=True,
                       help="Directory containing pre-generated validation responses")
    parser.add_argument("--output_path", type=str, default="~/data_selection/data/analysis/gradient_similarities_parallel.jsonl",
                       help="Output JSONL file for incremental similarity results")
    parser.add_argument("--kl_loss_coef", type=float, default=0.001,
                       help="KL loss coefficient")
    parser.add_argument("--clip_ratio", type=float, default=0.2,
                       help="PPO clip ratio")
    parser.add_argument("--mini_batch_size", type=int, default=8,
                       help="Mini batch size for gradient computation")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--mixed_precision", type=str, default="bf16", 
                       help="Mixed precision mode")
    parser.add_argument("--fsdp_auto_wrap_policy", type=str, 
                       default="transformer_auto_wrap_policy",
                       help="FSDP auto wrap policy")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=True,
                       help="Enable gradient checkpointing for memory efficiency")
    parser.add_argument("--cpu_offload", action="store_true",
                       help="Offload parameters to CPU when not in use")
    parser.add_argument("--distribute_val_data", action="store_true",
                       help="Distribute validation data across processes (vs main process only)")
    parser.add_argument("--max_num_samples", type=int, default=None,
                       help="Maximum number of samples to load from each dataset")
    parser.add_argument("--problem_set_path", type=str, required=True,
                       help="Path to the problem set file (train.jsonl) containing ground truth")
    parser.add_argument("--norm_before_accumlation", type=bool, default=False,
                       help="Normalize the gradients before accumulation")
    parser.add_argument("--use_optimizer", action="store_true", default=False,
                       help="Scale validation gradients with optimizer exp_avg_sq state")
    parser.add_argument("--optimizer_state_path", type=str, default=None,
                       help="Path to converted optimizer state (required when --use_optimizer)")
    parser.add_argument("--mode", default='sim', type=str)
    args = parser.parse_args()
    # print('fa', args.norm_before_accumlation)
    # exit()

    if args.use_optimizer and not args.optimizer_state_path:
        parser.error("--optimizer_state_path must be provided when --use_optimizer is set")

    # Initialize a simple accelerator for process coordination (not for model sharding yet)
    temp_accelerator = Accelerator()
    
    # Check if model exists (only on main process)
    if temp_accelerator.is_main_process:
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
        
        print(f"Running on {temp_accelerator.num_processes} GPUs")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Wait for all processes to sync
    temp_accelerator.wait_for_everyone()
    
    # Load already processed groups (only on main process)
    processed_groups = load_already_processed_groups(args.output_path, temp_accelerator)
    
    # Load ground truth mapping from problem set file
    ground_truth_map = load_ground_truth_mapping(args.problem_set_path, temp_accelerator)
    
    # Initialize parallel gradient analyzer (this will create its own accelerator)
    if temp_accelerator.is_main_process:
        print(f"Initializing parallel model from: {args.model_path}")
    
    analyzer = GRPOGradientAnalyzerParallel(
        model_path=args.model_path,
        kl_loss_coef=args.kl_loss_coef,
        clip_ratio=args.clip_ratio,
        norm_adv_by_std=False,
        max_length=args.max_length,
        mini_batch_size=args.mini_batch_size,
        mixed_precision=args.mixed_precision,
        fsdp_auto_wrap_policy=args.fsdp_auto_wrap_policy,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        cpu_offload=args.cpu_offload,
        distribute_val_data=args.distribute_val_data,
        use_optimizer=args.use_optimizer,
        optimizer_state_path=args.optimizer_state_path,
        mode=args.mode,
    )
    
    # Use the analyzer's accelerator for subsequent operations
    accelerator = analyzer.accelerator
    
    # Load training responses
    if accelerator.is_main_process:
        print(f"\nLoading training responses from: {args.train_responses_dir}")
    
    (
        train_prompts,
        train_responses,
        train_group_indices,
        train_expected_answers,
        train_passed,
    ) = load_dataset_responses(
        args.train_responses_dir, "train", accelerator, ground_truth_map, max_num_samples=args.max_num_samples
    )
    # Preserve full copies for advantage computation after sharding
    train_prompts_full = list(train_prompts)
    train_responses_full = list(train_responses)
    train_group_indices_full = list(train_group_indices)
    train_expected_answers_full = list(train_expected_answers)
    train_passed_full = list(train_passed)
    
    # Load validation responses
    if accelerator.is_main_process:
        print(f"\nLoading validation responses from: {args.val_responses_dir}")
    
    (
        val_prompts,
        val_responses,
        val_group_indices,
        val_expected_answers,
        val_passed,
    ) = load_dataset_responses(
        args.val_responses_dir, "val", accelerator, max_num_samples=args.max_num_samples
    )
    val_prompts_full = list(val_prompts)
    val_responses_full = list(val_responses)
    val_group_indices_full = list(val_group_indices)
    val_expected_answers_full = list(val_expected_answers)
    val_passed_full = list(val_passed)

    # Filter out already processed training groups
    if processed_groups:
        original_train_count = len(train_prompts)
        
        # Create masks for samples that belong to unprocessed groups
        unprocessed_mask = [group_id not in processed_groups for group_id in train_group_indices]
        
        # Filter training data
        train_prompts = [p for i, p in enumerate(train_prompts) if unprocessed_mask[i]]
        train_responses = [r for i, r in enumerate(train_responses) if unprocessed_mask[i]]
        train_group_indices = [g for i, g in enumerate(train_group_indices) if unprocessed_mask[i]]
        train_expected_answers = [a for i, a in enumerate(train_expected_answers) if unprocessed_mask[i]]
        train_passed = [v for i, v in enumerate(train_passed) if unprocessed_mask[i]]
        
        remaining_groups = set(train_group_indices)
        
        print(f"\nFiltered training data:")
        print(f"  Original: {original_train_count} samples")
        print(f"  Remaining: {len(train_prompts)} samples from {len(remaining_groups)} unprocessed groups")
        print(f"  Skipping {len(processed_groups)} already processed groups")
        
        if not train_prompts:
            print("All training groups have been processed! Generating final summary...")
            # Load all results for summary
            all_similarities = {}
            with open(args.output_path, 'r') as f:
                for line in f:
                    result = json.loads(line.strip())
                    if 'group_id' in result and 'similarity' in result:
                        all_similarities[result['group_id']] = result['similarity']
            
            save_final_summary(all_similarities, args.output_path, accelerator)
            print("Analysis complete!")
            return
    
    # Convert pass/fail flags into rewards for advantage computation
    if accelerator.is_main_process:
        print("\nComputing rewards from pass/fail signals...")
    
    train_rewards_full = [1.0 if flag else 0.0 for flag in train_passed_full]
    val_rewards_full = [1.0 if flag else 0.0 for flag in val_passed_full]
    
    if accelerator.is_main_process:
        print(f"Train: {len(train_responses_full)} responses from {len(set(train_group_indices_full))} groups")
        print(f"Train pass rate - Mean: {np.mean(train_rewards_full):.3f}")
        print(f"Val: {len(val_responses_full)} responses from {len(set(val_group_indices_full))} groups")
        print(f"Val pass rate - Mean: {np.mean(val_rewards_full):.3f}")
    
    # Load pre-sharded prepared datasets instead of in-process tokenization
    if accelerator.is_main_process:
        print("\nLoading pre-tokenized and pre-sharded datasets (CPU-only pipeline)...")

    def load_shard(responses_dir: str, world_size: int, rank: int) -> list:
        npz_path = os.path.join(responses_dir, 'data', f"data_{rank}.npz")
        pt_path = os.path.join(responses_dir, 'data', f"data_{rank}.pt")
        if os.path.exists(npz_path):
            import numpy as np
            arrs = np.load(npz_path, allow_pickle=True)
            ids_obj = arrs['input_ids']          # object array of 1D np.int32
            att_obj = arrs['attention_mask']     # object array of 1D np.uint8
            rsp_obj = arrs['response_mask']      # object array of 1D np.uint8
            gid_np = arrs['group_id']            # 1D np.int32
            gidx_np = arrs['global_index']       # 1D np.int64

            # Bulk conversions for numeric arrays
            gid_list = gid_np.tolist()
            gidx_list = gidx_np.tolist()

            n = len(gid_list)
            shard = [
                {
                    'input_ids': torch.from_numpy(ids_obj[i]).to(dtype=torch.long),
                    'attention_mask': torch.from_numpy(att_obj[i]).to(dtype=torch.long),
                    'response_mask': torch.from_numpy(rsp_obj[i]).to(dtype=torch.long),
                    'group_id': int(gid_list[i]),
                    'global_index': int(gidx_list[i]),
                }
                for i in range(n)
            ]
            return shard
        if os.path.exists(pt_path):
            return torch.load(pt_path, map_location='cpu')
        raise FileNotFoundError(f"Prepared shard not found: {npz_path} or {pt_path}")

    rank = accelerator.process_index
    world_size = accelerator.num_processes

    train_data = load_shard(args.train_responses_dir, world_size, rank)[:args.max_num_samples // world_size]
    val_data = load_shard(args.val_responses_dir, world_size, rank)[:args.max_num_samples // world_size]

    # Compute advantages on full datasets and attach by global_index
    train_advantages_full = compute_grpo_advantages(
        torch.tensor(train_rewards_full, dtype=torch.float32),
        np.array(train_group_indices_full),
        False, 1e-6
    ).detach()
    val_advantages_full = compute_grpo_advantages(
        torch.tensor(val_rewards_full, dtype=torch.float32),
        np.array(val_group_indices_full),
        False, 1e-6
    ).detach()
    # print('Len', len(train_data))
    # print('fa' + str([(entry['global_index'], entry['group_id']) for entry in train_data]))

    for entry in train_data:
        gi = entry.get('global_index', None)
        if gi is None:
            raise RuntimeError("Shard entry missing global_index for advantage mapping (train)")
        entry['advantages'] = train_advantages_full[gi]

    for entry in val_data:
        gi = entry.get('global_index', None)
        if gi is None:
            raise RuntimeError("Shard entry missing global_index for advantage mapping (val)")
        entry['advantages'] = val_advantages_full[gi]

    if accelerator.is_main_process:
        print(f"Loaded train shard with {len(train_data)} entries; val shard with {len(val_data)} entries")
    
    # Print memory stats before analysis
    analyzer.print_memory_stats()
    
    # Perform gradient analysis with incremental saving
    if accelerator.is_main_process:
        print(f"\n{'='*50}")
        print("STARTING PARALLEL GRADIENT ANALYSIS")
        print(f"Using {accelerator.num_processes} GPUs")
        print(f"Results will be saved incrementally to: {args.output_path}")
        print(f"{'='*50}")
    
    similarities = analyzer.analyze_gradients(train_data, val_data, args.output_path, processed_groups, args.norm_before_accumlation)
    
    # Print memory stats after analysis
    if accelerator.is_main_process:
        print("\nFinal memory usage:")
    analyzer.print_memory_stats()
    
    # Generate final summary
    if accelerator.is_main_process:
        # Load all results (including any previously processed ones)
        all_similarities = {}
        if os.path.exists(args.output_path):
            with open(args.output_path, 'r') as f:
                for line in f:
                    result = json.loads(line.strip())
                    if 'group_id' in result and 'similarity' in result:
                        all_similarities[result['group_id']] = result['similarity']
        
        save_final_summary(all_similarities, args.output_path, accelerator)
        
        # Print top and bottom similarities
        if all_similarities:
            sorted_similarities = sorted(all_similarities.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\nTop 10 most similar training groups to validation:")
            for i, (group_id, similarity) in enumerate(sorted_similarities[:10]):
                print(f"  {i+1}. Group {group_id}: {similarity:.4f}")
            
            print(f"\nBottom 10 least similar training groups to validation:")
            for i, (group_id, similarity) in enumerate(sorted_similarities[-10:]):
                print(f"  {i+1}. Group {group_id}: {similarity:.4f}")
        
        print(f"\nParallel gradient analysis completed! Results saved to: {args.output_path}")
    
    # Clean up distributed resources
    try:
        analyzer.cleanup()
    except:
        pass  # Ignore cleanup errors


if __name__ == "__main__":
    main() 