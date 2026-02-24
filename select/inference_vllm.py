#!/usr/bin/env python3
"""
Separate inference script using vLLM for response generation.
This script generates responses for given prompts and saves them to disk,
allowing training to happen separately without memory conflicts.
"""

import argparse
import json
import os
import torch
import re
import sys
from typing import List, Dict
from vllm import LLM, SamplingParams
import time
from datetime import datetime

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import compare_answers, extract_answer


def load_training_data(data_path: str, max_samples: int = 1000) -> List[Dict]:
    """Load training data from JSONL file with the specific format used in GRPO training."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = []
    cnt = 0
    with open(data_path, 'r') as f:
        for line in f:
            if cnt >= max_samples:
                break
            cnt += 1
            line = line.strip()
            if line:  # Skip empty lines
                entry = json.loads(line)
                data.append({
                    "problem": entry['prompt'][0]['content'],
                    "answer": entry['reward_model']['ground_truth'],
                    "original_entry": entry  # Keep original for reference
                })
    
    return data


def load_prompts_from_file(prompts_file: str, max_samples: int = None) -> List[Dict]:
    """Load prompts from JSON or JSONL file."""
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    
    data = []
    
    # Check if it's a JSONL file (based on extension or content)
    is_jsonl = prompts_file.endswith('.jsonl') or prompts_file.endswith('.jsonlines')
    
    if not is_jsonl:
        # Try to detect JSONL format by reading first line
        with open(prompts_file, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                try:
                    json.loads(first_line)
                    # If we can parse the first line as JSON, assume it's JSONL
                    is_jsonl = True
                except:
                    pass
    
    if is_jsonl:
        # Use the specialized loading function for JSONL training data format
        print(f"Loading JSONL format from: {prompts_file}")
        try:
            # Try the specific GRPO training format first
            data = load_training_data(prompts_file, max_samples or 10000)
            print(f"Successfully loaded using GRPO training format")
        except (KeyError, TypeError) as e:
            print(f"GRPO format failed ({e}), trying generic JSONL format...")
            # Fallback to generic JSONL format
            cnt = 0
            with open(prompts_file, 'r') as f:
                for line in f:
                    if max_samples and cnt >= max_samples:
                        break
                    cnt += 1
                    line = line.strip()
                    if line:  # Skip empty lines
                        entry = json.loads(line)
                        
                        # Generic fallback for other JSONL formats
                        problem = entry.get('problem', entry.get('prompt', str(entry)))
                        answer = entry.get('answer', entry.get('ground_truth', ''))
                        
                        data.append({
                            "problem": problem,
                            "answer": answer,
                            "original_entry": entry
                        })
    else:
        # Load regular JSON format
        print(f"Loading JSON format from: {prompts_file}")
        with open(prompts_file, 'r') as f:
            json_data = json.load(f)
            
        # Handle both list and single object
        if isinstance(json_data, list):
            data = json_data[:max_samples] if max_samples else json_data
        else:
            data = [json_data]
    
    print(f"Loaded {len(data)} entries")
    return data


def create_prompts_from_data(data: List[Dict], n_samples_per_prompt: int = 5) -> List[Dict]:
    """Create prompt data structure for inference."""
    prompts_data = []
    
    for group_idx, item in enumerate(data):
        # Handle different data formats
        if 'problem' in item:
            # Standard format with problem/answer
            prompt = item['problem']
            expected_answer = item.get('answer', '')
        elif 'prompt' in item:
            # Direct prompt format
            prompt = item['prompt']
            expected_answer = item.get('answer', '')
        else:
            # Generic format
            prompt = str(item)
            expected_answer = ''
        
        # Create multiple copies for group sampling
        for sample_idx in range(n_samples_per_prompt):
            prompts_data.append({
                'group_id': group_idx,
                'sample_id': sample_idx,
                'prompt': prompt,
                'expected_answer': expected_answer,
                'original_data': item
            })
    
    return prompts_data


def generate_responses_vllm(
    prompts_data: List[Dict],
    model_path: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    # data_parallel_size: int = 1,
    batch_size: int = 64,
    enable_problem_batching: bool = True
) -> List[Dict]:
    """Generate responses using vLLM with optional problem batching."""
    
    print(f"Initializing vLLM with model: {model_path}")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"Pipeline parallel size: {pipeline_parallel_size}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Problem batching: {'Enabled' if enable_problem_batching else 'Disabled'}")
    
    # Initialize vLLM
    llm = LLM(
        # pipeline_parallel_size=pipeline_parallel_size,
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        gpu_memory_utilization=0.9,
        max_model_len=max_tokens,  # Allow room for prompt + response
        max_num_seqs=512,
    )
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        skip_special_tokens=False,
        top_p=0.95
    )
    
    print(f"Generating responses for {len(prompts_data)} prompts...")
    start_time = time.time()
    
    if enable_problem_batching:
        # New approach: Batch different problems together
        print("Using problem batching mode for better efficiency...")
        
        # Extract all prompts for batch generation
        all_prompts = [item['prompt'] for item in prompts_data]
        
        # Generate all responses in batches
        all_responses = []
        for i in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[i:i + batch_size]
            batch_end = min(i + batch_size, len(all_prompts))
            
            print(f"Processing batch {i//batch_size + 1}/{(len(all_prompts) + batch_size - 1)//batch_size} "
                  f"(problems {i} to {batch_end-1})")
            
            outputs = llm.generate(batch_prompts, sampling_params)
            
            batch_responses = []
            for output in outputs:
                batch_responses.append(output.outputs[0].text)
            
            all_responses.extend(batch_responses)
        
        # Combine prompts data with responses
        results = []
        for i, prompt_data in enumerate(prompts_data):
            result = prompt_data.copy()
            result['response'] = all_responses[i]
            result['generation_time'] = 0  # Will be calculated below
            results.append(result)
            
    else:
        # Original approach: Process each problem separately
        print("Using sequential processing mode...")
        
        # Group by problem (group_id) to process each problem's samples together
        problems_dict = {}
        for item in prompts_data:
            group_id = item['group_id']
            if group_id not in problems_dict:
                problems_dict[group_id] = []
            problems_dict[group_id].append(item)
        
        results = []
        for group_id, group_items in problems_dict.items():
            print(f"Processing problem {group_id + 1}/{len(problems_dict)} "
                  f"({len(group_items)} samples)")
            
            # Extract prompts for this problem
            group_prompts = [item['prompt'] for item in group_items]
            
            # Generate responses for this problem's samples
            outputs = llm.generate(group_prompts, sampling_params)
            
            # Combine with original data
            for i, (output, original_item) in enumerate(zip(outputs, group_items)):
                result = original_item.copy()
                result['response'] = output.outputs[0].text
                result['generation_time'] = 0  # Will be calculated below
                results.append(result)
    
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    print(f"Average time per prompt: {generation_time/len(prompts_data):.3f} seconds")
    
    # Update generation time for all results
    avg_time = generation_time / len(prompts_data)
    for result in results:
        result['generation_time'] = avg_time
    
    return results


def extract_answer_from_response(response: str) -> str:
    """Extract the final answer from a response string using utility function."""
    return extract_answer(response)


def calculate_accuracy(responses_data: List[Dict]) -> Dict:
    """Calculate accuracy metrics for the generated responses."""
    total_responses = len(responses_data)
    correct_responses = 0
    group_accuracy = {}
    
    for item in responses_data:
        expected = str(item['expected_answer']).strip()
        response = item['response']
        
        # Extract answer from response using utility function
        extracted_answer = extract_answer_from_response(response)
        
        # Check if answer is correct using utility function
        is_correct = compare_answers(extracted_answer, expected)
        
        # Store results
        item['extracted_answer'] = extracted_answer
        item['is_correct'] = is_correct
        
        if is_correct:
            correct_responses += 1
        
        # Track per-group accuracy
        group_id = item['group_id']
        if group_id not in group_accuracy:
            group_accuracy[group_id] = {'correct': 0, 'total': 0}
        group_accuracy[group_id]['total'] += 1
        if is_correct:
            group_accuracy[group_id]['correct'] += 1
    
    # Calculate overall accuracy
    overall_accuracy = correct_responses / total_responses if total_responses > 0 else 0
    
    # Calculate per-group accuracy
    group_accuracy_rates = {}
    for group_id, stats in group_accuracy.items():
        group_accuracy_rates[group_id] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    
    # Calculate per-problem accuracy (best response per problem)
    problem_accuracy = {}
    for group_id in group_accuracy_rates:
        group_items = [item for item in responses_data if item['group_id'] == group_id]
        # Check if any response in this group is correct
        any_correct = any(item['is_correct'] for item in group_items)
        problem_accuracy[group_id] = 1.0 if any_correct else 0.0
    
    best_of_n_accuracy = sum(problem_accuracy.values()) / len(problem_accuracy) if problem_accuracy else 0
    
    return {
        'overall_accuracy': overall_accuracy,
        'correct_responses': correct_responses,
        'total_responses': total_responses,
        'group_accuracy_rates': group_accuracy_rates,
        'best_of_n_accuracy': best_of_n_accuracy,  # At least one correct response per problem
        'num_problems': len(group_accuracy_rates)
    }


def save_results(results: List[Dict], output_dir: str, metadata: Dict = None):
    """Save results to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main results
    results_file = os.path.join(output_dir, 'responses.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save metadata
    if metadata is None:
        metadata = {}
    
    metadata.update({
        'timestamp': datetime.now().isoformat(),
        'num_responses': len(results),
        'output_dir': output_dir
    })
    
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Results saved to: {output_dir}")
    print(f"  - Responses: {results_file}")
    print(f"  - Metadata: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="vLLM Inference Script for GRPO")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model for inference")
    parser.add_argument("--prompts_file", type=str, required=True,
                       help="Path to JSON/JSONL file containing prompts/problems")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save generated responses")
    parser.add_argument("--n_samples", type=int, default=5,
                       help="Number of samples per prompt")
    parser.add_argument("--max_problems", type=int, default=None,
                       help="Maximum number of problems to process (useful for testing)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="Maximum tokens to generate")
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                       help="Number of GPUs for tensor parallelism")
    # parser.add_argument("--pipeline_parallel_size", type=int, default=2,
    #                    help="Number of GPUs for pipeline parallelism")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1,
                       help="Number of GPUs for pipeline parallelism")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for generation")
    parser.add_argument("--enable_problem_batching", action="store_true",
                       help="Enable batching of different problems together for better efficiency")
    
    args = parser.parse_args()
    
    # Load prompts
    print(f"Loading prompts from: {args.prompts_file}")
    data = load_prompts_from_file(args.prompts_file, args.max_problems)
    print(f"Loaded {len(data)} items")
    
    # Create prompts data structure
    prompts_data = create_prompts_from_data(data, args.n_samples)
    print(f"Created {len(prompts_data)} prompts ({args.n_samples} samples per item)")
    
    # Generate responses
    results = generate_responses_vllm(
        prompts_data=prompts_data,
        model_path=args.model_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        batch_size=args.batch_size,
        enable_problem_batching=args.enable_problem_batching
    )
    
    # Calculate accuracy
    accuracy_metrics = calculate_accuracy(results)
    print("\nAccuracy Metrics:")
    print(f"Overall Accuracy: {accuracy_metrics['overall_accuracy']:.2%}")
    print(f"Correct Responses: {accuracy_metrics['correct_responses']}/{accuracy_metrics['total_responses']}")
    print(f"Best of N Accuracy (at least one correct per problem): {accuracy_metrics['best_of_n_accuracy']:.2%}")
    print(f"Number of Problems: {accuracy_metrics['num_problems']}")
    
    # Show sample accuracy per group (first 10 groups)
    sample_groups = dict(list(accuracy_metrics['group_accuracy_rates'].items())[:10])
    print(f"Sample Group Accuracy: {sample_groups}")
    
    # Save results
    metadata = {
        'model_path': args.model_path,
        'prompts_file': args.prompts_file,
        'n_samples': args.n_samples,
        'max_problems': args.max_problems,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'tensor_parallel_size': args.tensor_parallel_size,
        'pipeline_parallel_size': args.pipeline_parallel_size,
        # 'data_parallel_size': args.data_parallel_size,
        'batch_size': args.batch_size,
        'enable_problem_batching': args.enable_problem_batching,
        'accuracy_metrics': accuracy_metrics
    }
    
    save_results(results, args.output_dir, metadata)
    print("Inference completed successfully!")


if __name__ == "__main__":
    main() 