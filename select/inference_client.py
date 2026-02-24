#!/usr/bin/env python3
"""
Client script for sending inference requests to a local vLLM server.
This script generates responses for given prompts by communicating with a vLLM server
running with pipeline parallelism (tp=4, pp=2).
"""

import argparse
import json
import os
import sys
import time
import asyncio
from typing import List, Dict
from datetime import datetime
from openai import AsyncOpenAI
import aiohttp

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


async def check_server_health(server_url: str) -> bool:
    """Check if the vLLM server is running and healthy."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{server_url}/health") as response:
                return response.status == 200
    except:
        return False


async def generate_responses_client(
    prompts_data: List[Dict],
    server_url: str = "http://localhost:8000",
    model_name: str = "deepseek-r1-distill",
    temperature: float = 0.7,
    max_tokens: int = 512,
    batch_size: int = 32,  # Smaller batch size for API calls
    max_concurrent: int = 10,  # Maximum concurrent requests
) -> List[Dict]:
    """Generate responses using OpenAI client to communicate with vLLM server."""
    
    print(f"Connecting to vLLM server at: {server_url}")
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Max concurrent requests: {max_concurrent}")
    
    # Check server health
    if not await check_server_health(server_url):
        raise ConnectionError(f"vLLM server is not accessible at {server_url}. Please start the server first.")
    
    # Initialize OpenAI client
    client = AsyncOpenAI(
        base_url=f"{server_url}/v1",
        api_key="dummy"  # vLLM doesn't require a real API key
    )
    
    print(f"Generating responses for {len(prompts_data)} prompts...")
    start_time = time.time()
    
    # Create semaphore for controlling concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate_single_response(prompt_data: Dict) -> Dict:
        """Generate a single response with concurrency control."""
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt_data['prompt']}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.95
                )
                
                result = prompt_data.copy()
                result['response'] = response.choices[0].message.content
                result['generation_time'] = 0  # Will be calculated later
                return result
                
            except Exception as e:
                print(f"Error generating response: {e}")
                result = prompt_data.copy()
                result['response'] = f"ERROR: {str(e)}"
                result['generation_time'] = 0
                return result
    
    # Process all prompts with controlled concurrency
    print("Starting batch generation with concurrency control...")
    tasks = [generate_single_response(prompt_data) for prompt_data in prompts_data]
    
    # Process in batches to avoid overwhelming the server
    results = []
    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i:i + batch_size]
        batch_end = min(i + batch_size, len(tasks))
        
        print(f"Processing batch {i//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size} "
              f"(requests {i} to {batch_end-1})")
        
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Handle any exceptions in the batch
        for result in batch_results:
            if isinstance(result, Exception):
                print(f"Batch error: {result}")
                # Create a dummy result for failed requests
                error_result = {
                    'group_id': -1,
                    'sample_id': -1,
                    'prompt': 'ERROR',
                    'expected_answer': '',
                    'response': f"ERROR: {str(result)}",
                    'generation_time': 0
                }
                results.append(error_result)
            else:
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
        
        # Skip error responses
        if response.startswith("ERROR:"):
            continue
        
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
        any_correct = any(item.get('is_correct', False) for item in group_items)
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


async def main_async():
    parser = argparse.ArgumentParser(description="vLLM Client Inference Script for GRPO")
    parser.add_argument("--server_url", type=str, default="http://localhost:8000",
                       help="URL of the vLLM server")
    parser.add_argument("--model_name", type=str, default="deepseek-r1-distill",
                       help="Model name as served by the vLLM server")
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
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for API requests")
    parser.add_argument("--max_concurrent", type=int, default=10,
                       help="Maximum concurrent requests")
    
    args = parser.parse_args()
    
    # Load prompts
    print(f"Loading prompts from: {args.prompts_file}")
    data = load_prompts_from_file(args.prompts_file, args.max_problems)
    print(f"Loaded {len(data)} items")
    
    # Create prompts data structure
    prompts_data = create_prompts_from_data(data, args.n_samples)
    print(f"Created {len(prompts_data)} prompts ({args.n_samples} samples per item)")
    
    # Generate responses
    results = await generate_responses_client(
        prompts_data=prompts_data,
        server_url=args.server_url,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent
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
        'server_url': args.server_url,
        'model_name': args.model_name,
        'prompts_file': args.prompts_file,
        'n_samples': args.n_samples,
        'max_problems': args.max_problems,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'batch_size': args.batch_size,
        'max_concurrent': args.max_concurrent,
        'accuracy_metrics': accuracy_metrics
    }
    
    save_results(results, args.output_dir, metadata)
    print("Inference completed successfully!")


def main():
    """Synchronous wrapper for the async main function."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main() 