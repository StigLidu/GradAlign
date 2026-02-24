#!/usr/bin/env python3
"""
Batched offline inference script using Ray Data and vLLM.
This script uses Ray Data's native vLLM integration for efficient
data-parallel batch inference with tensor and pipeline parallelism.

Ray Data provides:
* Automatic sharding and load-balancing
* Optimized configuration with continuous batching
* Compatible with tensor/pipeline parallel inference
* Scalable to large datasets
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import List, Dict, Any, Callable, Optional, cast, Tuple
from datetime import datetime

import ray
from packaging.version import Version
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import compare_answers, extract_answer
from verl.utils.reward_score.model_reward import compute_score  # type: ignore[import-not-found]

# Check Ray version
assert Version(ray.__version__) >= Version("2.44.1"), (
    "Ray version must be at least 2.44.1 for native vLLM integration"
)


def load_and_preprocess_data(prompts_file: str, min_problems: Optional[int] = None, max_problems: Optional[int] = None, n_samples: int = 5) -> List[Dict]:
    """Load and preprocess JSONL data for Ray Data processing."""
    
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
                # Treat None as 1 (start from first problem)
                min_idx = min_problems or 1
                if line and cnt >= min_idx:  # Skip empty lines and earlier indices
                    entry = json.loads(line)
                    data.append({
                        "msg": entry['prompt'],
                        "answer": entry['reward_model']['ground_truth'],
                        "group_id": entry['extra_info']['index'],
                        "original_entry": entry,  # Keep original for reference
                        "data_source": entry['data_source'],
                    })
        
        return data

    print(f"Loading JSONL format from: {prompts_file}")
    
    # Load the training data
        # Try the specific GRPO training format first
    data = load_training_data(prompts_file, max_problems or 10000)
    print(f"Successfully loaded using GRPO training format")
    
    print(f"Loaded {len(data)} base problems")
    
    # Create multiple samples per problem
    processed_data = []
    for item in data:
        messages = item['msg']
        expected_answer = item['answer']
        
        # Create multiple copies for group sampling
        for sample_idx in range(n_samples):
            processed_data.append({
                'group_id': item['group_id'],
                'sample_id': sample_idx,
                'msg': messages,
                'expected_answer': expected_answer,
                'original_data': item,
                'data_source': item['data_source'],
            })
    
    print(f"Created {len(processed_data)} total samples ({n_samples} samples per problem)")
    return processed_data


async def _judge_responses_async(
    entries: List[Tuple[int, Dict[str, Any]]],
    concurrency: int,
) -> None:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _evaluate(idx: int, payload: Dict[str, Any]) -> None:
        async with semaphore:
            try:
                score = await compute_score(
                    task=payload["data_source"],
                    solution_str=payload["response"],
                    ground_truth=payload["expected_answer"],
                    extra_info=payload.get("extra_info", {}),
                )
                payload["score"] = float(score)
            except Exception as exc:  # noqa: BLE001
                print(f"[Model Judge] Error for index {idx}: {exc}")
                payload["score"] = 0.0

    await asyncio.gather(*(_evaluate(idx, payload) for idx, payload in entries))


def run_model_judge(
    results: List[Dict[str, Any]],
    concurrency: int,
) -> int:
    judge_candidates: List[Tuple[int, Dict[str, Any]]] = []
    for idx, item in enumerate(results):
        # if item.get("passed", item.get("is_correct", False)):
        #     continue
        expected_answer = str(item.get("expected_answer", "")).strip()
        if not expected_answer:
            continue
        original_data = item.get("original_data") or {}
        extra_info: Dict[str, Any] = {}
        if isinstance(original_data, dict):
            original_entry = original_data.get("original_entry")
            if isinstance(original_entry, dict):
                extra_info = original_entry.get("extra_info", {})
            else:
                extra_info = original_data.get("extra_info", {})

        judge_candidates.append(
            (
                idx,
                {
                    "data_source": item["data_source"],
                    "response": item.get("response", ""),
                    "expected_answer": expected_answer,
                    "extra_info": extra_info,
                },
            )
        )

    if not judge_candidates:
        return 0

    asyncio.run(_judge_responses_async(judge_candidates, concurrency))

    updated = 0
    for idx, payload in judge_candidates:
        score = float(payload.get("score", 0.0))
        results[idx]["model_judge_score"] = score
        results[idx]["passed"] = score >= 0.5
        results[idx]["is_correct"] = score >= 0.5
        updated += 1
        print('fafa', results[idx]['passed'], results[idx]["model_judge_score"])
    return updated


def make_preprocess_for_vllm(
    max_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.95,
    model_source: Optional[str] = None,
    tokenizer_source: Optional[str] = None,
    chat_template: Optional[str] = None,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Factory that builds a Ray-serializable preprocess function capturing sampling params.

    Ray Data passes only the row to preprocess, so we capture configuration (like
    max_tokens) in a closure.
    """

    # Lazily initialized tokenizer cached in closure for Ray workers
    tokenizer = None

    def _preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal tokenizer
        assert isinstance(row["msg"], list), f"Prompt is not a list: {row['msg']}"
        if tokenizer is None:
            from transformers.models.auto.tokenization_auto import AutoTokenizer

            src = tokenizer_source or model_source
            assert src is not None, "Tokenizer/model source must be provided"
            tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)

        conversation = row["msg"]
        add_generation_prompt = True
        if len(conversation) > 0 and isinstance(conversation[-1], dict):
            add_generation_prompt = conversation[-1].get("role") == "user"

        prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

        return {
            "prompt": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_tokens": int(max_tokens),
                "top_p": top_p,
            },
            # Keep original data for postprocessing
            "group_id": row["group_id"],
            "sample_id": row["sample_id"],
            "msg": row["msg"],
            "expected_answer": row["expected_answer"],
            "original_data": row["original_data"],
            "data_source": row["data_source"],
        }

    return _preprocess


def postprocess_from_vllm(row: Dict[str, Any]) -> Dict[str, Any]:
    """Postprocess function after vLLM generation."""
    response = str(row.get("generated_text", ""))
    # print('fafa', row)
    
    # Extract answer using utility function
    extracted_answer = extract_answer(response) or ""
    
    # Check if answer is correct using utility function (rule-based)
    expected = str(row.get("expected_answer", "")).strip()
    passed = compare_answers(extracted_answer, expected)
    
    return {
        "data_source": row["data_source"],
        "group_id": row["group_id"],
        "sample_id": row["sample_id"],
        "prompt": row["prompt"],
        "msg": row["msg"],
        "expected_answer": row["expected_answer"],
        "response": response,
        "extracted_answer": extracted_answer,
        "passed": passed,
        # Keep legacy field for downstream compatibility
        "is_correct": passed,
        "generation_time": 0,  # Will be calculated later
        "original_data": row["original_data"]
    }


def calculate_accuracy(results: List[Dict]) -> Dict:
    """Calculate accuracy metrics for the generated responses."""
    total_responses = len(results)
    correct_responses = sum(1 for item in results if item.get('passed', item.get('is_correct', False)))
    
    # Group by problem
    group_accuracy = {}
    for item in results:
        group_id = item['group_id']
        if group_id not in group_accuracy:
            group_accuracy[group_id] = {'correct': 0, 'total': 0}
        group_accuracy[group_id]['total'] += 1
        if item.get('passed', item.get('is_correct', False)):
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
        group_items = [item for item in results if item['group_id'] == group_id]
        # Check if any response in this group is correct
        any_correct = any(item.get('passed', item.get('is_correct', False)) for item in group_items)
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


def save_results(results: List[Dict], output_dir: str, metadata: Optional[Dict] = None):
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
    parser = argparse.ArgumentParser(description="Ray Data + vLLM Batched Inference Script for GRPO")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model for inference")
    parser.add_argument("--prompts_file", type=str, required=True,
                       help="Path to JSONL file containing prompts/problems")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save generated responses")
    parser.add_argument("--n_samples", type=int, default=5,
                       help="Number of samples per prompt")
    parser.add_argument("--min_problems", type=int, default=None,
                       help="Maximum number of problems to process (useful for testing)")
    parser.add_argument("--max_problems", type=int, default=None,
                       help="Maximum number of problems to process (useful for testing)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="Maximum tokens to generate")
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                       help="Number of GPUs for tensor parallelism")
    parser.add_argument("--pipeline_parallel_size", type=int, default=2,
                       help="Number of GPUs for pipeline parallelism")
    parser.add_argument("--concurrency", type=int, default=1,
                       help="Number of parallel vLLM replicas")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for generation")
    parser.add_argument("--max_model_len", type=int, default=16384,
                       help="Maximum model length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                       help="GPU memory utilization")
    parser.add_argument("--disable_progress_bars", action="store_true",
                       help="Disable Ray Data progress bars")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="HF repo id or local path to tokenizer (defaults to model_path)")
    parser.add_argument("--chat_template", type=str, default=None,
                       help="Optional chat template to apply in preprocess")
    parser.add_argument("--use_model_judge", action="store_true", default=False,
                       help="Use model judge for responses that fail rule-based checks")
    parser.add_argument("--model_judge_concurrency", type=int, default=3096,
                       help="Maximum number of concurrent model-judge requests")
    
    args = parser.parse_args()
    
    # Initialize Ray (uncomment to reduce clutter)
    if args.disable_progress_bars:
        ray.init(log_to_driver=False)
        ray.data.DataContext.get_current().enable_progress_bars = False
    else:
        ray.init()
    
    print(f"Ray version: {ray.__version__}")
    print(f"Ray cluster resources: {ray.cluster_resources()}")
    
    # Load and preprocess data
    print(f"Loading prompts from: {args.prompts_file}")
    processed_data = load_and_preprocess_data(
        args.prompts_file, 
        args.min_problems,
        args.max_problems, 
        args.n_samples
    )
    
    # Create Ray Dataset from the processed data
    print("Creating Ray Dataset...")
    ds = ray.data.from_items(processed_data)
    print(f"Dataset schema: {ds.schema()}")
    print(f"Dataset size: {ds.count()} samples")
    
    # Configure vLLM engine with tensor and pipeline parallelism
    print(f"Configuring vLLM engine:")
    print(f"  Model: {args.model_path}")
    print(f"  Tensor parallel size: {args.tensor_parallel_size}")
    print(f"  Pipeline parallel size: {args.pipeline_parallel_size}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max model length: {args.max_model_len}")
    print(f"  GPU memory utilization: {args.gpu_memory_utilization}")
    
    config = vLLMEngineProcessorConfig(  # type: ignore[call-arg]
        model_source=args.model_path,
        engine_kwargs={
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 163840,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": True,
            "trust_remote_code": True,
            # Make tokenizer explicit to avoid AutoProcessor and ensure visibility
            "tokenizer": args.tokenizer_path or args.model_path,
        },
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        # Bypass ChatTemplate/Tokenize/Detokenize stages that invoke AutoProcessor
        apply_chat_template=False,
        tokenize=False,
        detokenize=False,
        has_image=False,
    )
    
    # Create vLLM processor
    print("Building vLLM processor...")
    vllm_processor = build_llm_processor(
        cast(Any, config),
        preprocess=make_preprocess_for_vllm(
            args.max_tokens,
            args.temperature,
            model_source=args.model_path,
            tokenizer_source=args.tokenizer_path or args.model_path,
            chat_template=args.chat_template,
        ),
        postprocess=postprocess_from_vllm,
    )
    
    # Run batch inference
    print("Starting batch inference...")
    start_time = time.time()
    
    ds = vllm_processor(ds)
    
    # Collect all results
    print("Collecting results...")
    results = ds.take_all()
    
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    print(f"Average time per prompt: {generation_time/len(results):.3f} seconds")
    
    # Update generation time for all results
    avg_time = generation_time / len(results)
    for result in results:
        result['generation_time'] = avg_time
    
    if args.use_model_judge:
        print("\nRunning model judge fallback on unmatched responses...")
        # updated = run_model_judge(results, args.model_judge_concurrency)
        updated = run_model_judge(results, 4096)
        print(f"Model judge marked {updated} additional responses as passed.")

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
        'concurrency': args.concurrency,
        'batch_size': args.batch_size,
        'max_model_len': args.max_model_len,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'accuracy_metrics': accuracy_metrics,
        'ray_version': ray.__version__,
        'generation_time': generation_time,
        'model_judge_used': args.use_model_judge,
    }
    
    save_results(results, args.output_dir, metadata)
    print("Ray Data batch inference completed successfully!")
    
    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main() 