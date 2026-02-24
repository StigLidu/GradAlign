#!/usr/bin/env python3

import os
import argparse
import subprocess
import sys

from config import get_response_dir, get_model_path


def main():
    parser = argparse.ArgumentParser(description="Run local Ray+vLLM inference on a prompts file and sort outputs")
    parser.add_argument("--model", required=True, type=str, help="Model key used to resolve model path")
    parser.add_argument("--resp_dataset", required=True, type=str, help="Dataset key under which to place responses")
    parser.add_argument("--prompts_file", required=True, type=str, help="Prompts JSONL file to run inference on")
    parser.add_argument("--output_part", default="0", type=str, help="part index to write into responses split")
    parser.add_argument("--parts_root", default=None, type=str, help="Override parts root (will write into parts_root/part_<idx>)")
    parser.add_argument("--output_dir", default=None, type=str, help="Flat output dir override (write files directly here)")
    parser.add_argument("--model_path", default=None, type=str, help="Override model path for inference")

    # vLLM controls (mirrors run_rollout_local.sh defaults)
    parser.add_argument("--n_samples", default=128, type=int)
    parser.add_argument("--tensor_parallel_size", default=1, type=int)
    parser.add_argument("--pipeline_parallel_size", default=1, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--max_model_len", default=4096, type=int)
    parser.add_argument("--min_problems", default=1, type=int)
    parser.add_argument("--max_problems", default=40000, type=int)
    parser.add_argument("--concurrency", default=8, type=int)
    parser.add_argument("--max_tokens", default=4096, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--use_model_judge", action="store_true", default=False,
                        help="Enable model judge fallback after rule-based checks")
    parser.add_argument("--model_judge_concurrency", default=1024, type=int,
                        help="Maximum concurrent requests for model judge")

    args = parser.parse_args()

    model_path = args.model_path or get_model_path(args.model)
    # Determine destination directory
    if args.output_dir:
        dest_dir = args.output_dir
    else:
        resp_root = args.parts_root or (get_response_dir(args.resp_dataset, args.model) + "_split")
        dest_dir = os.path.join(resp_root, f"part_{args.output_part}")
    os.makedirs(dest_dir, exist_ok=True)

    # 1) Run local Ray vLLM inference
    ray_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "select", "inference_ray_batch.py"))
    cmd = [
        sys.executable, ray_script,
        "--model_path", model_path,
        "--prompts_file", args.prompts_file,
        "--output_dir", dest_dir,
        "--n_samples", str(args.n_samples),
        "--tensor_parallel_size", str(args.tensor_parallel_size),
        "--pipeline_parallel_size", str(args.pipeline_parallel_size),
        "--batch_size", str(args.batch_size),
        "--max_model_len", str(args.max_model_len),
        "--min_problems", str(args.min_problems),
        "--max_problems", str(args.max_problems),
        "--concurrency", str(args.concurrency),
        "--max_tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        # temperature is not directly accepted by inference_ray_batch.py sampling params; keep default inside
    ]
    if args.use_model_judge:
        cmd.append("--use_model_judge")
        cmd.extend(["--model_judge_concurrency", str(max(1, args.model_judge_concurrency))])

    print("Executing:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # 2) Sort responses.json -> responses_sorted.json
    sort_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "sort_responses.py"))
    sort_cmd = [
        sys.executable, sort_script,
        "--model_name", args.model,
        "--dataset", args.resp_dataset,
        "--responses_dir", dest_dir,
    ]
    print("Executing:", " ".join(sort_cmd))
    subprocess.run(sort_cmd, check=True)

    print(f"Local inference completed. Outputs at: {dest_dir}")


if __name__ == "__main__":
    main()


