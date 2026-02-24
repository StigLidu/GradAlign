#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess

from config import get_dataset_dir, get_response_dir, get_model_path


def part_dir_exists(parts_root: str, part_index: int) -> bool:
    return os.path.isdir(os.path.join(parts_root, f"part_{part_index}"))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate part indices as size*k+idx and run parallel analysis for each existing part."
        )
    )
    parser.add_argument("--size", required=True, type=int, help="Number of machines")
    parser.add_argument("--idx", required=True, type=int, help="This machine index [0..size-1]")

    # Response split location (consistent with split_responses.py convention)
    parser.add_argument("--model", required=True, type=str, help="Model key (used for responses dir and dataset resolution)")
    parser.add_argument("--resp_dataset", required=True, type=str, help="Dataset name used for responses")
    parser.add_argument("--parts_root", default=None, type=str, help="Override responses split root (use parts_root/part_*)")

    # Problem set used in analysis
    parser.add_argument("--problem_dataset", required=True, type=str, help="Dataset name for prompts")

    # Validation responses directory used by analysis
    parser.add_argument("--val_dataset", required=True, type=str, help="Validation responses dataset name")
    parser.add_argument("--val_responses_dir", default=None, type=str, help="Override validation responses directory")

    # Other analysis hyperparameters
    parser.add_argument("--num_gpus", default=8, type=int)
    parser.add_argument("--mini_batch_size", default=2, type=int)
    parser.add_argument("--mixed_precision", default="bf16", type=str)
    parser.add_argument("--max_length", default=4096, type=int)
    parser.add_argument("--cpu_offload", action="store_true", default=True)
    parser.add_argument("--max_num_samples", default=10240000, type=int)
    # Model path resolved from config
    # (no explicit --model_path; derived from --model)
    parser.add_argument("--start_k", default=0, type=int, help="Start enumeration from this k (default 0)")
    parser.add_argument("--max_k", default=None, type=int, help="Optional max k to try (exclusive)")
    parser.add_argument("--model_path", default=None, type=str, help="Override model path for analysis")
    parser.add_argument("--use_optimizer", action="store_true", default=False)
    parser.add_argument("--optimizer_state_path", default=None, type=str)
    parser.add_argument("--mode", default='sim', type=str)

    args = parser.parse_args()

    # assert 0 <= args.idx < args.size, "idx must be in [0, size)"

    parts_root = args.parts_root or (get_response_dir(args.resp_dataset, args.model) + "_split")
    problem_set = os.path.join(get_dataset_dir(args.problem_dataset, args.model), "train.jsonl")
    val_responses_dir = args.val_responses_dir or get_response_dir(args.val_dataset, args.model)

    if not os.path.isfile(problem_set):
        raise SystemExit(f"Problem set not found: {problem_set}")
    if not os.path.isdir(parts_root):
        raise SystemExit(f"Parts root not found: {parts_root}")
    if not os.path.isdir(val_responses_dir):
        raise SystemExit(f"Validation responses dir not found: {val_responses_dir}")

    parallel_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "select", "parallel"))
    launcher = os.path.join(parallel_dir, "launch_parallel_analysis.py")

    # Resolve model path from CLI override, env, or config
    model_path = args.model_path or os.environ.get("MODEL_PATH_OVERRIDE", get_model_path(args.model))

    k = args.start_k
    processed_any = False
    while True:
        if args.max_k is not None and k >= args.max_k:
            break
        part_index = args.size * k + args.idx
        if not part_dir_exists(parts_root, part_index):
            # Stop if the next assigned part is missing
            if not processed_any and k == args.start_k:
                print(f"No part found for part_index={part_index} at {parts_root}")
            break

        part_dir = os.path.join(parts_root, f"part_{part_index}")
        output_path = os.path.join(part_dir, "similarity_results_cosine_real.jsonl")

        cmd = [
            sys.executable, launcher,
            "--model_path", model_path,
            "--train_responses_dir", part_dir,
            "--val_responses_dir", val_responses_dir,
            "--output_path", output_path,
            "--problem_set_path", problem_set,
            "--num_gpus", str(args.num_gpus),
            "--mini_batch_size", str(args.mini_batch_size),
            "--mixed_precision", args.mixed_precision,
            "--max_length", str(args.max_length),
            "--mode", args.mode,
        ]
        if args.cpu_offload:
            cmd.append("--cpu_offload")
        if args.max_num_samples is not None:
            cmd.extend(["--max_num_samples", str(args.max_num_samples)])
        if args.use_optimizer:
            cmd.append("--use_optimizer")
            cmd.extend(["--optimizer_state_path", args.optimizer_state_path])

        env = os.environ.copy()
        env.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
        env.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "7200")

        print("Executing (cwd=parallel):", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=parallel_dir, env=env)
        processed_any = True
        k += 1

    print("Done.")


if __name__ == "__main__":
    main()


