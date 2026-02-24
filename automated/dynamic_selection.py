#!/usr/bin/env python3

import os
import json
import argparse
import subprocess
import sys
from typing import List, Tuple
import time

import datasets as hfds  # type: ignore[import-not-found]

from config import get_dataset_dir, get_response_dir
from launch_verl_training import build_exp_name


def _load_train_dataset(train_dir: str, train_parquet: str):
    parquet_path = os.path.join(train_dir, train_parquet)
    jsonl_path = os.path.join(train_dir, "train.jsonl")
    if os.path.isfile(parquet_path):
        return hfds.load_dataset("parquet", data_files=parquet_path, split="train")
    if os.path.isfile(jsonl_path):
        records: List[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    continue
                records.append(obj)
        return hfds.Dataset.from_list(records)
    raise SystemExit(f"No train parquet or jsonl found in {train_dir}")


def write_jsonl(records: List[dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def chunk_dataset(train_dir: str, exp_root: str, train_parquet: str, chunk_size: int, seed: int) -> Tuple[int, List[str]]:
    ds = _load_train_dataset(train_dir, train_parquet)
    total = len(ds)
    if total == 0:
        raise SystemExit("Empty training dataset; nothing to chunk")
    ds = ds.shuffle(seed=seed)

    if chunk_size <= 0:
        raise SystemExit("--chunk_size must be > 0")
    num_chunks = total // chunk_size
    if num_chunks <= 0:
        num_chunks = 1
        effective_chunk_size = total
    else:
        effective_chunk_size = chunk_size

    leftover = total - (num_chunks * effective_chunk_size)
    if leftover > 0:
        print(f"Info: dropping leftover {leftover} records to match floor(total/chunk_size)")

    chunk_dirs: List[str] = []
    for i in range(num_chunks):
        start = i * effective_chunk_size
        end = start + effective_chunk_size
        indices = list(range(start, min(end, total)))
        if not indices:
            continue
        chunk = ds.select(indices)
        chunk_dir = os.path.join(exp_root, f"chunk_{i}")
        os.makedirs(chunk_dir, exist_ok=True)
        out_parquet = os.path.join(chunk_dir, "train.parquet")
        chunk.to_parquet(out_parquet)
        # Also write JSONL for compatibility with inference scripts
        records = chunk.to_list()
        out_jsonl = os.path.join(chunk_dir, "train.jsonl")
        write_jsonl(records, out_jsonl)
        print(f"Wrote chunk {i}: {len(chunk)} rows -> {out_parquet} and {out_jsonl}")
        chunk_dirs.append(chunk_dir)

    # Write/update manifest
    manifest = {
        "experiment_name": os.path.basename(exp_root.rstrip("/")),
        "train_dir": train_dir,
        "total_rows": total,
        "chunk_size": effective_chunk_size,
        "num_chunks": num_chunks,
        "dropped_rows": leftover if leftover > 0 else 0,
        "seed": seed,
        "train_parquet": train_parquet,
    }
    with open(os.path.join(exp_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return num_chunks, chunk_dirs


def _kill_other_pythons() -> None:
    """Force-kill all other python processes except this one.

    This is a blunt instrument intended to free GPUs/CPUs before heavy stages.
    """
    try:
        # Use pkill to terminate python processes not matching this PID
        # Avoid killing this script by filtering our own PID via shell pattern
        my_pid = str(os.getpid())
        # Build the awk with proper quoting to avoid f-string braces issues
        awk_expr = f"$2~/(python|python3)/ && $1!={my_pid} {{print $1}}"
        shell_cmd = f"ps -eo pid,comm,cmd | awk '{awk_expr}' | xargs -r -n1 -P1 kill -9"
        cmd = ["bash", "-lc", shell_cmd]
        subprocess.run(cmd, check=False)
    except Exception:
        # Best-effort; do not crash orchestrator
        pass


def _set_random_seed(seed: int) -> None:
    # os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import random as _random
        _random.seed(seed)
    except Exception:
        pass
    try:
        import numpy as _np  # type: ignore
        _np.random.seed(seed)
    except Exception:
        pass
    # try:
    #     import torch as _torch  # type: ignore
    #     _torch.manual_seed(seed)
    #     if _torch.cuda.is_available():
    #         _torch.cuda.manual_seed_all(seed)
    #     if hasattr(_torch.backends, "cudnn"):
    #         _torch.backends.cudnn.deterministic = True
    #         _torch.backends.cudnn.benchmark = False
    # except Exception:
    #     pass


def main():
    parser = argparse.ArgumentParser(description="Dynamic selection orchestrator: chunk → infer → analyze → aggregate → select → train")
    parser.add_argument("--prefix", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--train_dataset", required=True, type=str)
    parser.add_argument("--val_dataset", required=True, type=str)

    # Optional explicit overrides (to perfectly match experiment naming)
    parser.add_argument("--train_dir", default=None, type=str,
                        help="Override directory containing the training parquet/jsonl")
    parser.add_argument("--val_dir", default=None, type=str,
                        help="Override directory containing the validation parquet/jsonl")
    parser.add_argument("--train_parquet", default="train.parquet", type=str)

    # Chunking controls
    parser.add_argument("--chunk_size", required=True, type=int,
                        help="Number of items per chunk")
    parser.add_argument("--k", required=True, type=int,
                        help="Number of items per chunk")
    parser.add_argument("--seed", default=42, type=int)

    # Selection controls
    parser.add_argument("--mode", choices=["sim", "simacc", "acc", "rand", "accgreedy", 'align', 'negsim', 'norm', 'dot'], required=True)
    parser.add_argument("--acc_low", type=float, default=0.2)
    parser.add_argument("--acc_high", type=float, default=0.8)
    parser.add_argument("--epochs_per_select", type=int, default=1,
                        help="Train epochs per selection iteration (we run 1 epoch per launch; this controls total iterations with total_epochs)")
    parser.add_argument("--num_selections", type=int, required=True,
                        help="Number of selection iterations to run")
    parser.add_argument("--train_batch_size", type=int, required=True,
                        help="Train batch size used to determine selection size")
    parser.add_argument("--iters_per_select", type=int, required=True,
                        help="Number of training iterations per selection (compute top N = batch_size * iters_per_select)")

    # Inference controls
    parser.add_argument("--minibatch_size", type=int, default=8)
    parser.add_argument("--n_samples_train", type=int, default=32)
    parser.add_argument("--n_samples_val", type=int, default=128)
    parser.add_argument("--min_problems", type=int, default=1)
    parser.add_argument("--max_problems", type=int, default=40000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--train_infer_batch_size", type=int, default=512)
    parser.add_argument("--val_infer_batch_size", type=int, default=512)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--use_optimizer", action="store_true", default=False)
    parser.add_argument("--reward_manager", type=str, default=None)
    parser.add_argument("--reward_path", type=str, default=None)
    
    # Checkpoint/merge controls
    parser.add_argument("--ckpt_root", type=str, default="",
                        help="Root directory where VERL checkpoints are saved")
    parser.add_argument("--merge_backend", type=str, default="megatron")
    parser.add_argument("--verl_val_set", type=str, default="amc_mc_aime_amc_mmlupro")

    args = parser.parse_args()
    _set_random_seed(args.seed)
    assert  (args.k * args.train_batch_size * args.iters_per_select) % args.chunk_size == 0
    train_epochs = (args.k * args.train_batch_size * args.iters_per_select) // args.chunk_size

    # Resolve dataset dirs when not explicitly provided
    train_dir = args.train_dir or get_dataset_dir(args.train_dataset, args.model)
    val_dir = args.val_dir or get_dataset_dir(args.val_dataset, args.model)

    # Use a fixed experiment name (prefix + model + datasets) to align with VERL checkpoints
    exp_name = build_exp_name(args.prefix, args.model, args.train_dataset, train_dir, val_dir)
    exp_root = os.path.join(train_dir, exp_name)
    os.makedirs(exp_root, exist_ok=True)

    # Prepare experiment checkpoint root and chunk storage under it
    # ckpt_root/{exp_name}/chunks/chunk_{i}
    exp_ckpt_root = os.path.join(args.ckpt_root, exp_name)
    os.makedirs(exp_ckpt_root, exist_ok=True)
    chunks_root = os.path.join(exp_ckpt_root, "chunks")
    os.makedirs(chunks_root, exist_ok=True)

    # 1) Chunk (idempotent with auto-resume): if chunks already exist, skip splitting
    existing_chunks = [d for d in os.listdir(chunks_root) if d.startswith("chunk_") and os.path.isdir(os.path.join(chunks_root, d))]
    if existing_chunks:
        existing_chunks.sort(key=lambda name: int(name.split("_")[-1]))
        num_chunks = len(existing_chunks)
        chunk_dirs = [os.path.join(chunks_root, d) for d in existing_chunks]
        print(f"Found existing {num_chunks} chunks under {chunks_root}; skipping chunking")
    else:
        num_chunks, chunk_dirs = chunk_dataset(train_dir, chunks_root, args.train_parquet, args.chunk_size, args.seed)
        print(f"Manifest written at {os.path.join(chunks_root, 'manifest.json')}")

    # Selection size per iteration
    select_n = args.chunk_size // args.k

    # Precompute chunk sizes (dataset rows per chunk) for resume logic
    def _count_lines(path: str) -> int:
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for _ in f:
                n += 1
        return n

    chunk_sizes = []
    for d in chunk_dirs:
        chunk_sizes.append(_count_lines(os.path.join(d, "train.jsonl")))

    # Auto-resume: find earliest incomplete iteration
    i_start = 0
    val_lines = _count_lines(os.path.join(val_dir, "train.jsonl")) if os.path.isfile(os.path.join(val_dir, "train.jsonl")) else 0
    for i_probe in range(1, args.num_selections + 10):
        step_root_probe = os.path.join(exp_ckpt_root, f"global_step_{i_probe * args.iters_per_select}")
        if os.path.isdir(step_root_probe):
            i_start = i_probe
            # break

    if i_start > 0:
        print(f"Auto-resume: resuming from iteration {i_start}")

    for i in range(i_start, args.num_selections):
        # Hard kill stray python processes at the start of each iteration
        _kill_other_pythons()
        chunk_idx = i % num_chunks
        chunk_dir = os.path.join(chunks_root, f"chunk_{chunk_idx}")
        prompts_file = os.path.join(chunk_dir, "train.jsonl")
        # Fixed global step directory naming to align with VERL
        gstep = i * args.iters_per_select
        step_root = os.path.join(exp_ckpt_root, f"global_step_{gstep}")
        train_parts_root = os.path.join(step_root, "train_split")
        val_dir_override = os.path.join(step_root, "val_responses")
        train_resp_dir = os.path.join(train_parts_root, "part_0")
        os.makedirs(train_resp_dir, exist_ok=True)
        os.makedirs(val_dir_override, exist_ok=True)

        # 2) If i>0: merge previous iteration checkpoint and override model path
        merged_model_path = None
        if i > 0:
            # Expect VERL to have produced checkpoints under ckpt_root/{prefix}_sel{iter-1}_{model}_{train_dataset}_{train_name}_{val_name}/global_step_{i}
            # We'll merge that step to dest_root and use it for inference/analysis
            train_name = os.path.basename(os.path.normpath(train_dir))
            val_name = os.path.basename(os.path.normpath(val_dir))
            # Use fixed experiment name rather than per-iter suffix
            exp_prev = exp_name
            # Write merged model directly into exp ckpt dir at this global step
            merged_target = os.path.join(exp_ckpt_root, f"global_step_{gstep}", "merged")
            os.makedirs(merged_target, exist_ok=True)
            merge_cmd = [
                sys.executable, os.path.join(os.path.dirname(__file__), "merge_model.py"),
                "--experiment_name", exp_prev,
                "--step", str(i * args.iters_per_select),
                "--output_model_name", f"{exp_prev}",
                "--backend", args.merge_backend,
                "--ckpt_root", args.ckpt_root,
                "--dest_root", os.path.join(args.ckpt_root, "merged_models"),
                "--target_dir", merged_target,
            ]
            print("Executing:", " ".join(merge_cmd))
            subprocess.run(merge_cmd, check=True)
            merged_model_path = merged_target
            if args.use_optimizer:
                optimizer_state_path = os.path.join(step_root, "opt_converted")
                convert_cmd = [
                    sys.executable, os.path.join(os.path.dirname(__file__), "convert_megatron_optimizer_to_hf.py"),
                    "--checkpoint_path", os.path.join(step_root, "actor"),
                    "--hf-config", merged_model_path,
                    "--output", optimizer_state_path,
                ]
                print("Executing:", " ".join(convert_cmd))
                subprocess.run(convert_cmd, check=True)
                if not os.path.isfile(optimizer_state_path):
                    raise SystemExit(f"Optimizer state not found: {optimizer_state_path}")

        # 3) Local training inference → responses.json in step-root train_split/part_0
        infer_cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), "run_inference_local.py"),
            "--model", args.model,
            "--resp_dataset", args.train_dataset,
            "--prompts_file", prompts_file,
            "--output_part", "0",
            "--parts_root", train_parts_root,
            *( ["--model_path", merged_model_path] if merged_model_path else [] ),
            "--n_samples", str(args.n_samples_train),
            "--min_problems", str(args.min_problems),
            "--max_problems", str(args.max_problems),
            "--temperature", str(args.temperature),
            "--max_tokens", str(args.max_tokens),
            "--tensor_parallel_size", str(args.tensor_parallel_size),
            "--pipeline_parallel_size", str(args.pipeline_parallel_size),
            "--batch_size", str(args.train_infer_batch_size),
            "--max_model_len", str(args.max_model_len),
            "--concurrency", str(args.concurrency),
        ]
        if args.reward_manager:
            infer_cmd.append("--use_model_judge")
        # infer_cmd.extend(["--model_judge_concurrency", str(max(1, args.concurrency))])

        _kill_other_pythons()
        # Skip train inference if responses already complete
        expected_train = chunk_sizes[chunk_idx] * args.n_samples_train
        train_part_dir = os.path.join(train_parts_root, "part_0")
        resp_json = os.path.join(train_part_dir, "responses.json")
        resp_sorted = os.path.join(train_part_dir, "responses_sorted.json")
        actual_train = 0
        if os.path.isfile(resp_json):
            try:
                with open(resp_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        actual_train = len(data)
            except json.JSONDecodeError:
                actual_train = 0
        elif os.path.isfile(resp_sorted):
            try:
                with open(resp_sorted, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        actual_train = len(data)
            except json.JSONDecodeError:
                actual_train = 0
        if actual_train >= expected_train and expected_train > 0:
            print(f"Skip train inference for step {i}: found {actual_train}/{expected_train} responses")
        else:
            print("Executing:", " ".join(infer_cmd))
            subprocess.run(infer_cmd, check=True)

        # 3b) Local validation inference → responses.json in step-root val_responses (flat)
        if args.mode in ['sim', 'simacc', 'negsim', 'dot']:
            val_prompts_file = os.path.join(val_dir, "train.jsonl")
            if not os.path.isfile(val_prompts_file):
                raise SystemExit(f"Validation prompts not found: {val_prompts_file}")
            infer_val_cmd = [
                sys.executable, os.path.join(os.path.dirname(__file__), "run_inference_local.py"),
                "--model", args.model,
                "--resp_dataset", args.val_dataset,
                "--prompts_file", val_prompts_file,
                "--output_part", "0",
                "--output_dir", val_dir_override,
                *( ["--model_path", merged_model_path] if merged_model_path else [] ),
                "--n_samples", str(args.n_samples_val),
                "--min_problems", str(args.min_problems),
                "--max_problems", str(args.max_problems),
                "--temperature", str(args.temperature),
                "--max_tokens", str(args.max_tokens),
                "--tensor_parallel_size", str(args.tensor_parallel_size),
                "--pipeline_parallel_size", str(args.pipeline_parallel_size),
                "--batch_size", str(args.val_infer_batch_size),
                "--max_model_len", str(args.max_model_len),
                "--concurrency", str(args.concurrency),
            ]
            if args.reward_manager:
                infer_val_cmd.append("--use_model_judge")
                # infer_val_cmd.extend(["--model_judge_concurrency", str(max(1, args.concurrency))])
            _kill_other_pythons()
            time.sleep(10)
            # Skip val inference if responses already complete
            expected_val = val_lines * args.n_samples_val if val_lines > 0 else 0
            val_resp_json = os.path.join(val_dir_override, "responses.json")
            val_resp_sorted = os.path.join(val_dir_override, "responses_sorted.json")
            actual_val = 0
            if os.path.isfile(val_resp_sorted):
                try:
                    with open(val_resp_sorted, "r", encoding="utf-8") as f:
                        v = json.load(f)
                        if isinstance(v, list):
                            actual_val = len(v)
                except json.JSONDecodeError:
                    actual_val = 0
            elif os.path.isfile(val_resp_sorted):
                try:
                    with open(val_resp_sorted, "r", encoding="utf-8") as f:
                        v = json.load(f)
                        if isinstance(v, list):
                            actual_val = len(v)
                except json.JSONDecodeError:
                    actual_val = 0
            if expected_val > 0 and actual_val >= expected_val:
                print(f"Skip val inference for step {i}: found {actual_val}/{expected_val} responses")
            else:
                print("Executing:", " ".join(infer_val_cmd))
                subprocess.run(infer_val_cmd, check=True)

        # 3) Analysis and aggregation (for sim/simacc)
        _kill_other_pythons()
        time.sleep(5)
        if args.mode in {"sim", "simacc", "align", "negsim", 'norm', 'dot'}:
            if args.mode == 'align' or args.mode == 'norm':
                args.val_dataset = args.train_dataset
                val_dir_override = train_resp_dir
            ana_cmd = [
                sys.executable, os.path.join(os.path.dirname(__file__), "run_parallel_analysis.py"),
                "--size", "1",
                "--idx", "0",
                "--model", args.model,
                "--resp_dataset", args.train_dataset,
                "--problem_dataset", args.train_dataset,
                "--val_dataset", args.val_dataset,
                "--parts_root", train_parts_root,
                "--val_responses_dir", val_dir_override,
                "--num_gpus", "8",
                "--mini_batch_size", str(args.minibatch_size),
                "--mixed_precision", "bf16",
                "--max_length", str(args.max_model_len),
            ]
            if merged_model_path:
                ana_cmd.extend(["--model_path", merged_model_path])
            if args.mode == 'norm' or args.mode == 'dot':
                # Pass mode as separate CLI argument and value
                ana_cmd.extend(["--mode", args.mode])
            # Skip analysis if per-response similarity already computed
            sim_path = os.path.join(train_part_dir, "similarity_results_cosine_real.jsonl")
            sim_lines = _count_lines(sim_path) if os.path.isfile(sim_path) else 0
            if sim_lines >= expected_train and expected_train > 0:
                print(f"Skip analysis for step {i}: found {sim_lines}/{expected_train} similarity rows")
            else:
                print("Executing:", " ".join(ana_cmd))
                subprocess.run(ana_cmd, check=True)

        # After inference and analysis, aggressively clean up other python processes
        _kill_other_pythons()
        time.sleep(10)

        # Aggregate accuracy and copy similarity
        if args.mode in {"sim", "simacc", "acc", "accgreedy", 'align', 'negsim', 'norm', 'dot'}:
            agg_cmd = [
                sys.executable, os.path.join(os.path.dirname(__file__), "aggregate.py"),
                "--model_name", args.model,
                "--dataset", args.train_dataset,
                "--parts_root", train_parts_root,
            ]
            print("Executing:", " ".join(agg_cmd))
            subprocess.run(agg_cmd, check=True)

        # 4) Select top-N according to mode
        iter_out = os.path.join(exp_ckpt_root, "selected", f"iter_{i}_{args.mode}_{select_n}")
        os.makedirs(iter_out, exist_ok=True)
        select_mode = args.mode
        if select_mode == 'dot' or select_mode == 'norm':
            select_mode = 'sim'
        sel_cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), "select_data.py"),
            "--mode", select_mode,
            "--dataset", args.train_dataset,
            "--model", args.model,
            "--n", str(select_n),
            "--output_dir", iter_out,
        ]
        if args.mode in {"sim", "simacc", "acc", "accgreedy", "align", "negsim", 'norm', 'dot'}:
            sel_cmd.extend(["--parts_root", train_parts_root])
        if args.mode == "acc":
            sel_cmd.extend(["--acc_low", str(args.acc_low), "--acc_high", str(args.acc_high)])
        print("Executing:", " ".join(sel_cmd))
        subprocess.run(sel_cmd, check=True)
        _kill_other_pythons()
        time.sleep(10)

        # 5) Launch training with selected set, fixed experiment name
        os.system(f"rm {step_root}/data.pt")
        # print('remove data.pt')
        train_cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), "launch_verl_training.py"),
            "--prefix", args.prefix,
            "--model", args.model,
            "--train_dataset", args.train_dataset,
            "--val_dataset", args.verl_val_set,
            "--train_dir", iter_out,
            "--total_epochs", f'{(i+1) * train_epochs}',
            "--exp_name", exp_name,
            '--train_batch_size', str(args.train_batch_size),
            '--max_response_length', str(args.max_tokens),
        ]
        if args.reward_manager:
            train_cmd.extend(["--reward_manager", args.reward_manager])
        if args.reward_path:
            train_cmd.extend(["--reward_path", args.reward_path])
            
        print("Executing:", " ".join(train_cmd))
        subprocess.run(train_cmd, check=True)
        _kill_other_pythons()
        time.sleep(10)


if __name__ == "__main__":
    main()


