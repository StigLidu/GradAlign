#!/usr/bin/env python3
"""
CPU-only pre-tokenization and dataset preparation for parallel GRPO analysis.

- Loads responses from response_sorted.jsonl (or responses_sorted.jsonl fallback)
- Uses multiprocessing with 128 worker processes to tokenize (one tokenizer per process)
- Builds entries like prepare_data in grpo_grad_analyze_parallel.py, WITHOUT model
  fields: input_ids, attention_mask, response_mask, group_id, global_index
- Aggregates, sorts by (group_id, seq_len), then splits into data_{rank}.pt files
  where each split contains entries with idx % world_size == rank in the sorted array.

This avoids any GPU usage during preparation and avoids tokenizer thread-safety issues.
"""

import argparse
import os
import json
from multiprocessing import get_context
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer

# Local utilities
from grpo_utils import create_response_mask


def read_jsonl_or_json_list(responses_dir: str) -> Tuple[List[Dict], str]:
    jsonl_path = os.path.join(responses_dir, 'response_sorted.jsonl')
    jsonl_alt = os.path.join(responses_dir, 'responses_sorted.jsonl')
    json_list = os.path.join(responses_dir, 'responses_sorted.json')

    if os.path.exists(jsonl_path):
        path = jsonl_path
        with open(path, 'r') as f:
            records = [json.loads(line) for line in f if line.strip()]
        return records, path
    if os.path.exists(jsonl_alt):
        path = jsonl_alt
        with open(path, 'r') as f:
            records = [json.loads(line) for line in f if line.strip()]
        return records, path
    if os.path.exists(json_list):
        path = json_list
        with open(path, 'r') as f:
            records = json.load(f)
        return records, path

    raise FileNotFoundError(
        f"Could not find response_sorted.jsonl/responses_sorted.jsonl/responses_sorted.json in {responses_dir}"
    )


TOKENIZER = None
MAX_LENGTH = None


def _proc_init(model_path: str, max_length: int):
    global TOKENIZER, MAX_LENGTH
    TOKENIZER = AutoTokenizer.from_pretrained(model_path)
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token
    MAX_LENGTH = max_length


def tokenize_record(args: Tuple[int, Dict]):
    idx, item = args
    prompt = item.get('prompt', '')
    response = item.get('response', '')
    group_id = int(item['group_id'])

    full_text = prompt + response
    inputs = TOKENIZER(
        full_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    response_mask = create_response_mask(TOKENIZER, prompt, full_text)
    if response_mask.shape[1] > inputs['input_ids'].shape[1] - 1:
        response_mask = response_mask[:, : inputs['input_ids'].shape[1] - 1]

    # Return plain Python lists to avoid torch Tensor fd-passing across processes
    entry = {
        'input_ids': inputs['input_ids'].squeeze(0).tolist(),
        'attention_mask': inputs['attention_mask'].squeeze(0).tolist(),
        'response_mask': response_mask.squeeze(0).tolist(),
        'group_id': group_id,
        'global_index': idx,
    }
    return entry


def seq_len(entry: Dict) -> int:
    # input_ids is a 1D list here
    return len(entry['input_ids'])


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize and split dataset for parallel GRPO analysis (CPU-only)")
    parser.add_argument('--responses_dir', type=str, required=True, help='Directory containing response_sorted.jsonl')
    parser.add_argument('--model_path', type=str, required=True, help='Model path (for tokenizer)')
    parser.add_argument('--world_size', type=int, required=True, help='Number of ranks to split into')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--num_workers', type=int, default=128, help='Number of worker threads for tokenization')
    parser.add_argument('--output_prefix', type=str, default='data_', help='Prefix for split files (data_{rank}.pt or .npz)')
    parser.add_argument('--save_format', type=str, default='npz', choices=['pt','npz'], help='Output format for shards')

    args = parser.parse_args()

    os.makedirs(args.responses_dir, exist_ok=True)
    records, src_path = read_jsonl_or_json_list(args.responses_dir)
    print(f"Loaded {len(records)} records from {src_path}")

    # Parallel tokenization using processes (one tokenizer per process)
    print(f"Tokenizing with {args.num_workers} processes (one tokenizer per process)...")
    entries: List[Dict] = []
    ctx = get_context("fork")
    with ctx.Pool(processes=args.num_workers, initializer=_proc_init, initargs=(args.model_path, args.max_length)) as pool:
        for entry in pool.imap_unordered(tokenize_record, enumerate(records), chunksize=64):
            entries.append(entry)

    print(f"Tokenized {len(entries)} entries. Sorting...")
    entries.sort(key=lambda e: (int(e['group_id']), seq_len(e)))

    # Split into world_size shards by modulo position in sorted array
    print(f"Splitting into {args.world_size} shards and saving...")
    out_dir = os.path.join(args.responses_dir, 'data')
    os.makedirs(out_dir, exist_ok=True)

    def build_and_save(rank: int):
        shard = [e for i, e in enumerate(entries) if (i % args.world_size) == rank]
        if args.save_format == 'pt':
            tensor_shard = []
            for e in shard:
                tensor_shard.append({
                    'input_ids': torch.tensor(e['input_ids'], dtype=torch.int64),
                    'attention_mask': torch.tensor(e['attention_mask'], dtype=torch.uint8),
                    'response_mask': torch.tensor(e['response_mask'], dtype=torch.uint8),
                    'group_id': e['group_id'],
                    'global_index': e['global_index'],
                })
            out_path = os.path.join(out_dir, f"{args.output_prefix}{rank}.pt")
            torch.save(tensor_shard, out_path)
            return rank, len(tensor_shard), out_path
        else:
            import numpy as np
            input_ids_obj = np.array([np.asarray(e['input_ids'], dtype=np.int32) for e in shard], dtype=object)
            attention_obj = np.array([np.asarray(e['attention_mask'], dtype=np.uint8) for e in shard], dtype=object)
            response_obj = np.array([np.asarray(e['response_mask'], dtype=np.uint8) for e in shard], dtype=object)
            group_ids = np.asarray([e['group_id'] for e in shard], dtype=np.int32)
            global_indices = np.asarray([e['global_index'] for e in shard], dtype=np.int64)
            out_path = os.path.join(out_dir, f"{args.output_prefix}{rank}.npz")
            np.savez_compressed(out_path,
                                input_ids=input_ids_obj,
                                attention_mask=attention_obj,
                                response_mask=response_obj,
                                group_id=group_ids,
                                global_index=global_indices)
            return rank, len(shard), out_path

    with ThreadPoolExecutor(max_workers=args.world_size) as executor:
        futures = [executor.submit(build_and_save, rank) for rank in range(args.world_size)]
        for fut in as_completed(futures):
            rank, count, out_path = fut.result()
            print(f"Saved rank {rank} shard with {count} entries to {out_path}")

    print("Preparation complete.")


if __name__ == '__main__':
    main()


