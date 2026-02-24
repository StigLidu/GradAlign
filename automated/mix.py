#!/usr/bin/env python3

import os
import json
import random
import argparse
from typing import List
from config import get_model_type

import datasets

from config import get_dataset_dir, BASE_DATA_DIR


def mix_datasets(
    dataset_names: List[str],
    num_samples: List[int],
    l_thresholds: List[int],
    output_name: str,
) -> str:
    """Mix slices from multiple prepared datasets for a given model.

    - For each dataset i, draw exactly num_samples[i] lines from
      the prepared file range (l_thresholds[i]+1) .. (l_thresholds[i]+num_samples[i]).
    - Shuffle the combined entries, reindex extra_info.index/original_index, and save.

    Returns the path to the mixed train.jsonl
    """
    assert len(dataset_names) == len(num_samples) == len(l_thresholds), (
        "dataset_names, num_samples, and l_thresholds must have the same length"
    )

    out_dir = f"{BASE_DATA_DIR}/{output_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/train.jsonl"

    entries = []
    for ds, n, l in zip(dataset_names, num_samples, l_thresholds):
        ds_dir = get_dataset_dir(ds)
        in_path = f"{ds_dir}/train.jsonl"
        if not os.path.exists(in_path):
            raise FileNotFoundError(f"Input not found: {in_path}. Prepare the dataset first.")

        start_line = l + 1
        end_line = l + n

        with open(in_path, "r", encoding="utf-8") as f:
            line_idx = 1
            taken = 0
            for line in f:
                if line_idx > end_line:
                    break
                if line_idx >= start_line:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in {in_path} at line {line_idx}: {e}")
                    entries.append(entry)
                    taken += 1
                line_idx += 1

        # if taken != n:
        #     raise ValueError(
        #         f"Requested {n} samples from {ds} lines [{start_line}, {end_line}] but got {taken}."
        #     )

    random.shuffle(entries)
    for i, entry in enumerate(entries):
        if "extra_info" not in entry:
            entry["extra_info"] = {}
        entry["extra_info"]["index"] = i
        entry["extra_info"]["original_index"] = i

    with open(out_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    parquet_path = out_path.replace(".jsonl", ".parquet")
    datasets.Dataset.from_list(entries).to_parquet(parquet_path)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Mix slices from prepared datasets.")
    # parser.add_argument("--model", required=True, type=str, help="Model key used for prepared datasets")
    parser.add_argument("--datasets", required=True, type=str, nargs="+", help="Dataset names in order")
    parser.add_argument("--nums", required=True, type=int, nargs="+", help="Number of samples per dataset")
    parser.add_argument("--ls", required=True, type=int, nargs="+", help="l thresholds per dataset")
    parser.add_argument("--output_name", required=True, type=str, help="Name for the mixed dataset directory")

    args = parser.parse_args()
    if not (len(args.datasets) == len(args.nums) == len(args.ls)):
        raise SystemExit("datasets, nums, and ls must be the same length")

    out = mix_datasets(args.datasets, args.nums, args.ls, args.output_name)
    print(f"Mixed dataset written to: {out}")


if __name__ == "__main__":
    main()


