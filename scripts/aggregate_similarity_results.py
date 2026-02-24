#!/usr/bin/env python3
"""
Aggregate similarity results from multiple part directories into a single JSONL.

It scans <parts_root>/part_*/similarity_results_cosine_real.jsonl in ascending
numeric order of part indices and concatenates all lines into the output file.

Usage:
  python scripts/aggregate_similarity_results.py \
    --parts-root ~/data_selection/data/responses/dapo_qwen_128_ckpt_parts \
    --output ~/data_selection/data/responses/dapo_qwen_128_ckpt_parts/similarity_results_aggregated.jsonl
"""

import argparse
import json
import os
import re
import sys
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate similarity JSONL results from part directories",
    )
    parser.add_argument(
        "--parts-root",
        type=str,
        required=True,
        help="Root directory containing part_{i} subdirectories",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the aggregated JSONL file",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="similarity_results_cosine_real.jsonl",
        help="Per-part JSONL filename to aggregate (default: similarity_results_cosine_real.jsonl)",
    )
    return parser.parse_args()


def find_part_dirs(parts_root: str) -> List[str]:
    part_dirs = []
    pattern = re.compile(r"^part_(\d+)$")
    if not os.path.isdir(parts_root):
        print(f"Error: parts root not found: {parts_root}")
        sys.exit(1)
    for name in os.listdir(parts_root):
        full = os.path.join(parts_root, name)
        if os.path.isdir(full):
            m = pattern.match(name)
            if m:
                idx = int(m.group(1))
                part_dirs.append((idx, full))
    part_dirs.sort(key=lambda x: x[0])
    return [d for _, d in part_dirs]


def aggregate(parts_root: str, output_path: str, per_part_filename: str) -> None:
    part_dirs = find_part_dirs(parts_root)
    if not part_dirs:
        print(f"No part_* directories found under {parts_root}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_lines = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for part_dir in part_dirs:
            in_file = os.path.join(part_dir, per_part_filename)
            if not os.path.isfile(in_file):
                print(f"Warning: missing file, skipping: {in_file}")
                continue
            with open(in_file, "r", encoding="utf-8") as in_f:
                for line_num, line in enumerate(in_f, 1):
                    s = line.strip()
                    if not s:
                        continue
                    # Validate JSON line to avoid corrupt output
                    try:
                        json.loads(s)
                    except json.JSONDecodeError as e:
                        print(f"Warning: skipping malformed JSON in {in_file}:{line_num}: {e}")
                        continue
                    out_f.write(s + "\n")
                    total_lines += 1

    print(f"Wrote {total_lines} lines to {output_path}")


def main() -> None:
    args = parse_args()
    aggregate(args.parts_root, args.output, args.filename)


if __name__ == "__main__":
    main()


