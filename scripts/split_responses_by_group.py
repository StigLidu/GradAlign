#!/usr/bin/env python3
"""
Split a list of problems (JSON) into k roughly balanced parts while keeping
all items with the same "group_id" together.

Input file schema:
- JSON array of dicts. Each dict must have a "group_id" key.

Output layout:
- <output_dir>/part_{i}/responses_sorted.json  (i starts at --start-index)

Balancing strategy:
- Greedy bin-packing by group size: sort groups by size (desc), assign each
  next group to the bin with the smallest total so far.

Usage example:
  python scripts/split_responses_by_group.py \
    --k 8 \
    --input ~/data_selection/data/responses/dapo_qwen_128_ckpt_new/responses_sorted.json \
    --output-dir ~/data_selection/data/responses/dapo_qwen_128_ckpt_new
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Any, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split responses_sorted.json into k balanced parts by group_id.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="!/data_selection/data/responses/dapo_qwen_128_ckpt_new/responses_sorted.json",
        help="Path to input responses_sorted.json",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="~/data_selection/data/responses/dapo_qwen_128_ckpt_new",
        help="Directory where part_{i}/responses_sorted.json will be written",
    )
    parser.add_argument(
        "--k",
        "-k",
        type=int,
        required=True,
        help="Number of parts to split into",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for part folders (default: 0)",
    )
    return parser.parse_args()


def read_items(input_path: str) -> List[Dict[str, Any]]:
    assert os.path.isfile(input_path), f"Input file not found: {input_path}"
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "Input JSON must be a list"
    return data


def group_by_group_id(items: List[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for item in items:
        assert isinstance(item, dict), "Each item must be a dict"
        assert "group_id" in item, "Each item must contain 'group_id'"
        gid = item["group_id"]
        if gid not in groups:
            groups[gid] = []
        groups[gid].append(item)
    return groups


def greedy_balance(groups: Dict[Any, List[Dict[str, Any]]], k: int) -> List[Dict[str, Any]]:
    assert k > 0, "k must be > 0"
    group_sizes: List[Tuple[Any, int]] = [(gid, len(items)) for gid, items in groups.items()]
    # Largest-first placement
    group_sizes.sort(key=lambda x: x[1], reverse=True)

    bins: List[Dict[str, Any]] = [
        {"total": 0, "groups": []} for _ in range(k)
    ]

    for gid, size in group_sizes:
        # Choose bin with minimum total
        min_index = min(range(k), key=lambda idx: bins[idx]["total"])
        bins[min_index]["groups"].append(gid)
        bins[min_index]["total"] += size

    return bins


def write_parts(
    bins: List[Dict[str, Any]],
    groups: Dict[Any, List[Dict[str, Any]]],
    output_dir: str,
    start_index: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for idx, bin_info in enumerate(bins):
        part_index = start_index + idx
        part_dir = os.path.join(output_dir, f"part_{part_index}")
        os.makedirs(part_dir, exist_ok=True)

        merged: List[Dict[str, Any]] = []
        for gid in bin_info["groups"]:
            merged.extend(groups[gid])

        out_path = os.path.join(part_dir, "responses_sorted.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        print(f"Wrote {len(merged)} items to {out_path}")


def main() -> None:
    args = parse_args()

    items = read_items(args.input)
    assert len(items) > 0, "Input list is empty"

    groups = group_by_group_id(items)
    bins = greedy_balance(groups, args.k)

    totals = [b["total"] for b in bins]
    print("Group count:", len(groups))
    print("Bin totals (item counts):", totals, "min=", min(totals), "max=", max(totals))

    write_parts(bins, groups, args.output_dir, args.start_index)


if __name__ == "__main__":
    main()


