#!/usr/bin/env python3

import os
import json
import argparse
import re
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import get_response_dir


def find_part_dirs(parts_root: str) -> List[str]:
    part_dirs: List[Tuple[int, str]] = []
    pattern = re.compile(r"^part_(\d+)$")
    if not os.path.isdir(parts_root):
        raise FileNotFoundError(f"Parts root not found: {parts_root}")
    for name in os.listdir(parts_root):
        full = os.path.join(parts_root, name)
        if os.path.isdir(full):
            m = pattern.match(name)
            if m:
                idx = int(m.group(1))
                part_dirs.append((idx, full))
    part_dirs.sort(key=lambda x: x[0])
    return [d for _, d in part_dirs]


def aggregate_similarity(parts_root: str, per_part_filename: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_lines = 0
    sims = []
    with open(output_path, "w", encoding="utf-8") as out_f:
        for part_dir in find_part_dirs(parts_root):
            in_file = os.path.join(part_dir, per_part_filename)
            if not os.path.isfile(in_file):
                continue
            with open(in_file, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    s = line.strip()
                    if not s:
                        continue
                    # Basic validation to avoid corrupt lines
                    try:
                        sims.append(json.loads(s))
                    except json.JSONDecodeError:
                        continue
                    out_f.write(s + "\n")
                    total_lines += 1
    return sims


def _acc_counts_for_part(part_dir: str) -> Tuple[Dict[int, int], Dict[int, int]]:
    resp_path = os.path.join(part_dir, "responses_sorted.json")
    total: Dict[int, int] = {}
    correct: Dict[int, int] = {}
    if not os.path.isfile(resp_path):
        return total, correct
    # Each file is a JSON list; load per part to keep memory bounded
    with open(resp_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
    if not isinstance(data, list):
        return total, correct
    for item in data:
        if not isinstance(item, dict):
            continue
        gid = item.get("group_id")
        is_correct = item.get("is_correct")
        if gid is None or not isinstance(is_correct, bool):
            continue
        total[gid] = total.get(gid, 0) + 1
        if is_correct:
            correct[gid] = correct.get(gid, 0) + 1
    return total, correct


def aggregate_accuracy(parts_root: str, output_path: str, num_workers: int, sim_map: Dict[int, float] | None = None) -> int:
    part_dirs = find_part_dirs(parts_root)
    if not part_dirs:
        raise FileNotFoundError(f"No part_* directories under {parts_root}")

    agg_total: Dict[int, int] = {}
    agg_correct: Dict[int, int] = {}

    with ProcessPoolExecutor(max_workers=max(1, num_workers)) as ex:
        futures = {ex.submit(_acc_counts_for_part, d): d for d in part_dirs}
        for fut in as_completed(futures):
            t_map, c_map = fut.result()
            for gid, t in t_map.items():
                agg_total[gid] = agg_total.get(gid, 0) + t
            for gid, c in c_map.items():
                agg_correct[gid] = agg_correct.get(gid, 0) + c

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    written = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for gid in sorted(agg_total.keys()):
            t = agg_total[gid]
            c = agg_correct.get(gid, 0)
            if t <= 0:
                continue
            if sim_map is not None and gid not in sim_map:
                continue
            rec = {
                "group_id": gid,
                "total": t,
                "correct": c,
                "accuracy": c / float(t),
                'similarity': (sim_map[gid] if sim_map is not None and gid in sim_map else None),
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    return written


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate similarity results and compute per-problem accuracy in parallel."
        )
    )
    parser.add_argument("--model_name", required=True, type=str, help="Model name used for responses")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset name of responses")
    parser.add_argument("--parts_root", default=None, type=str, help="Override parts root (use parts_root/part_*)")
    parser.add_argument("--num_workers", default=16, type=int, help="Number of parallel workers for accuracy")
    parser.add_argument(
        "--similarity_filename",
        default="similarity_results_cosine_real.jsonl",
        type=str,
        help="Per-part similarity filename to aggregate",
    )

    args = parser.parse_args()

    parts_root = args.parts_root or (get_response_dir(args.dataset, args.model_name) + "_split")
    if not os.path.isdir(parts_root):
        raise SystemExit(f"Parts root not found: {parts_root}")

    sim_out = os.path.join(parts_root, "similarity_results_aggregated.jsonl")
    acc_out = os.path.join(parts_root, "accuracy_by_problem.jsonl")

    total_sim = aggregate_similarity(parts_root, args.similarity_filename, sim_out)
    sim_map = None if not total_sim else {}
    if sim_map is not None:
        for entry in total_sim:
            sim_map[entry['group_id']] = entry['similarity']
    print(f"Aggregated {len(total_sim)} similarity lines into {sim_out}")

    written_acc = aggregate_accuracy(parts_root, acc_out, num_workers=args.num_workers, sim_map = sim_map)
    print(f"Wrote {written_acc} accuracy records to {acc_out}")


if __name__ == "__main__":
    main()


