#!/usr/bin/env python3

import os
import json
import argparse
import random
from typing import List, Set, Dict, Tuple

import datasets as hfds  # type: ignore[import-not-found]

from config import get_dataset_dir, get_response_dir


def _read_topn_group_ids(sim_jsonl_path: str, top_n: int, neg: bool = False) -> List[int]:
    pairs: List[Tuple[int, float]] = []
    with open(sim_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if "group_id" in obj and "similarity" in obj:
                try:
                    gid = int(obj["group_id"])
                    sim = float(obj["similarity"])
                except (TypeError, ValueError):
                    continue
                pairs.append((gid, sim))
    pairs.sort(key=lambda x: x[1], reverse=not neg)
    # print('n', top_n)
    print([x[1] for x in pairs[top_n-10:top_n]])
    return [gid for gid, _ in pairs[:top_n]]


def _read_similarity_map(sim_jsonl_path: str) -> Dict[int, float]:
    sim_map: Dict[int, float] = {}
    with open(sim_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if "group_id" in obj and "similarity" in obj:
                try:
                    gid = int(obj["group_id"])
                    sim = float(obj["similarity"])
                except (TypeError, ValueError):
                    continue
                sim_map[gid] = sim
    return sim_map


def _read_acc_map(acc_jsonl_path: str) -> Dict[int, float]:
    acc: Dict[int, float] = {}
    with open(acc_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if "group_id" in obj and "accuracy" in obj:
                try:
                    gid = int(obj["group_id"])
                    a = float(obj["accuracy"])
                except (TypeError, ValueError):
                    continue
                acc[gid] = a
    return acc


def _write_selected(dataset_train_path: str, selected_ids: Set[int], out_dir: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, "train.jsonl")
    selected: List[Dict] = []
    with open(dataset_train_path, "r", encoding="utf-8") as f_in, open(out_jsonl, "w", encoding="utf-8") as f_out:
        for line in f_in:
            obj = json.loads(line)
            idx = obj['extra_info']['index']
            if idx in selected_ids:
                selected.append(obj)
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    out_parquet = out_jsonl.replace(".jsonl", ".parquet")
    if selected:
        hfds.Dataset.from_list(selected).to_parquet(out_parquet)
    return out_jsonl, out_parquet


def select_by_similarity(dataset_dir: str, parts_root: str, top_n: int, output_dir: str, filename: str, neg: bool = False) -> None:
    sim_path = os.path.join(parts_root, filename)
    assert os.path.isfile(sim_path), f"Similarity file not found: {sim_path}"
    gids = _read_topn_group_ids(sim_path, top_n, neg)
    selected_ids = set(gids)
    dataset_train = os.path.join(dataset_dir, "train.jsonl")
    print('Selected', len(selected_ids))
    _write_selected(dataset_train, selected_ids, output_dir)


def select_by_accuracy(dataset_dir: str, parts_root: str, n: int, acc_low: float, acc_high: float, output_dir: str, lim: int | None = None) -> None:
    acc_path = os.path.join(parts_root, "accuracy_by_problem.jsonl")
    assert os.path.isfile(acc_path), f"Accuracy file not found: {acc_path}"
    acc_map = _read_acc_map(acc_path)
    # print(acc_map)
    pool = [gid for gid, a in acc_map.items() if acc_low <= a <= acc_high and (lim is None or gid < int(lim))]
    if not pool:
        raise SystemExit("No group_ids satisfy the accuracy constraints")
    if n > len(pool):
        print(f"Warning: requested {n} but only {len(pool)} available; selecting all.")
        n = len(pool)
    selected_ids = set(random.sample(pool, n))
    dataset_train = os.path.join(dataset_dir, "train.jsonl")
    print('Selected', len(selected_ids))
    _write_selected(dataset_train, selected_ids, output_dir)


def select_by_simacc(dataset_dir: str, parts_root: str, n: int, output_dir: str, filename: str) -> None:
    sim_path = os.path.join(parts_root, filename)
    acc_path = os.path.join(parts_root, "accuracy_by_problem.jsonl")
    assert os.path.isfile(sim_path), f"Similarity file not found: {sim_path}"
    assert os.path.isfile(acc_path), f"Accuracy file not found: {acc_path}"
    sim_map = _read_similarity_map(sim_path)
    acc_map = _read_acc_map(acc_path)
    pairs: List[Tuple[int, float, float, float]] = []
    for gid, sim in sim_map.items():
        if gid not in acc_map:
            continue
        a = acc_map[gid]
        score = sim * (a * (1.0 - a)) * 4
        pairs.append((gid, score, sim, a))
    if not pairs:
        raise SystemExit("No overlapping group_ids between similarity and accuracy inputs")
    pairs.sort(key=lambda x: x[1], reverse=True)
    if n > len(pairs):
        print(f"Warning: requested {n} but only {len(pairs)} available; selecting all.")
        n = len(pairs)
    print(pairs[n-10:n])
    selected_ids = set(gid for gid, _, _, _ in pairs[:n])
    dataset_train = os.path.join(dataset_dir, "train.jsonl")
    print('Selected', len(selected_ids))
    _write_selected(dataset_train, selected_ids, output_dir)


def select_by_accgreedy(dataset_dir: str, parts_root: str, n: int, output_dir: str) -> None:
    acc_path = os.path.join(parts_root, "accuracy_by_problem.jsonl")
    assert os.path.isfile(acc_path), f"Accuracy file not found: {acc_path}"
    acc_map = _read_acc_map(acc_path)
    if not acc_map:
        raise SystemExit("No accuracy records found for accgreedy selection")
    # Sort by closeness to 0.5
    ordered = sorted(acc_map.items(), key=lambda kv: abs(kv[1] - 0.5))
    if n > len(ordered):
        print(f"Warning: requested {n} but only {len(ordered)} available; selecting all.")
        n = len(ordered)
    selected_ids = set(gid for gid, _ in ordered[:n])
    dataset_train = os.path.join(dataset_dir, "train.jsonl")
    print('Selected', len(selected_ids))
    _write_selected(dataset_train, selected_ids, output_dir)


def select_random(dataset_dir: str, n: int, output_dir: str, max_num: int | None = None) -> None:
    # Count lines up to max_num (if provided)
    total = 0
    with open(os.path.join(dataset_dir, "train.jsonl"), "r", encoding="utf-8") as f:
        for _ in f:
            total += 1
            if max_num is not None and total >= int(max_num):
                break
    if n > total:
        print(f"Warning: requested {n} but only {total} available; selecting all.")
        n = total
    selected_indices = set(sorted(random.sample(range(total), n)))
    print('Selected', len(selected_indices))
    _write_selected(os.path.join(dataset_dir, "train.jsonl"), selected_indices, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Unified data selector by similarity/accuracy/random")
    parser.add_argument("--mode", required=True, choices=["sim", "acc", "rand", 'simacc', 'accgreedy', 'align', 'negsim'], help="Selection mode")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset key")
    parser.add_argument("--model", required=True, type=str, help="Tokenizer/model key used for dataset preparation")
    parser.add_argument("--n", type=int, default=10000, help="Number of items to select (used in all modes, replaces --top_n)")
    parser.add_argument("--acc_low", type=float, default=0.2)
    parser.add_argument("--acc_high", type=float, default=0.8)
    parser.add_argument("--lim", type=int, default=None, help="Optional limit on candidates (group_id upper bound in acc; max lines in rand)")
    parser.add_argument("--similarity_filename", type=str, default="similarity_results_aggregated.jsonl")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional explicit output directory")
    parser.add_argument("--parts_root", type=str, default=None, help="Override responses split root (compat with dynamic selection)")

    args = parser.parse_args()

    dataset_dir = get_dataset_dir(args.dataset, args.model)
    if args.output_dir is None:
        # Place selections under dataset_dir/selected/<mode-specific>
        selected_root = os.path.join(dataset_dir, "selected", args.model)
        if args.mode == "sim":
            subdir = f"sim_{args.n}"
        elif args.mode == "align":
            subdir = f"align_{args.n}"
        elif args.mode == "negsim":
            subdir = f"negsim_{args.n}"
        elif args.mode == "simacc":
            subdir = f"simacc_{args.n}"
        elif args.mode == "acc":
            subdir = f"acc_{args.acc_low}_{args.acc_high}_{args.n}"
        elif args.mode == "accgreedy":
            subdir = f"accgreedy_{args.n}"
        else:
            subdir = f"random_{args.n}"
        output_dir = os.path.join(selected_root, subdir)
    else:
        output_dir = args.output_dir

    parts_root = None
    if args.mode in {"sim", "acc", "simacc", "accgreedy", "align", "negsim"}:
        if not args.model:
            raise SystemExit("--model is required for sim/acc modes to locate responses")
        parts_root = args.parts_root or (get_response_dir(args.dataset, args.model) + "_split")
        if not os.path.isdir(parts_root):
            raise SystemExit(f"Parts root not found: {parts_root}")

    if args.mode == "sim" or args.mode == "align" or args.mode == "negsim":
        assert parts_root is not None
        select_by_similarity(dataset_dir, parts_root, int(args.n), output_dir, args.similarity_filename, neg=args.mode == "negsim")
    elif args.mode == "simacc":
        assert parts_root is not None
        select_by_simacc(dataset_dir, parts_root, int(args.n), output_dir, args.similarity_filename)
    elif args.mode == "acc":
        assert parts_root is not None
        select_by_accuracy(dataset_dir, parts_root, int(args.n), float(args.acc_low), float(args.acc_high), output_dir, lim=args.lim)
    elif args.mode == "accgreedy":
        assert parts_root is not None
        select_by_accgreedy(dataset_dir, parts_root, int(args.n), output_dir)
    else:
        select_random(dataset_dir, int(args.n), output_dir, max_num=args.lim)

    print(f"Selection complete. Output: {output_dir}")


if __name__ == "__main__":
    main()


