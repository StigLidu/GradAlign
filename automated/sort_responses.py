#!/usr/bin/env python3

import os
import argparse
import subprocess
import sys

from config import get_response_dir


def main():
    parser = argparse.ArgumentParser(description="Sort responses.json into responses_sorted.json for a dataset/model pair")
    parser.add_argument("--model_name", default='', type=str, help="Model name used for responses directory")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset key")
    parser.add_argument("--responses_dir", default=None, type=str, help="Override responses directory (write in-place)")
    args = parser.parse_args()

    resp_dir = args.responses_dir or get_response_dir(args.dataset, args.model_name)
    raw_path = os.path.join(resp_dir, "responses.json")
    sorted_path = os.path.join(resp_dir, "responses_sorted.json")

    if not os.path.isfile(raw_path):
        if os.path.isfile(sorted_path):
            print(f"Sorted file already exists: {sorted_path}")
            return
        raise SystemExit(f"Raw responses not found: {raw_path}")

    sorter = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sort_jsonl.py"))
    if not os.path.isfile(sorter):
        raise SystemExit(f"Sorter script not found: {sorter}")

    os.makedirs(resp_dir, exist_ok=True)
    cmd = [sys.executable, sorter, raw_path, sorted_path]
    print("Executing:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Sorted responses written to: {sorted_path}")


if __name__ == "__main__":
    main()


