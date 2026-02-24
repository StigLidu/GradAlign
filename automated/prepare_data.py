#!/usr/bin/env python3

import os
import argparse
from typing import List, Tuple, Optional, Any, cast

import datasets

from config import REWARD_STYLE, get_dataset_dir
from utils import extract_answer


def _build_user_prompt(problem_text: str, dataset_key: str) -> str:
    """Return the instruction + problem text for the user message."""
    if dataset_key.lower() in {"strategyqa", "strategy-qa"}:
        return (
            "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
            f"{problem_text}. Your answer should be True/False."
        )
    return (
        "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
        f"{problem_text}"
    )


# No chat template application; we pass plain chat messages to 'prompt'


def _standardize_aime24(ds: datasets.Dataset) -> datasets.Dataset:
    def transform(example):
        return {
            "id": str(example["id"]),
            "problem": example["problem"],
            "answer": str(extract_answer(example["solution"]))
        }

    target = datasets.Features(
        {
            "id": datasets.Value("string"),
            "problem": datasets.Value("string"),
            "answer": datasets.Value("string"),
        }
    )
    ds = ds.map(transform, remove_columns=ds.column_names)
    return ds.cast(target)


def _standardize_aime25(ds: datasets.Dataset) -> datasets.Dataset:
    def transform(example):
        return {
            "id": str(example["id"]),
            "problem": example["problem"],
            "answer": str(example["answer"]),
        }

    target = datasets.Features(
        {
            "id": datasets.Value("string"),
            "problem": datasets.Value("string"),
            "answer": datasets.Value("string"),
        }
    )
    ds = ds.map(transform, remove_columns=ds.column_names)
    return ds.cast(target)


def _load_and_standardize(dataset_key: str, max_samples: Optional[int] = None) -> Tuple[Any, str]:
    key = dataset_key.lower()

    if key in {"aime24", "aime-24"}:
        ds = cast(datasets.Dataset, datasets.load_dataset("math-ai/aime24", split="test"))
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        return _standardize_aime24(ds), "aime24"

    if key in {"aime25", "aime-25"}:
        ds = cast(datasets.Dataset, datasets.load_dataset("math-ai/aime25", split="test"))
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        return _standardize_aime25(ds), "aime25"

    if key in {"aime", "aime24_25", "aime-merged"}:
        raw24 = cast(datasets.Dataset, datasets.load_dataset("math-ai/aime24", split="test"))
        raw25 = cast(datasets.Dataset, datasets.load_dataset("math-ai/aime25", split="test"))
        if max_samples is not None:
            # Take from AIME24 first, then spill into AIME25
            take24 = min(max_samples, len(raw24))
            raw24 = raw24.select(range(take24))
            remaining = max_samples - take24
            if remaining > 0:
                raw25 = raw25.select(range(min(remaining, len(raw25))))
            else:
                raw25 = raw25.select(range(0))
        ds24 = _standardize_aime24(raw24)
        ds25 = _standardize_aime25(raw25)
        return datasets.concatenate_datasets([ds24, ds25]), "aime"

    if key in {"math", "hendrycks_math"}:
        subjects: List[str] = [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ]

        parts = []
        remaining = max_samples if max_samples is not None else None
        for subject in subjects:
            print(subject)
            part = cast(datasets.Dataset, datasets.load_dataset("EleutherAI/hendrycks_math", subject, split="train"))
            if remaining is not None:
                if remaining <= 0:
                    break
                take = min(remaining, len(part))
                part = part.select(range(take))

            def to_std(example):
                return {
                    "id": str(example.get("problem_id", "")),
                    "problem": example["problem"],
                    "answer": str(extract_answer(example["solution"]))
                }

            part = part.map(to_std, remove_columns=list(part.column_names))  # type: ignore[arg-type]
            target = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "problem": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                }
            )
            part = part.cast(target)
            parts.append(part)
            if remaining is not None:
                remaining -= len(part)
        return datasets.concatenate_datasets(parts) if parts else datasets.Dataset.from_list([]), "math"

    if key in {"metamath", "meta-math"}:
        ds = cast(datasets.Dataset, datasets.load_dataset("meta-math/MetaMathQA", split="train"))
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        def to_std(example):
            # The dataset's answer is inside the response text; keep the final segment
            answer = str(example["response"]).split("The answer is: ")[-1]
            return {
                "id": str(example.get("id", "")),
                "problem": example["query"],
                "answer": answer,
            }

        mapped = ds.map(to_std, remove_columns=list(ds.column_names))  # type: ignore[arg-type]
        target = datasets.Features(
            {
                "id": datasets.Value("string"),
                "problem": datasets.Value("string"),
                "answer": datasets.Value("string"),
            }
        )
        return mapped.cast(target), "metamath"

    if key in {"strategyqa", "strategy-qa"}:
        ds = cast(datasets.Dataset, datasets.load_dataset("tasksource/strategy-qa", split="train"))
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        def to_std(example):
            return {
                "id": str(example.get("id", "")),
                "problem": example["question"],
                "answer": str(example["answer"]),
            }

        mapped = ds.map(to_std, remove_columns=list(ds.column_names))  # type: ignore[arg-type]
        target = datasets.Features(
            {
                "id": datasets.Value("string"),
                "problem": datasets.Value("string"),
                "answer": datasets.Value("string"),
            }
        )
        return mapped.cast(target), "strategyqa"

    if key in {"webinstruct"}:
        ds = cast(datasets.Dataset, datasets.load_dataset("TIGER-Lab/WebInstruct-verified-unfiltered", split="train"))
        # ds = ds.filter(lambda example: example["answer_type"] == "Multiple Choice")
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        def to_std(example):
            return {
                "id": str(example.get("task_id", "")),
                "problem": example["original_question"],
                "answer": str(example["short_answer"]),
            }

        mapped = ds.map(to_std, remove_columns=list(ds.column_names))  # type: ignore[arg-type]
        target = datasets.Features(
            {
                "id": datasets.Value("string"),
                "problem": datasets.Value("string"),
                "answer": datasets.Value("string"),
            }
        )
        return mapped.cast(target), "webinstruct"

    if key in {"deepscaler"}:
        ds = cast(datasets.Dataset, datasets.load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train"))
        # ds = ds.filter(lambda example: example["answer_type"] == "Multiple Choice")
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        def to_std(example):
            return {
                "id": str(example.get("task_id", "")),
                "problem": example["problem"],
                "answer": str(example["answer"]),
            }

        mapped = ds.map(to_std, remove_columns=list(ds.column_names))  # type: ignore[arg-type]
        target = datasets.Features(
            {
                "id": datasets.Value("string"),
                "problem": datasets.Value("string"),
                "answer": datasets.Value("string"),
            }
        )
        return mapped.cast(target), "deepscaler"

    if key in {"mmlupro", "mmlu-pro", "mmlu_pro", "mmlupro-math"}:
        # Load official splits and map to our unified schema, appending options to the question.
        raw_test = cast(datasets.Dataset, datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="test"))
        raw_val = cast(datasets.Dataset, datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="validation"))
        if key == "mmlupro-math":
            raw_test = raw_test.filter(lambda example: example["category"] == "math")
            raw_val = raw_val.filter(lambda example: example["category"] == "math")

        # Respect max_samples by taking from test first, then spill into validation
        if max_samples is not None:
            take_test = min(max_samples, len(raw_test))
            raw_test = raw_test.select(range(take_test))
            remaining = max_samples - take_test
            if remaining > 0:
                raw_val = raw_val.select(range(min(remaining, len(raw_val))))
            else:
                raw_val = raw_val.select(range(0))

        def to_std_mmlu(split_label: str):
            def _inner(example):
                q = str(example["question"]).strip()
                options_list = example.get("options", [])
                try:
                    option_lines = [f"{chr(65 + i)}. {str(opt).strip()}" for i, opt in enumerate(options_list)]
                except Exception:
                    option_lines = []
                if option_lines:
                    problem = (
                        q
                        + "\n"
                        + "\n".join(option_lines)
                        + "\nChoose the correct option by its alphabet. You answer should be \\boxed{A/B/C/...}\\n"
                    )
                else:
                    problem = q
                # Prefer 'answer' letter if present; fall back to index mapping
                answer_value = str(example.get("answer", "")).strip()
                if not answer_value:
                    idx = example.get("answer_index", None)
                    if isinstance(idx, int) and 0 <= idx < 26:
                        answer_value = chr(65 + idx)
                return {
                    "id": str(example.get("question_id", "")),
                    "problem": problem,
                    "answer": answer_value,
                    "split": split_label,
                }
            return _inner

        test_std = raw_test.map(to_std_mmlu("train"), remove_columns=list(raw_test.column_names))  # type: ignore[arg-type]
        val_std = raw_val.map(to_std_mmlu("val"), remove_columns=list(raw_val.column_names))  # type: ignore[arg-type]

        target = datasets.Features(
            {
                "id": datasets.Value("string"),
                "problem": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "split": datasets.Value("string"),
            }
        )
        merged = datasets.concatenate_datasets([test_std.cast(target)])
        return merged, key

    if key in {"gsm8k", "gsm-8k"}:
        ds = cast(datasets.Dataset, datasets.load_dataset("openai/gsm8k", "main", split="train"))
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        def to_std(example):
            return {
                "id": str(example.get("id", "")),
                "problem": example["question"],
                "answer": str(extract_answer(example["answer"]))
            }

        mapped = ds.map(to_std, remove_columns=list(ds.column_names))  # type: ignore[arg-type]
        target = datasets.Features(
            {
                "id": datasets.Value("string"),
                "problem": datasets.Value("string"),
                "answer": datasets.Value("string"),
            }
        )
        return mapped.cast(target), "gsm8k"

    if key in {"theoremqa", "theoremqa_train", "theoremqa_test"}:
        ds = cast(datasets.Dataset, datasets.load_dataset("TIGER-Lab/TheoremQA", split="test"))
        ds = ds.shuffle(seed=42)
        if "train" in key:
            ds = ds.select(range(int(len(ds) * 0.5)))
        elif "test" in key:
            ds = ds.select(range(int(len(ds) * 0.5), len(ds)))

        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        def to_std(example):
            return {
                "id": str(example.get("id", "")),
                "problem": example["Question"],
                "answer": str(example["Answer"])
            }

        mapped = ds.map(to_std, remove_columns=list(ds.column_names))  # type: ignore[arg-type]
        target = datasets.Features(
            {
                "id": datasets.Value("string"),
                "problem": datasets.Value("string"),
                "answer": datasets.Value("string"),
            }
        )
        return mapped.cast(target), key

    if key in {"supergpqa", "supergpqa_train", "supergpqa_test"}:
        # Load official splits and map to our unified schema, appending options to the question.
        raw_train = cast(datasets.Dataset, datasets.load_dataset("m-a-p/SuperGPQA", split="train"))

        raw_train = raw_train.shuffle(seed=42)
        if "train" in key:
            raw_train = raw_train.select(range(int(len(raw_train) * 0.5)))
        elif "test" in key:
            raw_train = raw_train.select(range(int(len(raw_train) * 0.5), len(raw_train)))

        # Respect max_samples by taking from test first, then spill into validation
        if max_samples is not None:
            take_train = min(max_samples, len(raw_train))
            raw_train = raw_train.select(range(take_train))

        def to_std_supergpqa(split_label: str):
            def _inner(example):
                q = str(example["question"]).strip()
                options_list = example.get("options", [])
                try:
                    option_lines = [f"{chr(65 + i)}. {str(opt).strip()}" for i, opt in enumerate(options_list)]
                except Exception:
                    option_lines = []
                if option_lines:
                    problem = (
                        q
                        + "\n"
                        + "\n".join(option_lines)
                        + "\nChoose the correct option by its alphabet. You answer should be \\boxed{A/B/C/...}\\n"
                    )
                else:
                    problem = q
                # Prefer 'answer' letter if present; fall back to index mapping
                answer_value = str(example.get("answer_letter", "")).strip()
                if not answer_value:
                    idx = example.get("answer_index", None)
                    if isinstance(idx, int) and 0 <= idx < 26:
                        answer_value = chr(65 + idx)
                return {
                    "id": str(example.get("question_id", "")),
                    "problem": problem,
                    "answer": answer_value,
                    "split": split_label,
                }
            return _inner

        train_std = raw_train.map(to_std_supergpqa("train"), remove_columns=list(raw_train.column_names))  # type: ignore[arg-type]
        # val_std = raw_val.map(to_std_mmlu("val"), remove_columns=list(raw_val.column_names))  # type: ignore[arg-type]

        target = datasets.Features(
            {
                "id": datasets.Value("string"),
                "problem": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "split": datasets.Value("string"),
            }
        )
        merged = train_std.cast(target)
        return merged, key

    if key in {"gsm8k", "gsm-8k"}:
        ds = cast(datasets.Dataset, datasets.load_dataset("openai/gsm8k", "main", split="train"))
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        def to_std(example):
            return {
                "id": str(example.get("id", "")),
                "problem": example["question"],
                "answer": str(extract_answer(example["answer"]))
            }

        mapped = ds.map(to_std, remove_columns=list(ds.column_names))  # type: ignore[arg-type]
        target = datasets.Features(
            {
                "id": datasets.Value("string"),
                "problem": datasets.Value("string"),
                "answer": datasets.Value("string"),
            }
        )
        return mapped.cast(target), "gsm8k"

    if key in {"amc", 'amc22', 'amc23'}:
        ds = cast(datasets.Dataset, datasets.load_dataset("AI-MO/aimo-validation-amc", split="train"))
        if key == 'amc22':
            ds = ds.filter(lambda example: "2022_AMC" in example["url"])
        elif key == 'amc23':
            ds = ds.filter(lambda example: "2023_AMC" in example["url"])
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        def to_std(example):
            return {
                "id": str(example.get("id", "")),
                "problem": example["problem"],
                "answer": str(int(example["answer"]))
            }

        # Explicitly set output features to enforce string type for 'answer'
        target = datasets.Features(
            {
                "id": datasets.Value("string"),
                "problem": datasets.Value("string"),
                "answer": datasets.Value("string"),
            }
        )
        mapped = ds.map(to_std, remove_columns=list(ds.column_names))  # type: ignore[arg-type]
        return mapped.cast(target), key

    if key in {"math500"}:
        ds = cast(datasets.Dataset, datasets.load_dataset("HuggingFaceH4/MATH-500", split="test"))
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        def to_std(example):
            return {
                "id": str(example.get("id", "")),
                "problem": example["problem"],
                "answer": str(int(example["answer"]))
            }

        # Explicitly set output features to enforce string type for 'answer'
        target = datasets.Features(
            {
                "id": datasets.Value("string"),
                "problem": datasets.Value("string"),
                "answer": datasets.Value("string"),
            }
        )
        mapped = ds.map(to_std, remove_columns=list(ds.column_names))  # type: ignore[arg-type]
        return mapped.cast(target), "amc"

    if key in {"dapo", "dapo-math-17k"}:
        raw = cast(datasets.Dataset, datasets.load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", "default", split="train"))
        if max_samples is not None:
            raw = raw.select(range(min(max_samples, len(raw))))

        # DAPO already has prompt and reward_model fields. We will standardize to our schema.
        def to_std(example):
            # Example has prompt as a list of chat messages; extract the user content.
            # We keep the original user message as problem text to avoid double wrapping.
            prompt_list = example.get("prompt", [])
            user_text = ""
            if isinstance(prompt_list, list) and prompt_list:
                first = prompt_list[0]
                user_text = first.get("content", "") if isinstance(first, dict) else ""
            return {
                "id": str(example.get("extra_info", {}).get("index", "")),
                "problem": user_text.split('is the answer to the problem.\n\n')[-1].split('\nRemember to put your answer on its own line after \"Answer:\".')[0],
                "answer": str(example.get("reward_model", {}).get("ground_truth", "")),
            }

        mapped = raw.map(to_std, remove_columns=list(raw.column_names))  # type: ignore[arg-type]
        target = datasets.Features(
            {
                "id": datasets.Value("string"),
                "problem": datasets.Value("string"),
                "answer": datasets.Value("string"),
            }
        )
        return mapped.cast(target), "dapo"
        

    known = [
        "aime24", "aime25", "aime",
        "math", "metamath", "strategyqa", "gsm8k", "amc", "dapo", "mmlupro", "webinstruct",
    ]
    raise ValueError(f"Unsupported dataset '{dataset_key}'. Choose one of: {', '.join(known)}")


def _process_to_verl(ds: datasets.Dataset, dataset_label: str) -> datasets.Dataset:
    def process_fn(example, idx):
        user_message = _build_user_prompt(example["problem"], dataset_label)
        # Build plain chat messages
        messages = []
        # if "qwen" in model_key.lower():
        #     messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append({"role": "user", "content": user_message})
        # Propagate split if present; default to 'train'
        split_value = str(example.get("split", "train"))
        return {
            "data_source": dataset_label,
            "prompt": messages,
            "reward_model": {
                "ground_truth": str(example["answer"]),
                "style": REWARD_STYLE,
            },
            "extra_info": {
                "split": split_value,
                "index": idx,
                "original_index": idx,
            },
        }

    return ds.map(
        function=process_fn,
        with_indices=True,
        remove_columns=ds.column_names,
    )


def _write_outputs(processed: datasets.Dataset, output_dir: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    # Partition by split when available; otherwise fall back to the whole set for both
    try:
        train_data = processed.filter(lambda ex: ex["extra_info"]["split"] == "train")
        val_data = processed.filter(lambda ex: ex["extra_info"]["split"] == "val")
        if len(train_data) == 0:
            train_data = processed
        if len(val_data) == 0:
            val_data = processed
    except Exception:
        train_data = processed
        val_data = processed

    train_file = os.path.join(output_dir, "train.parquet")
    train_data.to_parquet(train_file)

    train_jsonl = os.path.join(output_dir, "train.jsonl")
    with open(train_jsonl, "w", encoding="utf-8") as f:
        for ex in train_data:
            import json
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")

    val_file = os.path.join(output_dir, "val.parquet")
    val_data.to_parquet(val_file)

    val_jsonl = os.path.join(output_dir, "val.jsonl")
    with open(val_jsonl, "w", encoding="utf-8") as f:
        for ex in val_data:
            import json
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")

    return train_file, val_file


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Unified dataset downloader/converter. Produces VERL-like train/val parquet and JSONL."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Which dataset to prepare: one of {aime24, aime25, aime, math, metamath, strategyqa, gsm8k, mmlupro}"
        ),
    )
    # parser.add_argument(
    #     "--model",
    #     type=str,
    #     required=True,
    #     help=(
    #         "Model key for tokenizer chat template (see config.MODELS)."
    #     ),
    # )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory to write outputs. If omitted, it becomes "
            "'~/data_selection/data/huggingface/{dataset}_{model}'."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=40000,
        help="Optionally limit number of samples for quick tests",
    )

    args = parser.parse_args()

    base_ds, dataset_label = _load_and_standardize(args.dataset, args.max_samples)

    processed = _process_to_verl(base_ds, dataset_label)
    output_dir = args.output_dir or get_dataset_dir(dataset_label)
    train_file, val_file = _write_outputs(processed, output_dir)

    print("\nData preprocessing complete!")
    print(f"Training data: {train_file}")
    print(f"Validation data: {val_file}")


if __name__ == "__main__":
    main()


