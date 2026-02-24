# GradAlign: Gradient-Aligned Data Selection for LLM Reinforcement Learning

Official implementation for the paper:

>**GradAlign: Gradient-Aligned Data Selection for LLM Reinforcement Learning**  
>Ningyuan Yang*, Weihua Du*, Weiwei Sun, Sean Welleck, Yiming Yang  
>Preprint

GradAlign is an online RL data-selection method that ranks candidate training problems by alignment between:
- candidate GRPO gradients and
- an aggregated gradient from a small trusted validation set.

The method is designed for non-stationary LLM RL and targets three practical regimes from the paper:
- unreliable reward signals,
- distribution imbalance, and
- low-utility large-scale web corpora.

## Repository layout

```text
.
|-- automated/                   # End-to-end orchestration scripts
|   |-- prepare_data.py          # Dataset download + conversion to VERL-style JSONL/Parquet
|   |-- dynamic_selection.py     # Main loop: infer -> analyze -> aggregate -> select -> train
|   |-- run_inference_local.py   # vLLM/Ray rollout generation
|   |-- run_parallel_analysis.py # Distributed gradient alignment driver
|   |-- aggregate.py             # Per-problem similarity/accuracy aggregation
|   |-- select_data.py           # Selection policies (sim, dot, accgreedy, align, rand, ...)
|   |-- launch_verl_training.py  # GRPO training launcher (VERL backend)
|   |-- merge_model.py           # Merge distributed checkpoints to HF format
|   `-- config.py                # Local model/data path config
|-- select/                      # Gradient/similarity analysis implementation
|-- verl/                        # Modified VERL training framework
`-- scripts/                     # Utility scripts
```

## Selection modes

| Paper name | `--mode` | Description |
|---|---|---|
| GradAlign (cosine) | `sim` | Cosine similarity between candidate and validation gradients (default in paper) |
| GradAlign (dot) | `dot` | Dot product similarity |
| LearnAlign | `align` | Gradient alignment without external validation set |
| AccGreedy | `accgreedy` | Prioritize problems with pass rate closest to 50% |
| Random | `rand` | Uniform random selection |

## 1) Environment setup

Follow VERL installation instructions: https://verl.readthedocs.io/en/latest/start/install.html

Typical local setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r verl/requirements.txt
pip install -e ./verl
```

Notes:
- `select/inference_ray_batch.py` requires `ray>=2.44.1`.
- Multi-GPU analysis/training depends on your local CUDA, NCCL, and vLLM stack.

## 2) Configuration

### 2.1 Edit `automated/config.py`

Set model aliases and data roots:

```python
MODELS = {
    "qwen2.5-1.5b-math": "/path/to/Qwen2.5-1.5B-Math-Instruct",
    "qwen3-8b-base": "/path/to/Qwen3-8B-Base",
}

BASE_DATA_DIR = "/path/to/data"
BASE_RESPONSES_DIR = "/path/to/responses"
```

### 2.2 Configure the model-judge reward

`verl/verl/utils/reward_score/model_reward.py` contains the async model-judge path used by this codebase. Configure your API client/service there before running experiments that enable model judging.

### 2.3 Important path assumption

`automated/launch_verl_training.py` currently assumes VERL config/reward files under `~/data_selection/...` by default.  
If your repo path differs, update that script or pass matching paths/overrides before launching large runs.

## 3) Data preparation

Prepare datasets to unified VERL-style records (`train.jsonl`, `train.parquet`, `val.jsonl`, `val.parquet`):

```bash
cd automated
python prepare_data.py --dataset dapo
python prepare_data.py --dataset webinstruct
python prepare_data.py --dataset amc22
python prepare_data.py --dataset amc23
python prepare_data.py --dataset aime
```

Supported dataset keys in `prepare_data.py` include:
`aime24`, `aime25`, `aime`, `math`, `metamath`, `strategyqa`, `webinstruct`, `deepscaler`, `mmlupro`, `gsm8k`, `theoremqa`, `supergpqa`, `amc22`, `amc23`, `math500`, `dapo`.

Optional dataset mixing:

```bash
cd automated
python mix.py \
  --datasets dapo webinstruct \
  --nums 10000 20000 \
  --ls 0 0 \
  --output_name dapo_webinstruct_mix
```

## 4) Run GradAlign end-to-end

`dynamic_selection.py` orchestrates:
1. chunk candidate data,
2. rollout generation,
3. gradient alignment analysis,
4. top-k selection,
5. GRPO training on selected subsets.

Example:

```bash
cd automated
python dynamic_selection.py \
  --prefix gradalign_exp \
  --model qwen3-8b-base \
  --train_dataset webinstruct \
  --val_dataset amc22 \
  --chunk_size 5120 \
  --k 20 \
  --mode sim \
  --num_selections 100 \
  --train_batch_size 128 \
  --iters_per_select 10 \
  --n_samples_train 128 \
  --n_samples_val 16 \
  --minibatch_size 8 \
  --max_tokens 3072 \
  --max_model_len 4096 \
  --ckpt_root /path/to/checkpoints \
  --verl_val_set amc22
```

Important arguments:
- `--chunk_size`: number of candidate problems per round (`M` in paper).
- `--k`: selection ratio (`q` in paper), selecting `chunk_size / k` each round.
- `--mode`: selection policy (`sim`, `dot`, `accgreedy`, `align`, `rand`, ...).
- `--n_samples_train`, `--n_samples_val`: rollout counts used in gradient estimation (`k_r`, `k_v`).
- `--iters_per_select`: GRPO update steps between selection rounds.
- `--num_selections`: number of selection rounds.

Constraint enforced by code:
- `(k * train_batch_size * iters_per_select) % chunk_size == 0`

## 5) Baselines

Random baseline through the same loop:

```bash
cd automated
python dynamic_selection.py \
  --prefix random_baseline \
  --model qwen3-8b-base \
  --train_dataset webinstruct \
  --val_dataset amc22 \
  --chunk_size 5120 \
  --k 20 \
  --mode rand \
  --num_selections 100 \
  --train_batch_size 128 \
  --iters_per_select 10 \
  --ckpt_root /path/to/checkpoints \
  --verl_val_set amc22
```

Direct GRPO training without data selection:

```bash
cd automated
python launch_verl_training.py \
  --prefix grpo_baseline \
  --model qwen3-8b-base \
  --train_dataset webinstruct \
  --val_dataset amc22 \
  --train_batch_size 128 \
  --max_response_length 3072 \
  --total_epochs 1000
```

## 6) Export merged model

`merge_model.py` merges checkpoints from:
`<ckpt_root>/<experiment_name>/global_step_<step>/actor`

Manual merge example:

```bash
cd automated
python merge_model.py \
  --experiment_name <experiment_name> \
  --step <global_step> \
  --output_model_name <merged_name> \
  --backend megatron \
  --ckpt_root /path/to/checkpoints \
  --dest_root /path/to/merged_models
```

## 7) Output artifacts

Typical artifacts under `ckpt_root/<experiment_name>/`:
- `chunks/chunk_*/train.jsonl` and `chunks/chunk_*/train.parquet`: chunked candidate pools.
- `global_step_*/train_split/part_0/responses.json` and `responses_sorted.json`: train rollouts.
- `global_step_*/train_split/part_0/similarity_results_cosine_real.jsonl`: per-response similarity (parallel analysis output).
- `global_step_*/train_split/similarity_results_aggregated.jsonl`: per-problem aggregated similarity.
- `global_step_*/train_split/accuracy_by_problem.jsonl`: per-problem accuracy (and similarity when available).
- `global_step_*/val_responses/responses.json` and `responses_sorted.json`: validation rollouts (for similarity-based modes).
- `global_step_*/merged/`: merged model used by the next selection round (dynamic selection runs).
- `selected/iter_<i>_<mode>_<n>/train.jsonl` and `train.parquet`: selected training subset for that iteration.
- `global_step_*/actor/` (from VERL): training checkpoint shards used as merge input.

## Citation

```bibtex
Coming soon!
```

## Acknowledgment

This repository is built on top of VERL (`verl/`) and includes project-specific modifications for GradAlign data selection and analysis.
