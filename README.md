# GradAlign: Online Data Selection for Scalable LLM Reinforcement Learning via Gradient Alignment

GradAlign is an online data selection method for reinforcement learning (RL) post-training of large language models. It selects high-quality training problems by measuring the alignment between per-problem policy gradients and the aggregated validation gradient.


## Code Structure

```
├── automated/                  # Orchestration and pipeline scripts
│   ├── config.py               # Model paths, dataset directories, model type mappings
│   ├── prepare_data.py         # Download and standardize datasets to VERL format
│   ├── mix.py                  # Mix slices from multiple datasets
│   ├── dynamic_selection.py    # Main selection loop: chunk → infer → analyze → select → train
│   ├── launch_verl_training.py # Launch GRPO training via VERL
│   ├── run_inference_local.py  # Run vLLM inference for rollout generation
│   ├── run_parallel_analysis.py# Dispatch gradient analysis across parts
│   ├── aggregate.py            # Aggregate per-response results to per-problem scores
│   ├── select_data.py          # Select top-N problems by similarity, accuracy, or random
│   ├── merge_model.py          # Merge distributed checkpoints to HuggingFace format
│   ├── convert_megatron_optimizer_to_hf.py  # Convert Megatron optimizer state
│   └── sort_responses.py       # Sort inference responses by group
│
├── select/                     # Core gradient analysis modules
│   ├── grpo_grad_analyze.py    # Single-GPU GRPO gradient analyzer
│   ├── grpo_loss.py            # GRPO loss: advantages, policy loss, KL loss, log-probs
│   ├── grpo_utils.py           # Batching, masking, checkpointing utilities
│   ├── grpo_trainer.py         # GRPO trainer (for reference/baseline training)
│   ├── analyze_grpo.py         # End-to-end analysis driver (single-GPU)
│   ├── analyze_similarity.py   # Cosine similarity computation between gradient vectors
│   ├── inference_vllm.py       # vLLM-based rollout generation
│   ├── inference_ray_batch.py  # Ray-based batch inference
│   ├── inference_client.py     # API-based inference client
│   ├── train_grpo.py           # Standalone GRPO training script
│   ├── utils.py                # Shared utilities
│   └── parallel/               # Multi-GPU parallel gradient analysis
│       ├── launch_parallel_analysis.py     # Launcher for distributed analysis
│       ├── grpo_grad_analyze_parallel.py   # FSDP-based parallel gradient computation
│       ├── analyze_grpo_parallel.py        # Parallel analysis coordinator
│       ├── grpo_loss.py                    # GRPO loss (parallel variant)
│       ├── grpo_utils.py                   # Utilities (parallel variant)
│       └── prepare.py                      # Data preparation for parallel analysis
│
├── verl/                       # Modified VERL framework (RL training backbone)
│   └── verl/utils/reward_score/
│       └── model_reward.py     # Custom reward: rule-based + LLM judge (Qwen-72B)
```

## Supported Selection Modes

| Mode | Flag | Description |
|------|------|-------------|
| **GradAlign (cosine)** | `--mode sim` | Rank by cosine similarity between candidate and validation gradients (default, recommended) |
| **GradAlign (dot product)** | `--mode dot` | Rank by inner product between gradients |
| **AccGreedy** | `--mode accgreedy` | Select problems with pass rates closest to 50% |
| **LearnAlign** | `--mode align` | Gradient similarity within training set (no external validation) |
| **Random** | Please directly use launch_verl_training |

## Setup

### Installation
See https://verl.readthedocs.io/en/latest/start/install.html.

### Configuration

Edit `automated/config.py` to set your environment:

```python
# Map model keys to local/remote paths
MODELS = {
    "qwen2.5-1.5b-math": "/path/to/Qwen2.5-1.5B-Math-Instruct",
    "qwen3-8b-base":     "/path/to/Qwen3-8B-Base",
}

# Base directories for datasets and responses
BASE_DATA_DIR = "/path/to/data"
BASE_RESPONSES_DIR = "/path/to/responses"
```

The reward function in `verl/verl/utils/reward_score/model_reward.py` uses a model judge (Qwen2.5-72B-Instruct by default). Configure the API client at the top of that file.

## Usage

### 1. Prepare Datasets

Download and convert datasets to the standardized format:

```bash
cd automated

# Training data
python prepare_data.py --dataset dapo
python prepare_data.py --dataset webinstruct

# Validation / test data
python prepare_data.py --dataset amc22
python prepare_data.py --dataset amc23
python prepare_data.py --dataset aime
```

Supported datasets: `dapo`, `webinstruct`, `amc22`, `amc23`, `aime`, `math`, `gsm8k`, `mmlupro`, `supergpqa`, `theoremqa`, `deepscaler`, `metamath`, `strategyqa`, `math500`.

### 2. Mix Datasets (optional)

Combine datasets to create training pools:

```bash
cd automated
python mix.py --datasets countdown webinstruct dapo \
              --nums 4000 20000 10000 \
              --ls 0 0 0 \
              --output_name countdown_mixed
```

### 3. Run GradAlign Selection + Training

The `dynamic_selection.py` script orchestrates the full pipeline loop (chunk data, generate rollouts, compute gradient alignment, select top-k, train):

```bash
cd automated
python dynamic_selection.py \
    --prefix gradalign_exp \
    --model qwen3-8b-base \
    --train_dataset webinstruct \
    --val_dataset amc22 \
    --chunk_size 5120 \
    --iters_per_select 10 \
    --max_tokens 3072 \
    --k 20 \
    --mode sim \
    --num_selections 200 \
    --train_batch_size 128 \
    --verl_val_set amc_mc_aime_amc_mmlupro \
    --minibatch_size 8 \
    --reward_path model_reward.py \
    --reward_manager prime \
    --n_samples_train 64
```

Key parameters:
- `--chunk_size`: Number of candidate problems per selection round (M in the paper)
- `--k`: Selection ratio; selects chunk_size/k problems per round
- `--mode`: Selection strategy (`sim`, `dot`, `accgreedy`, `align`, `rand`)
- `--n_samples_train` / `--n_samples_val`: Rollout samples per problem for gradient estimation (k_r and k_v in the paper)
- `--iters_per_select`: Number of GRPO training iterations between selection rounds
- `--num_selections`: Total number of selection rounds

### 4. Random Selection Baseline (No Data Selection)

Equivalent to training on the random dataset:

```bash
cd automated
python launch_verl_training.py \
    --prefix baseline \
    --model qwen3-8b-base \
    --train_dataset webinstruct \
    --val_dataset amc_mc_aime_amc_mmlupro \
    --max_response_length 3072 \
    --save_freq 10 \
    --total_epochs 1000 \
    --train_batch_size 128 \
    --reward_path model_reward.py \
    --reward_manager prime
```

## Citation

```bibtex
@article{gradalign2025,
  title={GradAlign: Online Data Selection for Scalable LLM Reinforcement Learning via Gradient Alignment},
  year={2025}
}
```
