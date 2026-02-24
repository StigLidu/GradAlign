#!/usr/bin/env python3

import os
import argparse
import subprocess
import sys

from config import get_model_path, get_dataset_dir


def build_exp_name(prefix: str, model_name: str, train_dataset, train_dir: str, val_dir: str) -> str:
    train_name = os.path.basename(os.path.normpath(train_dir))
    val_name = os.path.basename(os.path.normpath(val_dir))
    return f"{prefix}_{model_name}_{train_dataset}_{train_name}_{val_name}"


def main():
    parser = argparse.ArgumentParser(description="Launch VERL PPO training with standardized naming")
    parser.add_argument("--prefix", required=True, type=str, help="Experiment name prefix")
    parser.add_argument("--model", required=True, type=str, help="Tokenizer/model key (used to resolve model path and dataset dirs)")
    parser.add_argument("--train_dataset", required=True, type=str, help="Training dataset key (used to resolve dataset dir)")
    parser.add_argument("--val_dataset", required=True, type=str, help="Validation dataset key (used to resolve dataset dir)")

    # Optional explicit overrides
    parser.add_argument("--model_path", default=None, type=str, help="Override base model path to fine-tune")
    parser.add_argument("--train_dir", default=None, type=str, help="Override directory containing train.parquet for training")
    parser.add_argument("--val_dir", default=None, type=str, help="Override directory containing train/val parquet for validation")

    # Optional overrides / settings
    parser.add_argument("--project_name", default="Dynamic-restart", type=str)
    parser.add_argument("--ckpts_root", default="~/data_selection/data/chkpt", type=str)
    parser.add_argument("--train_parquet", default="train.parquet", type=str, help="Filename of training parquet inside train_dir")
    parser.add_argument("--val_parquet", default="train.parquet", type=str, help="Filename of validation parquet inside val_dir")

    # Core PPO settings (defaults mirror run_mixed_filtered.sh)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_response_length", type=int, default=4096)
    parser.add_argument("--ppo_mini_batch_size", type=int, default=32)
    parser.add_argument("--ppo_micro_batch_size_per_gpu", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--use_kl_loss", action="store_true", default=True)
    parser.add_argument("--entropy_coeff", type=float, default=0.0)
    parser.add_argument("--kl_loss_coef", type=float, default=0.001)
    parser.add_argument("--clip_ratio_low", type=float, default=0.2)
    parser.add_argument("--clip_ratio_high", type=float, default=0.3)

    # Rollout/ref settings
    parser.add_argument("--rollout_name", default="vllm", type=str)
    parser.add_argument("--rollout_n", type=int, default=32)
    parser.add_argument("--rollout_val_do_sample", action="store_true", default=True)
    parser.add_argument("--rollout_val_temperature", type=float, default=0.7)
    parser.add_argument("--rollout_val_n", type=int, default=16)
    parser.add_argument("--rollout_max_num_batched_tokens", type=int, default=163840)

    # Parallelism/offload controls
    parser.add_argument("--ref_pipeline_mp", type=int, default=4)
    parser.add_argument("--ref_tensor_mp", type=int, default=2)
    parser.add_argument("--actor_pipeline_mp", type=int, default=4)
    parser.add_argument("--actor_tensor_mp", type=int, default=2)
    parser.add_argument("--param_offload", action="store_true", default=True)
    parser.add_argument("--optimizer_offload", action="store_true", default=True)
    parser.add_argument("--grad_offload", action="store_true", default=True)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)

    # Trainer settings
    parser.add_argument("--n_gpus_per_node", type=int, default=8)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=2)
    parser.add_argument("--test_freq", type=int, default=5)
    parser.add_argument("--total_epochs", type=int, default=1500)
    parser.add_argument("--resume_mode", type=str, default="auto")
    # Optional override for experiment naming to keep it fixed across iterations
    parser.add_argument("--exp_name", default=None, type=str, help="Override experiment name (otherwise derived)")

    # Paths to config and reward function
    parser.add_argument("--config_path", default="~/data_selection/verl/verl/trainer/config", type=str)
    parser.add_argument("--config_name", default="ppo_megatron_trainer.yaml", type=str)
    # parser.add_argument("--reward_path", default="math_reward.py", type=str)
    parser.add_argument("--reward_path", default="math_reward.py", type=str)
    parser.add_argument("--reward_name", default="compute_score", type=str)
    parser.add_argument("--backend", default="megatron", type=str)
    parser.add_argument("--reward_manager", default="naive", type=str)

    args = parser.parse_args()

    # Resolve model path and dataset directories if not explicitly provided
    model_path = args.model_path or get_model_path(args.model)
    train_dir = args.train_dir or get_dataset_dir(args.train_dataset, args.model)
    val_dir = args.val_dir or get_dataset_dir(args.val_dataset, args.model)

    exp_name = args.exp_name or build_exp_name(args.prefix, args.model, args.train_dataset, train_dir, val_dir)
    print('exp', exp_name)
    ckpts_dir = os.path.join(args.ckpts_root, exp_name)
    os.makedirs(ckpts_dir, exist_ok=True)

    train_file = os.path.join(train_dir, args.train_parquet)
    val_file = os.path.join(val_dir, args.val_parquet)

    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    if args.backend == "megatron":
        cmd = [
            sys.executable, "-m", "verl.trainer.main_ppo",
            "--config-path=" + args.config_path,
            "--config-name=" + args.config_name,
            "algorithm.adv_estimator=grpo",
            f"data.train_files={train_file}",
            f"data.val_files={val_file}",
            f"data.train_batch_size={args.train_batch_size}",
            f"data.max_prompt_length={args.max_prompt_length}",
            f"data.max_response_length={args.max_response_length}",
            "data.filter_overlong_prompts=True",
            "data.truncation=error",
            f"actor_rollout_ref.model.path={model_path}",
            f"actor_rollout_ref.actor.optim.lr={args.lr}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={args.ppo_mini_batch_size}",
            f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.ppo_micro_batch_size_per_gpu}",
            f"actor_rollout_ref.actor.use_kl_loss={str(args.use_kl_loss)}",
            f"actor_rollout_ref.actor.entropy_coeff={args.entropy_coeff}",
            f"actor_rollout_ref.actor.kl_loss_coef={args.kl_loss_coef}",
            f"actor_rollout_ref.actor.clip_ratio_low={args.clip_ratio_low}",
            f"actor_rollout_ref.actor.clip_ratio_high={args.clip_ratio_high}",
            # "actor_rollout_ref.model.enable_gradient_checkpointing=True",
            f"actor_rollout_ref.rollout.name={args.rollout_name}",
            f"actor_rollout_ref.rollout.n={args.rollout_n}",
            f"actor_rollout_ref.rollout.val_kwargs.do_sample={str(args.rollout_val_do_sample)}",
            f"actor_rollout_ref.rollout.val_kwargs.temperature={args.rollout_val_temperature}",
            f"actor_rollout_ref.rollout.val_kwargs.n={args.rollout_val_n}",
            f"actor_rollout_ref.rollout.max_num_batched_tokens={args.rollout_max_num_batched_tokens}",
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
            f"actor_rollout_ref.ref.megatron.pipeline_model_parallel_size={args.ref_pipeline_mp}",
            f"actor_rollout_ref.ref.megatron.tensor_model_parallel_size={args.ref_tensor_mp}",
            "actor_rollout_ref.actor.megatron.use_dist_checkpointing=False",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2",
            "actor_rollout_ref.ref.megatron.use_dist_checkpointing=False",
            f"actor_rollout_ref.actor.megatron.param_offload={str(args.param_offload)}",
            f"actor_rollout_ref.actor.megatron.pipeline_model_parallel_size={args.actor_pipeline_mp}",
            f"actor_rollout_ref.actor.megatron.tensor_model_parallel_size={args.actor_tensor_mp}",
            f"actor_rollout_ref.actor.megatron.optimizer_offload={str(args.optimizer_offload)}",
            f"actor_rollout_ref.actor.megatron.grad_offload={str(args.grad_offload)}",
            "actor_rollout_ref.ref.megatron.param_offload=True",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={args.gpu_memory_utilization}",
            "trainer.logger=['console','tensorboard']",
            f"custom_reward_function.path=~/data_selection/verl/verl/utils/reward_score/{args.reward_path}",
            f"custom_reward_function.name={args.reward_name}",
            f"trainer.project_name={args.project_name}",
            f"trainer.experiment_name={exp_name}",
            f"trainer.n_gpus_per_node={args.n_gpus_per_node}",
            f"trainer.nnodes={args.nnodes}",
            f"trainer.save_freq={args.save_freq}",
            f"trainer.test_freq={args.test_freq}",
            f"trainer.total_epochs={args.total_epochs}",
            f"trainer.resume_mode={args.resume_mode}",
            f"trainer.default_local_dir={ckpts_dir}",
            f"reward_model.reward_manager={args.reward_manager}",
        ]
    else:
        cmd = [
            sys.executable, "-m", "verl.trainer.main_ppo",
            "algorithm.adv_estimator=grpo",
            f"data.train_files={train_file}",
            f"data.val_files={val_file}",
            f"data.train_batch_size={args.train_batch_size}",
            f"data.max_prompt_length={args.max_prompt_length}",
            f"data.max_response_length={args.max_response_length}",
            "data.filter_overlong_prompts=True",
            "data.truncation=error",
            f"actor_rollout_ref.model.path={model_path}",
            f"actor_rollout_ref.actor.optim.lr={args.lr}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={args.ppo_mini_batch_size}",
            f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={int(args.ppo_micro_batch_size_per_gpu) * 8}",
            f"actor_rollout_ref.actor.use_kl_loss={str(args.use_kl_loss)}",
            f"actor_rollout_ref.actor.entropy_coeff={args.entropy_coeff}",
            f"actor_rollout_ref.actor.kl_loss_coef={args.kl_loss_coef}",
            f"actor_rollout_ref.actor.clip_ratio_low={args.clip_ratio_low}",
            f"actor_rollout_ref.actor.clip_ratio_high={args.clip_ratio_high}",
            "actor_rollout_ref.model.enable_gradient_checkpointing=True",
            f"actor_rollout_ref.rollout.name={args.rollout_name}",
            f"actor_rollout_ref.rollout.n={args.rollout_n}",
            f"actor_rollout_ref.rollout.val_kwargs.do_sample={str(args.rollout_val_do_sample)}",
            f"actor_rollout_ref.rollout.val_kwargs.temperature={args.rollout_val_temperature}",
            f"actor_rollout_ref.rollout.val_kwargs.n={args.rollout_val_n}",
            f"actor_rollout_ref.rollout.max_num_batched_tokens={args.rollout_max_num_batched_tokens}",
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={args.gpu_memory_utilization}",
            "trainer.logger=['console','tensorboard']",
            f"custom_reward_function.path=~/data_selection/verl/verl/utils/reward_score/{args.reward_path}",
            f"custom_reward_function.name={args.reward_name}",
            f"trainer.project_name={args.project_name}",
            f"trainer.experiment_name={exp_name}",
            f"trainer.n_gpus_per_node={args.n_gpus_per_node}",
            f"trainer.nnodes={args.nnodes}",
            f"trainer.save_freq={args.save_freq}",
            f"trainer.test_freq={args.test_freq}",
            f"trainer.total_epochs={args.total_epochs}",
            f"trainer.resume_mode={args.resume_mode}",
            f"trainer.default_local_dir={ckpts_dir}",
        ]

    print("Executing:\n" + " \\n+    \n".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Training launched. Checkpoints at: {ckpts_dir}")


if __name__ == "__main__":
    main()


