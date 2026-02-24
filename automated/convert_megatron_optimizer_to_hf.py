#!/usr/bin/env python3
"""Convert Megatron optimizer state to HuggingFace-style optimizer tensors."""

from __future__ import annotations

import argparse
import math
import os
from pprint import pprint
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, cast

import torch


# Ensure verl package is importable
sys.path.append("verl")

from transformers.models.auto.configuration_auto import AutoConfig  # noqa: E402

from megatron.core import parallel_state as mpu  # noqa: E402
from megatron.core.dist_checkpointing.serialization import (  # noqa: E402
    get_default_load_sharded_strategy,
    load_sharded_metadata,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (  # noqa: E402
    FullyParallelLoadStrategyWrapper,
)

from verl.utils.megatron.dist_checkpointing import load_dist_checkpointing  # noqa: E402
from verl.utils.megatron_utils import get_dist_checkpoint_path  # noqa: E402


def detect_parallel_config(dist_ckpt_path: Path) -> Tuple[int | None, int | None]:
    """Best-effort detection of PP/TP degree from dist checkpoint files."""

    if not dist_ckpt_path.exists():
        return None, None

    distcp_files = [f.name for f in dist_ckpt_path.iterdir() if f.name.endswith(".distcp")]
    max_pp = 0
    max_tp = 0

    for name in distcp_files:
        if not name.startswith("__") or "_" not in name:
            continue
        try:
            pp_rank, tp_rank = (int(part) for part in name[2:-7].split("_"))
        except ValueError:
            continue
        max_pp = max(max_pp, pp_rank + 1)
        max_tp = max(max_tp, tp_rank + 1)

    return (max_pp or None), (max_tp or None)


def init_minimal_megatron(tp_size: int, pp_size: int) -> None:
    """Initialize Megatron parallel groups in a single-process environment."""

    if not torch.distributed.is_initialized():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")

        torch.distributed.init_process_group(
            backend="gloo",
            init_method="env://",
            world_size=1,
            rank=0,
        )

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )


def load_megatron_optimizer_state(
    checkpoint_path: Path, tp_size: int | None, pp_size: int | None
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Load Megatron distributed checkpoint and extract optimizer tensors."""

    dist_ckpt_path = Path(get_dist_checkpoint_path(str(checkpoint_path)))
    if not dist_ckpt_path.exists():
        raise FileNotFoundError(f"Distributed checkpoint directory not found: {dist_ckpt_path}")

    detected_pp, detected_tp = detect_parallel_config(dist_ckpt_path)
    tp = tp_size or detected_tp or 1
    pp = pp_size or detected_pp or 1

    init_minimal_megatron(tp, pp)

    load_strategy = get_default_load_sharded_strategy(str(dist_ckpt_path))
    load_strategy = FullyParallelLoadStrategyWrapper(
        load_strategy,
        mpu.get_data_parallel_group(with_context_parallel=True),
    )

    sharded_metadata = load_sharded_metadata(str(dist_ckpt_path), load_strategy)
    full_state = load_dist_checkpointing(
            sharded_state_dict=sharded_metadata,
            ckpt_dir=str(dist_ckpt_path),
        )
    
    # pprint(full_state[0])
    step = full_state['optimizer']['optimizer']['param_groups'][0]['step']
    full_state = cast(
        Dict[str, torch.Tensor],
        full_state
    )

    optimizer_state: Dict[str, torch.Tensor] = {}
    for key, value in full_state.items():
        if not key.startswith("optimizer.state."):
            continue
        if not isinstance(value, torch.Tensor):
            continue
        if value.numel() == 0:
            continue
        optimizer_state[key] = value.cpu()

    return optimizer_state, int(step)


def describe_target(
    megatron_param: str,
    config,
) -> Tuple[bool, List[Tuple[str, Sequence[int]]]] | None:
    """Map a Megatron parameter path to HuggingFace-equivalent tensors."""

    hidden = config.hidden_size
    intermediate = config.intermediate_size
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = hidden // num_heads
    kv_dim = num_kv_heads * head_dim

    attn_norm_dim = getattr(config, "rope_embeds_per_head", None)
    if attn_norm_dim is None:
        attn_norm_dim = head_dim
    layer_input_ln_dim = hidden

    per_layer = ".layers." in megatron_param

    if "_extra_state" in megatron_param:
        return None

    if megatron_param == "embedding.word_embeddings.weight":
        return False, [("embed_tokens.weight", (config.vocab_size, hidden))]

    if megatron_param == "embedding.word_embeddings.bias":
        return False, [("embed_tokens.bias", (config.vocab_size,))]

    if megatron_param == "decoder.final_layernorm.weight":
        return False, [("norm.weight", (hidden,))]

    if megatron_param == "decoder.final_layernorm.bias":
        return False, [("norm.bias", (hidden,))]

    if megatron_param == "decoder.layers.self_attention.linear_proj.weight":
        return True, [("layers.{layer}.self_attn.o_proj.weight", (hidden, hidden))]

    if megatron_param == "decoder.layers.self_attention.linear_proj.bias":
        return True, [("layers.{layer}.self_attn.o_proj.bias", (hidden,))]

    if megatron_param == "decoder.layers.self_attention.linear_qkv.layer_norm_weight":
        return True, [("layers.{layer}.input_layernorm.weight", (layer_input_ln_dim,))]

    if megatron_param == "decoder.layers.self_attention.linear_qkv.layer_norm_bias":
        return True, [("layers.{layer}.input_layernorm.bias", (layer_input_ln_dim,))]

    if megatron_param == "decoder.layers.self_attention.q_layernorm.weight":
        return True, [("layers.{layer}.self_attn.q_norm.weight", (attn_norm_dim,))]

    if megatron_param == "decoder.layers.self_attention.k_layernorm.weight":
        return True, [("layers.{layer}.self_attn.k_norm.weight", (attn_norm_dim,))]

    if megatron_param == "decoder.layers.self_attention.linear_qkv.weight":
        return True, [
            ("layers.{layer}.self_attn.q_proj.weight", (hidden, hidden)),
            ("layers.{layer}.self_attn.k_proj.weight", (kv_dim, hidden)),
            ("layers.{layer}.self_attn.v_proj.weight", (kv_dim, hidden)),
        ]

    if megatron_param == "decoder.layers.self_attention.linear_qkv.bias":
        return True, [
            ("layers.{layer}.self_attn.q_proj.bias", (hidden,)),
            ("layers.{layer}.self_attn.k_proj.bias", (kv_dim,)),
            ("layers.{layer}.self_attn.v_proj.bias", (kv_dim,)),
        ]

    if megatron_param == "decoder.layers.mlp.linear_fc1.weight":
        return True, [
            ("layers.{layer}.mlp.gate_proj.weight", (intermediate, hidden)),
            ("layers.{layer}.mlp.up_proj.weight", (intermediate, hidden)),
        ]

    if megatron_param == "decoder.layers.mlp.linear_fc1.bias":
        return True, [
            ("layers.{layer}.mlp.gate_proj.bias", (intermediate,)),
            ("layers.{layer}.mlp.up_proj.bias", (intermediate,)),
        ]

    if megatron_param == "decoder.layers.mlp.linear_fc1.layer_norm_weight":
        return True, [("layers.{layer}.post_attention_layernorm.weight", (hidden,))]

    if megatron_param == "decoder.layers.mlp.linear_fc1.layer_norm_bias":
        return True, [("layers.{layer}.post_attention_layernorm.bias", (hidden,))]

    if megatron_param == "decoder.layers.mlp.linear_fc2.weight":
        return True, [("layers.{layer}.mlp.down_proj.weight", (hidden, intermediate))]

    if megatron_param == "decoder.layers.mlp.linear_fc2.bias":
        return True, [("layers.{layer}.mlp.down_proj.bias", (hidden,))]

    if megatron_param == "output_layer.weight":
        return False, [("lm_head.weight", (config.vocab_size, hidden))]

    return None


def slice_tensor(
    tensor: torch.Tensor,
    per_layer: bool,
    specs: Sequence[Tuple[str, Sequence[int]]],
    num_layers: int,
) -> Iterable[Tuple[str, torch.Tensor]]:
    """Yield HF tensors sliced from the Megatron optimizer tensor."""

    flat = tensor.reshape(-1)
    spec_sizes = [math.prod(shape) for _, shape in specs]
    total_per_layer = sum(spec_sizes)

    if per_layer:
        expected = total_per_layer * num_layers
        if flat.numel() != expected:
            raise ValueError(
                f"Unexpected tensor size {flat.numel()} for per-layer mapping (expected {expected})."
            )
        layer_chunks = flat.reshape(num_layers, total_per_layer)
        for layer_idx in range(num_layers):
            offset = 0
            for (name_template, shape), size in zip(specs, spec_sizes):
                chunk = layer_chunks[layer_idx, offset : offset + size]
                offset += size
                yield name_template.format(layer=layer_idx), chunk.reshape(shape).clone()
    else:
        expected = total_per_layer
        if flat.numel() != expected:
            raise ValueError(
                f"Unexpected tensor size {flat.numel()} for global mapping (expected {expected})."
            )
        offset = 0
        for (name_template, shape), size in zip(specs, spec_sizes):
            chunk = flat[offset : offset + size]
            offset += size
            yield name_template, chunk.reshape(shape).clone()


def convert_qkv_weight(
    tensor: torch.Tensor,
    config,
    stat_name: str,
    num_layers: int,
) -> Dict[str, torch.Tensor]:
    hidden = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = hidden // num_heads
    kv_dim = num_kv_heads * head_dim

    rows_per_layer = hidden + 2 * kv_dim
    expected = num_layers * rows_per_layer * hidden
    if tensor.numel() != expected:
        raise ValueError(
            f"Unexpected tensor size {tensor.numel()} for qkv weight (expected {expected})."
        )

    tensor = tensor.view(num_layers, rows_per_layer, hidden)

    q_rows_per_chunk = hidden // num_kv_heads
    kv_rows_per_chunk = kv_dim // num_kv_heads

    converted: Dict[str, torch.Tensor] = {}

    for layer_idx in range(num_layers):
        layer_tensor = tensor[layer_idx]
        chunks = layer_tensor.chunk(num_kv_heads, dim=0)
        if any(chunk.size(0) != q_rows_per_chunk + 2 * kv_rows_per_chunk for chunk in chunks):
            raise ValueError("Unexpected chunk size while splitting qkv weight")

        q_parts, k_parts, v_parts = [], [], []
        for chunk in chunks:
            q_part, k_part, v_part = chunk.split(
                (q_rows_per_chunk, kv_rows_per_chunk, kv_rows_per_chunk), dim=0
            )
            q_parts.append(q_part)
            k_parts.append(k_part)
            v_parts.append(v_part)

        q_tensor = torch.cat(q_parts, dim=0).contiguous()
        k_tensor = torch.cat(k_parts, dim=0).contiguous()
        v_tensor = torch.cat(v_parts, dim=0).contiguous()

        converted[f"optimizer.state.{stat_name}.layers.{layer_idx}.self_attn.q_proj.weight"] = q_tensor
        converted[f"optimizer.state.{stat_name}.layers.{layer_idx}.self_attn.k_proj.weight"] = k_tensor
        converted[f"optimizer.state.{stat_name}.layers.{layer_idx}.self_attn.v_proj.weight"] = v_tensor

    return converted


def convert_qkv_bias(
    tensor: torch.Tensor,
    config,
    stat_name: str,
    num_layers: int,
) -> Dict[str, torch.Tensor]:
    hidden = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = hidden // num_heads
    kv_dim = num_kv_heads * head_dim

    rows_per_layer = hidden + 2 * kv_dim
    expected = num_layers * rows_per_layer
    if tensor.numel() != expected:
        raise ValueError(
            f"Unexpected tensor size {tensor.numel()} for qkv bias (expected {expected})."
        )

    tensor = tensor.view(num_layers, rows_per_layer)

    q_rows_per_chunk = hidden // num_kv_heads
    kv_rows_per_chunk = kv_dim // num_kv_heads

    converted: Dict[str, torch.Tensor] = {}

    for layer_idx in range(num_layers):
        layer_tensor = tensor[layer_idx]
        chunks = layer_tensor.chunk(num_kv_heads, dim=0)
        if any(chunk.size(0) != q_rows_per_chunk + 2 * kv_rows_per_chunk for chunk in chunks):
            raise ValueError("Unexpected chunk size while splitting qkv bias")

        q_parts, k_parts, v_parts = [], [], []
        for chunk in chunks:
            q_part, k_part, v_part = chunk.split(
                (q_rows_per_chunk, kv_rows_per_chunk, kv_rows_per_chunk), dim=0
            )
            q_parts.append(q_part)
            k_parts.append(k_part)
            v_parts.append(v_part)

        q_tensor = torch.cat(q_parts, dim=0).contiguous()
        k_tensor = torch.cat(k_parts, dim=0).contiguous()
        v_tensor = torch.cat(v_parts, dim=0).contiguous()

        converted[f"optimizer.state.{stat_name}.layers.{layer_idx}.self_attn.q_proj.bias"] = q_tensor
        converted[f"optimizer.state.{stat_name}.layers.{layer_idx}.self_attn.k_proj.bias"] = k_tensor
        converted[f"optimizer.state.{stat_name}.layers.{layer_idx}.self_attn.v_proj.bias"] = v_tensor

    return converted


def convert_optimizer_state(
    optimizer_state: Dict[str, torch.Tensor],
    config,
) -> Dict[str, torch.Tensor]:
    """Convert Megatron optimizer tensors to HuggingFace parameter naming."""

    converted: Dict[str, torch.Tensor] = {}
    num_layers = config.num_hidden_layers
    prefix = "optimizer.state."

    for key, tensor in optimizer_state.items():
        if not key.startswith(prefix):
            raise ValueError(f"Unexpected optimizer key prefix: {key}")

        remainder = key[len(prefix) :]
        if "." not in remainder:
            raise ValueError(f"Unexpected optimizer key format: {key}")
        stat_name, megatron_param = remainder.split(".", 1)

        if megatron_param == "decoder.layers.self_attention.linear_qkv.weight":
            qkv_mapping = convert_qkv_weight(tensor, config, stat_name, num_layers)
            converted.update(qkv_mapping)
            continue

        if megatron_param == "decoder.layers.self_attention.linear_qkv.bias":
            qkv_mapping = convert_qkv_bias(tensor, config, stat_name, num_layers)
            converted.update(qkv_mapping)
            continue

        descriptor = describe_target(megatron_param, config)
        if descriptor is None:
            raise KeyError(f"No mapping rule for Megatron parameter: {megatron_param}")

        per_layer, specs = descriptor

        for target_name, target_tensor in slice_tensor(tensor, per_layer, specs, num_layers):
            converted_key = f"optimizer.state.{stat_name}.{target_name}"
            converted[converted_key] = target_tensor

    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Megatron optimizer state tensors to HuggingFace-style keys.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the base checkpoint directory (e.g. .../global_step_XX/actor)",
    )
    parser.add_argument(
        "--hf-config",
        required=True,
        type=str,
        help="Path to the HuggingFace model directory containing config.json.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Path to save the converted optimizer state (torch.save).",
    )
    parser.add_argument("--tp-size", type=int, default=None, help="Tensor parallel degree (optional)")
    parser.add_argument("--pp-size", type=int, default=None, help="Pipeline parallel degree (optional)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_config_path = args.hf_config
    checkpoint_path = args.checkpoint_path
    output_path = args.output

    config = AutoConfig.from_pretrained(str(hf_config_path), trust_remote_code=True)

    optimizer_state, step = load_megatron_optimizer_state(
        checkpoint_path=Path(checkpoint_path),
        tp_size=args.tp_size,
        pp_size=args.pp_size,
    )

    converted = convert_optimizer_state(optimizer_state, config)
    converted['step'] = step

    torch.save(converted, output_path)
    print(f"Converted optimizer state saved to {output_path}")
    print(f"Total tensors: {len(converted)}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

