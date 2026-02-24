"""
Centralized configuration for model paths and common constants.

Users should update `MODELS` to reflect their local/remote model store.
"""

from typing import Dict


# Map a short, stable model key to its absolute model path.
# Update these paths to match your environment.
MODELS: Dict[str, str] = {
    # MODEL NAME: MODEL PATH
}


# Reward style used across datasets for consistency
REWARD_STYLE: str = "rule-lighteval/MATH_v2"


# Base directory for auto-selected dataset outputs
BASE_DATA_DIR: str = "/path/to/data" #SET YOUR OWN PATH

# Base directory for API responses
BASE_RESPONSES_DIR: str = "/path/to/responses" #SET YOUR OWN PATH


# Map model keys to short model type identifiers used in dataset paths
MODEL_TYPES: Dict[str, str] = {
    # Meta Llama
    # NAME: Model_type (e.g. qwen3: qwen)
}

def get_model_path(model_key: str) -> str:
    """Return the absolute model path for a given key, or raise with guidance."""
    if model_key not in MODELS:
        known = ", ".join(sorted(MODELS.keys()))
        raise KeyError(
            f"Unknown model key '{model_key}'. Known keys: {known}. "
            "Update config.MODELS if you need a new mapping."
        )
    return MODELS[model_key]


def get_model_type(model_key: str) -> str:
    """Return a short model type string for use in dataset paths.

    Falls back to simple heuristics if the key is not explicitly mapped.
    """
    if model_key in MODEL_TYPES:
        return MODEL_TYPES[model_key]
    key = model_key.lower()
    if "qwen" in key:
        return "qwen"
    if "llama" in key:
        return "llama"
    if "deepseek" in key:
        return "deepseek"
    if "mistral" in key or "mixtral" in key:
        return "mistral"
    return key.split("-")[0]


def get_dataset_dir(dataset_label: str, model_key: str = None) -> str:
    """Return dataset dir as {BASE_DATA_DIR}/{dataset}/{model_type}."""
    safe_dataset = dataset_label.replace("/", "-")
    return f"{BASE_DATA_DIR}/{safe_dataset}"


def get_response_dir(dataset_label: str, model_name: str) -> str:
    """Return responses dir as {BASE_RESPONSES_DIR}/{model_name}/{dataset}."""
    safe_dataset = dataset_label.replace("/", "-")
    safe_model_name = model_name.replace("/", "-")
    return f"{BASE_RESPONSES_DIR}/{safe_model_name}/{safe_dataset}"


# Backward-compat alias (to avoid breakage while migrating code)
get_output_dir = get_dataset_dir


