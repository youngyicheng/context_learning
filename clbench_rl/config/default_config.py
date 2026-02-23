"""Default configuration for CL-bench RL training."""

from typing import Any, Dict, Optional


def get_default_config() -> Dict[str, Any]:
    """Return default configuration dict."""
    return {
        "data": {
            "split": "train",
            "max_samples": 10,
            "subset": None,
            "cache_dir": None,
        },
        "challenge_model": {
            "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "device": None,
            "max_new_tokens": 1024,
            "temperature": 0.7,
        },
        "solver_model": {
            "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "device": None,
            "max_new_tokens": 2048,
            "temperature": 0.7,
        },
        "reward": {
            "use_llm_judge": False,
            "judge_model": "gpt-4",
            # Challenge reward component weights
            "challenge_correctness_weight": 1.0,
            "repetition_penalty_weight": 0.3,
            "format_penalty_weight": 0.2,
            "relevance_weight": 0.3,
            "rubric_quality_weight": 0.2,
            # Solver reward component weights
            "solver_correctness_weight": 1.0,
            "context_grounding_weight": 0.3,
            "tool_usage_weight": 0.2,
        },
        "training": {
            "train_solver": True,
            "train_challenge": False,
            "epochs": 1,
            "lr": 1e-5,
            "checkpoint_dir": "checkpoints",
            "save_every": 100,
            "log_every": 5,
        },
    }


def merge_config(user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge user config over defaults with shallow merge per top-level key."""
    base = get_default_config()
    if not user_config:
        return base
    for key, val in user_config.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            base[key] = {**base[key], **val}
        else:
            base[key] = val
    return base
