"""
Hardware tier detection and model roster configuration.
Reads HARDWARE_TIER env var to determine the active configuration.
"""

import os
from typing import Dict, List

HARDWARE_TIER = os.getenv("HARDWARE_TIER", "BUILD")

# Model rosters per tier: {model_name: estimated_vram_gb}
MODEL_ROSTERS: Dict[str, Dict[str, float]] = {
    "BUILD": {
        "qwen2.5-0.5b-instruct-q4_k_m.gguf": 0.5,
        "qwen2.5-coder-3b-instruct-q4_k_m.gguf": 1.5,
    },
    "DEMO": {
        "qwen2.5-1.5b-instruct-q4_k_m.gguf": 1.5,
        "qwen2.5-coder-3b-instruct-q4_k_m.gguf": 1.5,
    },
}

TIER_VRAM_GB: Dict[str, float] = {
    "BUILD": 4.0,
    "DEMO": 8.0,
}


def get_tier() -> str:
    """Return the current hardware tier."""
    return HARDWARE_TIER


def get_max_vram_gb() -> float:
    """Return max VRAM in GB for the current tier."""
    return TIER_VRAM_GB.get(HARDWARE_TIER, 4.0)


def get_model_roster() -> Dict[str, float]:
    """Return the model roster for the current tier."""
    return MODEL_ROSTERS.get(HARDWARE_TIER, MODEL_ROSTERS["BUILD"])


def get_model_path(model_name: str) -> str:
    """Return the filesystem path for a given model name."""
    return os.path.join("models", model_name)


def get_router_model() -> str:
    """Return the router model name for the current tier."""
    roster = get_model_roster()
    return next(iter(roster.keys()))


def get_coder_model() -> str:
    """Return the coder model name for the current tier."""
    roster = get_model_roster()
    keys = list(roster.keys())
    return keys[1] if len(keys) > 1 else keys[0]
