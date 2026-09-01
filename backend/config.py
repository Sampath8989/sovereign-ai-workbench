"""
Hardware tier detection and model roster configuration.
Reads HARDWARE_TIER env var to determine the active configuration.
"""

import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

def _detect_tier() -> str:
    """Auto-detect the best tier based on available GPU VRAM."""
    env_tier = os.getenv("HARDWARE_TIER")
    if env_tier:
        return env_tier

    # Query live VRAM to decide
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            total_mb = float(result.stdout.strip().split("\n")[0])
            total_gb = total_mb / 1024
            if total_gb >= 8.0:
                return "DEMO"
            else:
                return "BUILD"  # 4GB card: use smaller models
    except Exception:
        pass

    return "BUILD"  # Safe default


HARDWARE_TIER = _detect_tier()

# Model rosters per tier: {model_name: estimated_vram_gb}
# Recalculated for 5B-class model (2.29 GB weights + ~0.5 GB KV cache = 2.8 GB VRAM)
MODEL_ROSTERS: Dict[str, Dict[str, float]] = {
    "BUILD": {
        "qwen1_5-4b-chat-q4_k_m.gguf": 2.8,
    },
    "DEMO": {
        "qwen1_5-4b-chat-q4_k_m.gguf": 2.8,
    },
}

# Emergency fallback model on disk (0.5B kept for low-VRAM/OOM emergencies)
EMERGENCY_FALLBACK_MODEL = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

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


def _model_file_valid(model_name: str) -> bool:
    """Check if a model file exists AND is large enough to be a real GGUF.
    
    Incomplete downloads (e.g., partial wget) are rejected by checking
    minimum file size — a valid GGUF for a 5B model should be > 2GB.
    """
    path = os.path.join("models", model_name)
    if not os.path.exists(path):
        return False
    size_mb = os.path.getsize(path) / (1024 * 1024)
    # Minimum: 2000MB for 5B model, 100MB for emergency 0.5B fallback
    min_mb = 100 if "0.5b" in model_name.lower() else 2000
    if size_mb < min_mb:
        logger.warning(
            f"Model file {path} is only {size_mb:.0f}MB (expected >{min_mb}MB). "
            f"Possibly incomplete download."
        )
        return False
    return True


def get_router_model() -> str:
    """Return the router model name for the current tier.
    
    If the 5B model is not available or incomplete, falls back to the emergency
    0.5B fallback model with a loud warning.
    """
    roster = get_model_roster()
    primary_model = next(iter(roster.keys()))
    if _model_file_valid(primary_model):
        return primary_model

    # Check emergency fallback
    if _model_file_valid(EMERGENCY_FALLBACK_MODEL):
        logger.warning(
            f"\n"
            f"======================================================================\n"
            f"SAFETY WARNING: Primary 5B model {primary_model} not ready.\n"
            f"Using emergency fallback {EMERGENCY_FALLBACK_MODEL} to prevent crash.\n"
            f"======================================================================\n"
        )
        return EMERGENCY_FALLBACK_MODEL

    logger.critical(
        f"\n"
        f"CRITICAL: Neither 5B model ({primary_model}) nor emergency fallback ({EMERGENCY_FALLBACK_MODEL}) are valid.\n"
        f"System will use MockLLM.\n"
    )
    return primary_model


def get_coder_model() -> str:
    """Return the coder/generator model name for the current tier."""
    return get_router_model()
