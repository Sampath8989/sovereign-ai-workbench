"""
Hardware tier detection and model roster configuration.
Reads HARDWARE_TIER env var to determine the active configuration.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

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

# Comprehensive metadata registry for all local models
MODEL_METADATA_REGISTRY: Dict[str, Dict[str, Any]] = {
    "deepseek-r1-7b.gguf": {
        "name": "DeepSeek R1 7B",
        "category": "REASONING",
        "param_size": "7B",
        "vram_gb": 4.5,
        "description": "Distilled reasoning model for math, complex logic, and step-by-step verification.",
    },
    "phi4-14b.gguf": {
        "name": "Phi-4 14B",
        "category": "REASONING",
        "param_size": "14B",
        "vram_gb": 9.0,
        "description": "High-capacity 14B reasoning powerhouse for deep synthesis, complex problem solving, and architecture design.",
    },
    "qwen2.5-coder-7b-instruct-q3_k_m.gguf": {
        "name": "Qwen 2.5 Coder 7B",
        "category": "CODE",
        "param_size": "7B",
        "vram_gb": 4.0,
        "description": "Specialized coding and deliverable synthesis model for Python scripts, debugging, algorithms, and documents.",
    },
    "llava-7b.gguf": {
        "name": "LLaVA 7B (Vision)",
        "category": "VISION",
        "param_size": "7B",
        "vram_gb": 4.5,
        "description": "Multimodal vision-language model for image reasoning, diagram analysis, and document OCR.",
    },
    "qwen2.5-7b-instruct-q3_k_m.gguf": {
        "name": "Qwen 2.5 7B Instruct",
        "category": "GENERAL",
        "param_size": "7B",
        "vram_gb": 4.0,
        "description": "High-accuracy general instruction-tuned model for structured analysis, reporting, and Q&A.",
    },
    "qwen2.5-7b.gguf": {
        "name": "Qwen 2.5 7B",
        "category": "GENERAL",
        "param_size": "7B",
        "vram_gb": 4.5,
        "description": "General conversational 7B foundational model for multi-domain queries.",
    },
    "qwen1_5-4b-chat-q4_k_m.gguf": {
        "name": "Qwen 1.5 4B Chat",
        "category": "GENERAL",
        "param_size": "4B",
        "vram_gb": 2.8,
        "description": "Fast, low-latency conversational model optimized for 4GB VRAM hardware.",
    },
    "qwen2.5-coder-3b-instruct-q4_k_m.gguf": {
        "name": "Qwen 2.5 Coder 3B",
        "category": "CODE",
        "param_size": "3B",
        "vram_gb": 2.0,
        "description": "Lightweight code generator for quick script synthesis.",
    },
    "qwen2.5-0.5b-instruct-q4_k_m.gguf": {
        "name": "Qwen 2.5 0.5B",
        "category": "FALLBACK",
        "param_size": "0.5B",
        "vram_gb": 0.8,
        "description": "Ultra-low memory emergency fallback model.",
    },
}

# Model rosters per tier: {model_name: estimated_vram_gb}
MODEL_ROSTERS: Dict[str, Dict[str, float]] = {
    "BUILD": {
        "qwen1_5-4b-chat-q4_k_m.gguf": 2.8,
        "qwen2.5-coder-7b-instruct-q3_k_m.gguf": 3.0,
        "deepseek-r1-7b.gguf": 3.2,
        "qwen2.5-7b-instruct-q3_k_m.gguf": 3.0,
        "qwen2.5-7b.gguf": 3.2,
        "llava-7b.gguf": 3.2,
        "phi4-14b.gguf": 3.6,
        "qwen2.5-coder-3b-instruct-q4_k_m.gguf": 2.0,
        "qwen2.5-0.5b-instruct-q4_k_m.gguf": 0.8,
    },
    "DEMO": {
        "qwen2.5-coder-7b-instruct-q3_k_m.gguf": 4.0,
        "deepseek-r1-7b.gguf": 4.5,
        "phi4-14b.gguf": 9.0,
        "llava-7b.gguf": 4.5,
        "qwen2.5-7b-instruct-q3_k_m.gguf": 4.0,
        "qwen2.5-7b.gguf": 4.5,
        "qwen1_5-4b-chat-q4_k_m.gguf": 2.8,
        "qwen2.5-coder-3b-instruct-q4_k_m.gguf": 2.0,
        "qwen2.5-0.5b-instruct-q4_k_m.gguf": 0.8,
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
    """Check if a model file exists AND is large enough to be a real GGUF."""
    path = os.path.join("models", model_name)
    if not os.path.exists(path):
        return False
    size_mb = os.path.getsize(path) / (1024 * 1024)
    min_mb = 100 if "0.5b" in model_name.lower() else 500
    if size_mb < min_mb:
        logger.warning(
            f"Model file {path} is only {size_mb:.0f}MB (expected >{min_mb}MB). "
            f"Possibly incomplete download."
        )
        return False
    return True


def get_available_models() -> List[Dict[str, Any]]:
    """
    Scan models directory and return detailed metadata for all available models,
    including the Auto (Intelligent Routing) option at the top.
    """
    models_dir = Path("models")
    available = []

    # 1. Always include Auto option as the default top choice
    available.append({
        "id": "auto",
        "name": "Auto (Intelligent Routing)",
        "category": "AUTO",
        "param_size": "Auto",
        "vram_gb": 0.0,
        "size_gb": 0.0,
        "description": "Automatically selects the best 7B/14B model based on task intent (Coding, Math, Vision, or Chat).",
        "is_present": True,
    })

    # 2. Check all known registry models in order
    for filename, meta in MODEL_METADATA_REGISTRY.items():
        file_path = models_dir / filename
        is_present = file_path.exists()
        size_gb = round(file_path.stat().st_size / (1024 ** 3), 2) if is_present else 0.0

        if is_present:
            available.append({
                "id": filename,
                "name": meta["name"],
                "category": meta["category"],
                "param_size": meta.get("param_size", ""),
                "vram_gb": meta.get("vram_gb", 4.0),
                "size_gb": size_gb,
                "description": meta["description"],
                "is_present": True,
            })

    # 3. Check for any other .gguf files in models/ not in registry
    if models_dir.exists():
        for p in models_dir.glob("*.gguf"):
            if p.name not in MODEL_METADATA_REGISTRY and not p.name.endswith("-mmproj.gguf"):
                size_gb = round(p.stat().st_size / (1024 ** 3), 2)
                available.append({
                    "id": p.name,
                    "name": p.stem.replace("-", " ").title(),
                    "category": "CUSTOM",
                    "param_size": "Custom",
                    "vram_gb": 4.0,
                    "size_gb": size_gb,
                    "description": f"Custom model file: {p.name}",
                    "is_present": True,
                })

    return available


def get_router_model() -> str:
    """Return the router model name for the current tier."""
    roster = get_model_roster()
    for model_name in roster.keys():
        if _model_file_valid(model_name):
            return model_name

    # Check emergency fallback
    if _model_file_valid(EMERGENCY_FALLBACK_MODEL):
        logger.warning(
            f"Using emergency fallback {EMERGENCY_FALLBACK_MODEL}."
        )
        return EMERGENCY_FALLBACK_MODEL

    return "qwen1_5-4b-chat-q4_k_m.gguf"


def get_coder_model() -> str:
    """Return the coder/generator model name for the current tier."""
    if _model_file_valid("qwen2.5-coder-7b-instruct-q3_k_m.gguf"):
        return "qwen2.5-coder-7b-instruct-q3_k_m.gguf"
    if _model_file_valid("qwen2.5-coder-3b-instruct-q4_k_m.gguf"):
        return "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
    return get_router_model()
