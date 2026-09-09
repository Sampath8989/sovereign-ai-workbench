"""
Test suite for 3B Model Variants and Pop!_OS / Linux Compatibility.
Tests:
- 3B models registration in MODEL_METADATA_REGISTRY
- Discovery of 3B models (Llama 3.2 3B, Qwen 2.5 Coder 3B, Qwen 2.5 3B)
- Auto-routing with PREFER_3B_MODELS=true
- GET /models endpoint returning 3B model metadata
"""

import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.config import get_available_models, MODEL_METADATA_REGISTRY, get_router_model, get_coder_model
from backend.core.router import auto_select_model


class Test3BModelRegistry:
    """Test that all 3B models are registered with valid metadata."""

    def test_3b_models_in_metadata_registry(self):
        expected_3b = [
            "llama-3.2-3b-instruct-q4_k_m.gguf",
            "llama-3.2-3b-instruct-q5_k_m.gguf",
            "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
            "qwen2.5-coder-3b-instruct-q5_k_m.gguf",
            "qwen2.5-3b-instruct-q4_k_m.gguf",
            "qwen2.5-3b-instruct-q5_k_m.gguf",
        ]
        for m in expected_3b:
            assert m in MODEL_METADATA_REGISTRY, f"Missing 3B model in registry: {m}"
            meta = MODEL_METADATA_REGISTRY[m]
            assert meta["param_size"] == "3B"
            assert meta["vram_gb"] <= 2.5, f"3B model {m} VRAM estimate should fit in 4GB card"
            assert "description" in meta

    def test_available_models_discovers_llama_3b(self):
        models = get_available_models()
        model_ids = [m["id"] for m in models]
        assert "llama-3.2-3b-instruct-q4_k_m.gguf" in model_ids

    def test_models_endpoint_contains_3b(self):
        client = TestClient(app)
        res = client.get("/models")
        assert res.status_code == 200
        data = res.json()
        ids = [m["id"] for m in data["models"]]
        assert "llama-3.2-3b-instruct-q4_k_m.gguf" in ids


class Test3BAutoRouting:
    """Test auto-routing behavior when PREFER_3B_MODELS=true."""

    def test_prefer_3b_routing(self, monkeypatch):
        monkeypatch.setenv("PREFER_3B_MODELS", "true")

        # General prompt should route to 3B model (Llama 3.2 3B)
        gen_model = auto_select_model("Hello, what is machine learning?")
        assert "3b" in gen_model.lower()

        # Math / reasoning prompt in 3B mode should route to 3B model
        math_model = auto_select_model("calculate the square root of 144 and solve step-by-step")
        assert "3b" in math_model.lower() or "deepseek" in math_model.lower()

    def test_get_coder_model_with_prefer_3b(self, monkeypatch):
        monkeypatch.setenv("PREFER_3B_MODELS", "true")
        coder_model = get_coder_model()
        assert "3b" in coder_model.lower()
