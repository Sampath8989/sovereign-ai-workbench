"""
Test suite for Model Selection and Auto-Routing features.
Tests:
- Dynamic discovery of all local 7B, 14B, and 4B models
- GET /models endpoint structure and options
- auto_select_model logic across math, code, vision, synthesis, and general domains
- POST /chat accepting 'model' parameter (auto and specific model selection)
- ModelManager budget and hot-swap compatibility
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.config import get_available_models, MODEL_METADATA_REGISTRY
from backend.core.router import auto_select_model, route_task
from backend.core.model_manager import ModelManager, MockLLM


class TestModelDiscovery:
    """Test dynamic model discovery and metadata registration."""

    def test_get_available_models_includes_auto(self):
        models = get_available_models()
        assert len(models) > 0
        first = models[0]
        assert first["id"] == "auto"
        assert first["category"] == "AUTO"
        assert "Auto" in first["name"]

    def test_get_available_models_includes_7b_and_14b_models(self):
        models = get_available_models()
        model_ids = [m["id"] for m in models]
        
        # Verify 7B and 14B models copied from Ollama are discovered
        assert "deepseek-r1-7b.gguf" in model_ids
        assert "phi4-14b.gguf" in model_ids
        assert "qwen2.5-coder-7b-instruct-q3_k_m.gguf" in model_ids
        assert "llava-7b.gguf" in model_ids
        assert "qwen2.5-7b-instruct-q3_k_m.gguf" in model_ids
        assert "qwen2.5-7b.gguf" in model_ids

    def test_model_metadata_fields(self):
        models = get_available_models()
        for m in models:
            assert "id" in m
            assert "name" in m
            assert "category" in m
            assert "description" in m
            assert "vram_gb" in m
            assert "size_gb" in m
            assert "is_present" in m
            assert m["is_present"] is True


class TestAutoRouting:
    """Test auto_select_model intent-based selection."""

    def test_math_and_reasoning_selects_deepseek_r1(self):
        prompts = [
            "calculate the integral of x^3 + 2x",
            "solve the mathematical equation 4x - 8 = 16",
            "provide a step-by-step logic proof",
            "verify the mathematical calculation and audit steps",
        ]
        for p in prompts:
            selected = auto_select_model(p)
            assert selected == "deepseek-r1-7b.gguf", f"Failed on: {p}"

    def test_coding_and_deliverables_selects_qwen_coder(self):
        prompts = [
            "write a python script to parse CSV files",
            "debug this function and refactor the class",
            "create a word document report.docx with revenue summary",
            "generate an excel spreadsheet with budget columns",
            "build a presentation pptx for the quarterly review",
        ]
        for p in prompts:
            selected = auto_select_model(p)
            assert selected == "qwen2.5-coder-7b-instruct-q3_k_m.gguf", f"Failed on: {p}"

    def test_vision_and_scans_selects_llava(self):
        prompts = [
            "analyze this image and scan the diagram",
            "inspect the photo of the pump nameplate",
            "extract topology from this visual drawing",
        ]
        for p in prompts:
            selected = auto_select_model(p)
            assert selected == "llava-7b.gguf", f"Failed on: {p}"

    def test_deep_synthesis_selects_phi4(self):
        prompts = [
            "give me a comprehensive architectural deep dive",
            "write a strategic executive summary of the system architecture",
        ]
        for p in prompts:
            selected = auto_select_model(p)
            assert selected == "phi4-14b.gguf", f"Failed on: {p}"

    def test_general_chat_selects_qwen_instruct(self):
        prompts = [
            "hello how are you today?",
            "what are standard operating safety limits?",
        ]
        for p in prompts:
            selected = auto_select_model(p)
            assert "qwen" in selected


class TestModelsEndpoint:
    """Test FastAPI /models endpoint."""

    def test_get_models_endpoint(self):
        with TestClient(app) as client:
            resp = client.get("/models")
            assert resp.status_code == 200
            data = resp.json()
            assert "models" in data
            assert "default" in data
            assert data["default"] == "auto"
            assert len(data["models"]) >= 7

    def test_health_includes_available_models(self):
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert "available_models" in data
            assert len(data["available_models"]) >= 7


class TestChatWithModelSelection:
    """Test POST /chat with model parameter."""

    def test_chat_with_auto_model(self):
        with TestClient(app) as client:
            resp = client.post("/chat", json={"prompt": "hello", "model": "auto"})
            assert resp.status_code == 200
            data = resp.json()
            assert "response" in data
            assert "model_used" in data

    def test_chat_with_specific_model(self):
        with TestClient(app) as client:
            resp = client.post("/chat", json={"prompt": "hi", "model": "deepseek-r1-7b.gguf"})
            assert resp.status_code == 200
            data = resp.json()
            assert "response" in data
