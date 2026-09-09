"""
Regression tests for Multimodal Vision Pipeline in Sovereign AI Workbench.
Tests:
1. End-to-end multimodal pipeline: uploaded image -> base64 encoding -> OpenAI-style image_url payload -> model call
2. Outgoing payload logging with format, prefix, and base64 length
3. Hard failure (HTTP 400 + image_attach_failed) when image is missing or empty
4. ModelManager singleton: resident models correctly reported by GET /health and GET /models
5. Outgoing request payload verification before inference dispatch
"""

import base64
import logging
import sys
from pathlib import Path
from PIL import Image

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.model_manager import ModelInferenceError, get_model_manager, MockLLM
from backend.main import app

SANDBOX_DIR = PROJECT_ROOT / "workspace" / "sandbox_files"
TEST_IMAGE_NAME = "test_multimodal_sample.png"


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture
def sample_png():
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    img_path = SANDBOX_DIR / TEST_IMAGE_NAME
    img = Image.new("RGB", (32, 32), color="blue")
    img.save(str(img_path))
    yield img_path
    img_path.unlink(missing_ok=True)


class _CaptureModel:
    """Mock model that records the incoming messages payload."""

    def __init__(self):
        self.captured_messages = None
        self.captured_kwargs = None

    def create_chat_completion(self, messages=None, **kwargs):
        self.captured_messages = messages
        self.captured_kwargs = kwargs
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The image shows a solid blue square.",
                    }
                }
            ]
        }

    def create_completion(self, prompt, **kwargs):
        return {"choices": [{"text": "fallback text"}]}


class TestMultimodalPipeline:
    def test_multimodal_image_reaches_model_call(self, client, sample_png, caplog, monkeypatch):
        """Confirm the image is read from disk, base64-encoded, and passed as OpenAI-format image_url."""
        caplog.set_level(logging.INFO)
        capture_model = _CaptureModel()

        mm = get_model_manager()
        monkeypatch.setattr(mm, "load_model", lambda name, reject_oversized=False: capture_model)

        res = client.post(
            "/chat",
            json={
                "prompt": f"Analyze uploaded file: {TEST_IMAGE_NAME}",
                "model": "llava-7b.gguf",
            },
            params={"role": "engineer"},
        )

        assert res.status_code == 200, res.text
        data = res.json()
        assert "blue square" in data["response"].lower()

        # 1. Check captured messages received by the model
        assert capture_model.captured_messages is not None
        user_msg = next(
            (m for m in capture_model.captured_messages if m.get("role") == "user"), None
        )
        assert user_msg is not None
        content = user_msg.get("content")
        assert isinstance(content, list), f"Expected list content for multimodal call, got {type(content)}"

        # Verify text block and image_url block
        has_text = any(item.get("type") == "text" for item in content)
        image_block = next((item for item in content if item.get("type") == "image_url"), None)
        assert has_text, "Missing text part in content"
        assert image_block is not None, "Missing image_url block in content"

        url = image_block["image_url"]["url"]
        assert url.startswith("data:image/png;base64,"), f"Unexpected URL prefix: {url[:30]}"
        b64_part = url.split("data:image/png;base64,")[1]
        decoded = base64.b64decode(b64_part)
        assert len(decoded) > 0

        # 2. Check logging of outgoing multimodal payload
        payload_records = [
            r.getMessage() for r in caplog.records if "[MULTIMODAL PAYLOAD OUTGOING]" in r.getMessage()
        ]
        assert payload_records, "No [MULTIMODAL PAYLOAD OUTGOING] log record was emitted"
        log_text = payload_records[0]
        assert "llava-7b.gguf" in log_text
        assert "Base64 length" in log_text or "base64" in log_text.lower()

    def test_hard_failure_on_missing_image(self, client):
        """Non-existent image file must fail immediately with HTTP 400 and code image_attach_failed."""
        res = client.post(
            "/chat",
            json={
                "prompt": "Analyze uploaded file: completely_nonexistent_image_12345.png",
                "model": "llava-7b.gguf",
            },
            params={"role": "engineer"},
        )
        assert res.status_code == 400, f"Expected 400 for missing image, got {res.status_code}: {res.text}"
        body = res.json().get("detail", {})
        assert body.get("error") == "image_attach_failed"
        assert "not found" in body.get("detail", "").lower()

    def test_hard_failure_on_empty_image(self, client):
        """0-byte image file must fail with HTTP 400 and code image_attach_failed."""
        empty_name = "zero_byte_sample.png"
        empty_path = SANDBOX_DIR / empty_name
        empty_path.write_bytes(b"")

        try:
            res = client.post(
                "/chat",
                json={
                    "prompt": f"Analyze uploaded file: {empty_name}",
                    "model": "llava-7b.gguf",
                },
                params={"role": "engineer"},
            )
            assert res.status_code == 400, f"Expected 400 for empty image, got {res.status_code}: {res.text}"
            body = res.json().get("detail", {})
            assert body.get("error") == "image_attach_failed"
            assert "empty" in body.get("detail", "").lower()
        finally:
            empty_path.unlink(missing_ok=True)

    def test_vram_panel_and_health_reports_resident_model(self, client):
        """Verify GET /health and GET /models report loaded models via shared ModelManager singleton."""
        mm = get_model_manager()
        # Mock load a model into resident_models
        fake_model = _CaptureModel()
        mm.resident_models["llava-7b.gguf"] = fake_model
        mm.vram_usage["llava-7b.gguf"] = 3.9

        try:
            health_res = client.get("/health")
            assert health_res.status_code == 200
            health_data = health_res.json()
            resident_in_health = health_data.get("resident_models", {}).get("resident_models", {})
            assert "llava-7b.gguf" in resident_in_health, (
                f"VRAM panel /health did not reflect resident model: {resident_in_health}"
            )

            models_res = client.get("/models")
            assert models_res.status_code == 200
            models_data = models_res.json()
            assert "llava-7b.gguf" in models_data.get("resident_models", [])
        finally:
            mm.resident_models.pop("llava-7b.gguf", None)
            mm.vram_usage.pop("llava-7b.gguf", None)
