"""
Regression tests for the vision/LLaVA "500 with no detail" bug.

Real GGUF inference (e.g. LLaVA-7B Q4, ~3.9 GB on disk, offloaded to a 4 GB
GPU with ~3.7 GB free) can die mid-generation with a llama.cpp
RuntimeError("CUDA error: out of memory ..."). Previously that exception was
re-raised through the chat endpoint as a bare HTTP 500 and the frontend showed
only axios's generic message.

These tests pin the fix:
  - OOM exceptions are detected, logged with a full traceback, and re-raised as
    ModelInferenceError("vision_model_oom" | "model_oom", <readable detail>);
  - the GPU offload count is clamped to what the currently-free VRAM can hold;
  - the chat API maps ModelInferenceError to HTTP 503 with the structured body
    {"error": ..., "detail": ...} that the frontend displays.
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.model_manager import ModelInferenceError, ModelManager

MODELS_DIR = PROJECT_ROOT / "models"
LLAVA_PATH = str(MODELS_DIR / "llava-7b.gguf")


class _FakeRealModel:
    """Non-MockLLM stand-in for a loaded GGUF handle whose inference fails."""

    def __init__(self, exc: Exception, generation_calls=None):
        self.exc = exc
        self.generation_calls = generation_calls if generation_calls is not None else []

    def create_completion(self, prompt, **kwargs):
        self.generation_calls.append(("completion", kwargs.get("max_tokens")))
        raise self.exc

    def create_chat_completion(self, messages=None, **kwargs):
        self.generation_calls.append(("chat", kwargs.get("max_tokens")))
        raise self.exc


def _cuda_oom() -> RuntimeError:
    return RuntimeError(
        "CUDA error: out of memory (ggml_cuda: failed to allocate "
        "1024.00 MiB on device 0)"
    )


class TestCudaOomBecomesTypedError:
    def test_llava_oom_maps_to_vision_model_oom(self, monkeypatch):
        calls = []
        model = _FakeRealModel(_cuda_oom(), generation_calls=calls)
        mm = ModelManager()
        monkeypatch.setattr(mm, "load_model", lambda name, reject_oversized=False: model)

        with pytest.raises(ModelInferenceError) as exc_info:
            mm.generate("llava-7b.gguf", "describe this photo", max_tokens=64)

        err = exc_info.value
        assert err.code == "vision_model_oom"
        assert "llava-7b.gguf" in err.message
        assert "CUDA out of memory" in err.message
        assert "VRAM" in err.message or "GB" in err.message

    def test_non_vision_model_oom_uses_generic_code(self, monkeypatch):
        model = _FakeRealModel(_cuda_oom())
        mm = ModelManager()
        monkeypatch.setattr(mm, "load_model", lambda name, reject_oversized=False: model)

        with pytest.raises(ModelInferenceError) as exc_info:
            mm.generate("qwen2.5-7b-instruct-q3_k_m.gguf", "hi", max_tokens=32)
        assert exc_info.value.code == "model_oom"

    def test_chat_completion_oom_does_not_retry_via_prompt_fallback(self, monkeypatch):
        calls = []
        model = _FakeRealModel(_cuda_oom(), generation_calls=calls)
        mm = ModelManager()
        monkeypatch.setattr(mm, "load_model", lambda name, reject_oversized=False: model)

        with pytest.raises(ModelInferenceError):
            mm.generate_from_messages(
                "llava-7b.gguf",
                [{"role": "user", "content": "what is in this image?"}],
                max_tokens=64,
            )
        # OOM must not trigger the secondary prompt-completion attempt.
        assert len(calls) == 1, f"Expected no fallback retry after OOM, calls={calls}"

    def test_non_oom_error_re_raised_unwrapped(self, monkeypatch):
        model = _FakeRealModel(RuntimeError("some other llama failure"))
        mm = ModelManager()
        monkeypatch.setattr(mm, "load_model", lambda name, reject_oversized=False: model)

        with pytest.raises(RuntimeError):
            mm.generate("qwen2.5-7b-instruct-q3_k_m.gguf", "hi", max_tokens=32)


class TestContextOverflowHandling:
    """Prompt longer than n_ctx must surface as a typed, actionable error and
    the summarize step must truncate content before it reaches the model."""

    def test_context_overflow_maps_to_typed_error(self, monkeypatch):
        calls = []
        exc = ValueError("Requested tokens (31520) exceed context window of 2048")
        model = _FakeRealModel(exc, generation_calls=calls)
        mm = ModelManager()
        monkeypatch.setattr(mm, "load_model", lambda name, reject_oversized=False: model)

        with pytest.raises(ModelInferenceError) as exc_info:
            mm.generate_from_messages(
                "llava-7b.gguf",
                [{"role": "user", "content": "x" * 90000}],
                max_tokens=64,
            )
        assert exc_info.value.code == "context_overflow"
        assert "context window" in exc_info.value.message
        # No pointless retry through the prompt-completion fallback.
        assert len(calls) == 1

    def test_summarize_step_caps_context_before_model(self, monkeypatch):
        from backend.agents.executor import execute_step, reset_artifact_tracking

        reset_artifact_tracking()
        seen = {}

        class _Recorder:
            """Stands in for ModelManager; records what summarize would send."""

            def load_model(self, name, reject_oversized=False):
                # Not a MockLLM => executor takes the real-model path and calls
                # model_manager.generate_from_messages with the messages it built.
                return object()

            def generate_from_messages(self, model_name, messages, **kwargs):
                seen["user_content"] = messages[-1]["content"]
                return "ok"

        mm = _Recorder()
        # A 200k-char context (as if binary had been decoded to text).
        context = {"step_0_result": "raw output " + "A" * 200000}
        step = {"tool": "llm", "action": "summarize", "args": []}
        result = execute_step(step, context, mm)
        assert "ok" in result
        assert len(seen["user_content"]) <= 3200, (
            f"Summarize sent {len(seen['user_content'])} chars to the model"
        )
        reset_artifact_tracking()


class TestVramFitOffload:
    def test_llava_7b_q4_capped_on_4gb_card(self):
        # llava-7b Q4 is ~3.9 GB; with 3.7 GB free the old fixed 26-layer
        # offload would OOM. The fitter must return fewer layers (or 0).
        layers = ModelManager._fit_gpu_layers(LLAVA_PATH, 26, free_vram_gb=3.7)
        assert 0 <= layers < 26, f"Expected reduced offload, got {layers}"

    def test_ample_vram_allows_planned_offload(self):
        layers = ModelManager._fit_gpu_layers(LLAVA_PATH, 26, free_vram_gb=8.0)
        assert layers == 26

    def test_full_offload_flag_clamped_to_total_layers(self):
        # planned -1 means "full offload"; never return more layers than ~32.
        layers = ModelManager._fit_gpu_layers(LLAVA_PATH, -1, free_vram_gb=8.0)
        assert 0 < layers <= 32

    def test_tiny_free_vram_returns_cpu(self):
        layers = ModelManager._fit_gpu_layers(LLAVA_PATH, 26, free_vram_gb=0.4)
        assert layers == 0

    def test_unknown_vram_leaves_planned_untouched(self):
        layers = ModelManager._fit_gpu_layers(LLAVA_PATH, 26, free_vram_gb=None)
        assert layers == 26


class TestApiReturnsStructuredError:
    def test_chat_endpoint_maps_model_error_to_503_body(self, monkeypatch):
        from fastapi.testclient import TestClient
        import backend.agents.graph as graph_mod
        from backend.main import app

        def boom(*args, **kwargs):
            raise ModelInferenceError(
                "vision_model_oom",
                "llava-7b.gguf could not run: CUDA out of memory while generating. "
                "Model needs ~3.92 GB on disk; only ~3.70 GB VRAM free.",
            )

        monkeypatch.setattr(graph_mod.app, "invoke", boom)

        client = TestClient(app)
        res = client.post(
            "/chat",
            json={
                "prompt": "Analyze the nameplate in the field photo Screenshot_2026-09-03_15-43-10.png",
                "model": "auto",
            },
            params={"role": "engineer"},
        )
        assert res.status_code == 503
        body = res.json().get("detail", {})
        assert body.get("error") == "vision_model_oom"
        assert "CUDA out of memory" in body.get("detail", "")
