"""
End-to-end regression for the file-upload + LLaVA "Request failed with status
code 500 (no detail)" bug.

Reproduces the real user flow at the HTTP boundary:
  1. POST /upload stores a small PNG under its exact, mixed-case filename
     (``Screenshot_2026-09-03_15-43-10.png``), mirroring what the OS produces.
  2. POST /chat asks the workbench to analyze that uploaded photo with
     ``model="llava-7b.gguf"`` (what the UI does when the user forces the
     vision model). The loaded real GGUF model dies mid-generation with the
     exact llama.cpp CUDA-OOM RuntimeError seen on a 4 GB GPU with ~3.7 GB
     free.

What this pins:
  - the API answers HTTP 503 with a structured ``{"error": "vision_model_oom",
    "detail": ...}`` body the frontend can render — never a bare 500;
  - the underlying CUDA exception is logged WITH its full traceback
    (``exc_info=True``), so the real cause is visible in stderr AND the
    rotating log file (backend/main.py file logging);
  - a non-OOM inference crash likewise returns a readable body instead of a
    bare 500, with the traceback preserved in the log.
"""

import logging
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.model_manager import ModelInferenceError
from backend.main import app

SANDBOX_DIR = PROJECT_ROOT / "workspace" / "sandbox_files"
MIXED_CASE_NAME = "Screenshot_2026-09-03_15-43-10.png"
VISION_PROMPT = f"Analyze the nameplate in the uploaded field photo {MIXED_CASE_NAME}"


def _cuda_oom() -> RuntimeError:
    """The exact llama.cpp CUDA-OOM message from a real GPU build."""
    return RuntimeError(
        "CUDA error: out of memory (ggml_cuda: failed to allocate "
        "1024.00 MiB on device 0)"
    )


class _FakeRealModel:
    """Non-MockLLM stand-in for a loaded GGUF handle whose inference fails."""

    def __init__(self, exc: Exception):
        self.exc = exc

    def create_completion(self, prompt, **kwargs):
        raise self.exc

    def create_chat_completion(self, messages=None, **kwargs):
        raise self.exc


@pytest.fixture(scope="module")
def client():
    """TestClient WITHOUT the lifespan context (no RAG/sentinel needed)."""
    return TestClient(app)


@pytest.fixture()
def uploaded_png(client):
    """Upload a real small PNG under its exact mixed-case name."""
    try:
        from PIL import Image
        import io

        buf = io.BytesIO()
        Image.new("RGB", (64, 48), "white").save(buf, format="PNG")
        content = buf.getvalue()
    except Exception:
        content = b"\x89PNG\r\n\x1a\n" + b"0" * 256

    res = client.post(
        f"/upload?target_filename={MIXED_CASE_NAME}",
        files={"file": ("upload.bin", content, "image/png")},
    )
    assert res.status_code == 200, res.text
    on_disk = SANDBOX_DIR / MIXED_CASE_NAME
    assert on_disk.is_file(), f"Uploaded file missing at {on_disk}"
    yield on_disk
    on_disk.unlink(missing_ok=True)


@pytest.fixture()
def graph_manager_with_oom(monkeypatch):
    """Point the agent graph at a ModelManager whose real model raises a
    CUDA OOM during generation — the exact failure the user hit on a 4 GB
    GPU with ~3.7 GB free."""
    return _install_graph_manager(monkeypatch, _cuda_oom())


def _install_graph_manager(monkeypatch, exc: Exception):
    """Point the agent graph at a REAL ModelManager whose load_model returns a
    non-Mock GGUF handle that raises ``exc`` during generation. Using the real
    manager exercises the actual OOM-wrapping / fallback code paths in
    ModelManager.generate_from_messages / generate."""
    from backend.core.model_manager import ModelManager
    import backend.agents.graph as graph_mod

    mm = ModelManager()

    def _load(name, reject_oversized=False):
        return _FakeRealModel(exc)

    monkeypatch.setattr(mm, "load_model", _load)
    monkeypatch.setattr(graph_mod, "_get_model_manager", lambda: mm)
    return mm


def _post_chat(client, prompt: str, model: str = "llava-7b.gguf"):
    return client.post(
        "/chat",
        json={"prompt": prompt, "model": model},
        params={"role": "engineer"},
    )


class TestUploadThenLlavaOom:
    def test_oom_returns_structured_503_not_bare_500(
        self, client, uploaded_png, graph_manager_with_oom, caplog
    ):
        caplog.set_level(logging.ERROR)

        res = _post_chat(client, VISION_PROMPT, model="llava-7b.gguf")

        # The bug: generic "Request failed with status code 500" with no body.
        # The fix: a structured 503 the frontend displays verbatim.
        assert res.status_code == 503, res.text
        body = res.json().get("detail", {})
        assert body.get("error") == "vision_model_oom"
        detail = body.get("detail", "")
        assert "llava-7b.gguf" in detail
        assert "CUDA out of memory" in detail

        # Step 1 requirement: the REAL underlying exception is visible — the
        # traceback of the CUDA OOM must be logged (stderr + log file), never
        # swallowed into a message-less 500.
        oom_records = [
            r
            for r in caplog.records
            if "cuda error: out of memory" in r.getMessage().lower()
        ]
        assert oom_records, (
            "No log record carried the CUDA OOM message. Records:\n"
            + "\n".join(r.getMessage() for r in caplog.records)
        )
        assert any(r.exc_info for r in oom_records), (
            "The CUDA OOM log record was emitted WITHOUT a traceback "
            "(exc_info not set) — the real error would still be invisible."
        )

    def test_non_oom_inference_crash_returns_readable_body(
        self, client, uploaded_png, monkeypatch, caplog
    ):
        """A non-OOM inference crash must also surface a readable detail +
        traceback log rather than an empty 500."""
        _install_graph_manager(
            monkeypatch, RuntimeError("ggml internal error: tensor mismatch")
        )
        caplog.set_level(logging.ERROR)

        res = _post_chat(client, VISION_PROMPT, model="llava-7b.gguf")

        assert res.status_code == 500, res.text
        detail = res.json().get("detail", "")
        assert "tensor mismatch" in detail
        crash_records = [
            r
            for r in caplog.records
            if "tensor mismatch" in r.getMessage()
        ]
        assert crash_records and any(r.exc_info for r in crash_records)


class TestTracebackPersistsToLogFile:
    def test_backend_log_file_handler_is_active(self):
        """The server must write logs (incl. tracebacks) to a file, so a 500
        seen in the UI can always be traced after the fact."""
        import logging
        from logging.handlers import RotatingFileHandler

        root = logging.getLogger()
        file_handlers = [
            h
            for h in root.handlers
            if isinstance(h, RotatingFileHandler)
            and str(h.baseFilename).endswith(("backend.log",))
        ]
        assert file_handlers, "No RotatingFileHandler attached to the root logger"
        log_path = Path(file_handlers[0].baseFilename)
        assert log_path.parent.is_dir(), f"Log directory missing: {log_path.parent}"
