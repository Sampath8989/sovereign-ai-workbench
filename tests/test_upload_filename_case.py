"""
Regression tests for BUG 1 — case-sensitive file lookup breaking on real
uploaded filenames.

Scenario: a user uploads "Screenshot_2026-09-03_15-43-10.png" (mixed case +
underscores + numbers, exactly what macOS/Windows produce). The planner used to
lowercase the prompt before extracting the filename, so the tool call resolved
"screenshot_2026-09-03_15-43-10.png" — which does not exist on a case-sensitive
filesystem — and every lookup failed with "File not found".

These tests pin the fix: filename extraction preserves the exact case, the
vision-tool plan references the actual uploaded file (never the hardcoded
test_*.jpg/png demo files), and lookups resolve to the real file on disk.
"""

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure project root on path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents.executor import reset_artifact_tracking
from backend.core.model_manager import MockLLM
from backend.tools.file_io import read_file

SANDBOX_DIR = PROJECT_ROOT / "workspace" / "sandbox_files"
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

# The OS-default, mixed-case filename from the bug report.
UPLOAD_FNAME = "Screenshot_2026-09-03_15-43-10.png"
UPLOAD_PATH = SANDBOX_DIR / UPLOAD_FNAME

# Planner system prompt used by planner.generate_plan (subset is enough to
# trigger the MockLLM planning branches exercised here).
_PLAN_PROMPT = (
    "You are a task planner. Given a user request, produce a JSON array of steps. "
    "Each step is an object with keys: \"tool\", \"action\", \"args\". "
    "Output ONLY the JSON array."
)


@pytest.fixture(autouse=True)
def uploaded_screenshot():
    """Simulate an upload of the mixed-case screenshot into the sandbox."""
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    # Use Pillow when available for a real PNG; otherwise plain bytes are fine —
    # the mock vision tools only require the file to exist.
    try:
        from PIL import Image
        img = Image.new("RGB", (64, 48), "white")
        img.save(str(UPLOAD_PATH))
    except Exception:
        UPLOAD_PATH.write_bytes(b"\x89PNG\r\n\x1a\nfake-screenshot-bytes-2026")
    yield
    UPLOAD_PATH.unlink(missing_ok=True)


def _plan_for(prompt: str):
    """Return the parsed plan MockLLM produces for a prompt in planning context."""
    llm = MockLLM()
    res = llm.create_chat_completion(
        [
            {"role": "system", "content": _PLAN_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    text = res["choices"][0]["text"]
    payload = json.loads(text)
    return payload["plan"] if isinstance(payload, dict) else payload


class TestUploadedFilenameCasePreserved:
    """The filename referenced in the prompt must never be lowercased."""

    def test_read_plan_preserves_exact_case(self):
        plan = _plan_for(f"Analyze uploaded file: {UPLOAD_FNAME}")
        assert plan, "Expected a non-empty plan"
        assert plan[0]["tool"] == "file_io"
        captured = plan[0]["args"][0]
        assert captured == UPLOAD_FNAME, (
            f"Plan filename was lowercased/mangled: {captured!r} != {UPLOAD_FNAME!r}"
        )

    def test_read_plan_never_lowercases_screenshot(self):
        """Directly reproduce the bug report: the plan arg must keep the exact
        on-disk case ("Screenshot_2026-09-03_15-43-10.png"), not the
        lowercased "screenshot_2026-09-03_15-43-10.png" that failed lookup."""
        plan = _plan_for(f"Analyze uploaded file: {UPLOAD_FNAME}")
        captured = plan[0]["args"][0]
        assert captured == UPLOAD_FNAME
        assert captured.lower() != captured, "Captured filename should retain mixed case"


class TestVisionToolResolvesUploadedFile:
    """Vision-tool routing must use the actual uploaded file, exact case."""

    def test_photo_analyzer_plan_points_at_uploaded_file(self):
        plan = _plan_for(f"Analyze the nameplate in the field photo {UPLOAD_FNAME}")
        assert plan[0]["tool"] == "photo_analyzer"
        arg = plan[0]["args"][0]
        assert arg == f"workspace/sandbox_files/{UPLOAD_FNAME}", (
            f"photo_analyzer did not receive the uploaded file path: {arg!r}"
        )

    def test_handwriting_plan_points_at_uploaded_file(self):
        plan = _plan_for(f"Read the handwriting from the note {UPLOAD_FNAME}")
        assert plan[0]["tool"] == "handwriting_triage"
        arg = plan[0]["args"][0]
        assert arg == f"workspace/sandbox_files/{UPLOAD_FNAME}"

    def test_pid_plan_points_at_uploaded_file(self):
        plan = _plan_for(f"Extract topology from the P&ID {UPLOAD_FNAME}")
        assert plan[0]["tool"] == "pid_extractor"
        arg = plan[0]["args"][0]
        assert arg == f"workspace/sandbox_files/{UPLOAD_FNAME}"

    def test_photo_analyzer_runs_on_uploaded_file(self):
        """End-to-end vision tool call: the exact mixed-case path resolves and
        the analyzer reports the real (exact-case) source filename."""
        from backend.tools.photo_analyzer import analyze_nameplate

        result = analyze_nameplate(f"workspace/sandbox_files/{UPLOAD_FNAME}")
        assert result.get("status") != "no_match_found", f"Vision lookup failed: {result}"
        assert result.get("source") == UPLOAD_FNAME, (
            f"Vision tool resolved the wrong file: {result.get('source')!r}"
        )
        assert result.get("model") not in ("unknown", "")

    def test_handwriting_and_pid_run_on_uploaded_file(self):
        from backend.tools.handwriting_triage import read_note
        from backend.tools.pid_extractor import extract_topology

        note = read_note(f"workspace/sandbox_files/{UPLOAD_FNAME}")
        assert note.get("status") != "no_match_found", f"Handwriting lookup failed: {note}"
        assert note.get("source") == UPLOAD_FNAME

        graph = extract_topology(f"workspace/sandbox_files/{UPLOAD_FNAME}")
        assert "no_match_found" not in json.dumps(graph)


class TestLookupResolvesRealOnDiskName:
    """Even a case-variant reference must resolve to the real file on disk."""

    def test_read_file_resolves_case_variant_to_real_file(self):
        content = read_file(UPLOAD_FNAME.lower())
        assert "File not found" not in content, f"Lowercase lookup failed: {content[:80]}"
        assert len(content) > 0

    def test_exact_case_wins_when_ambiguous(self):
        """When two files differ only by case, the exact name must win."""
        twin = SANDBOX_DIR / UPLOAD_FNAME.lower()
        try:
            twin.write_bytes(b"TWIN-ONLY-MARKER-123456789")
            content_exact = read_file(UPLOAD_FNAME)
            assert "File not found" not in content_exact
            # The exact-case upload was read, never the case-variant twin.
            assert "TWIN-ONLY-MARKER-123456789" not in content_exact
        finally:
            twin.unlink(missing_ok=True)


class TestImageReadReturnsGuidance:
    """Binary image content must never be decoded into token-exploding text
    ("Requested tokens (31520) exceed context window of 2048")."""

    def test_read_png_returns_guidance_not_binary(self):
        result = read_file(UPLOAD_FNAME)
        assert "Image file" in result
        assert UPLOAD_FNAME in result
        # Decoded binary garbage must not reach any prompt.
        assert "\x89PNG" not in result

    def test_read_unknown_binary_extension_returns_guidance(self):
        blob = SANDBOX_DIR / "snippet.rtf"
        blob.write_bytes(b"{\\rtf1\\ansi\\b\\f0\\b Test}"[:0] + b"abc\x00def" + b"\x00" * 100)
        try:
            result = read_file("snippet.rtf")
            assert "Binary file" in result
        finally:
            blob.unlink(missing_ok=True)


class TestChatPipelineNoFileNotFound:
    """End-to-end /chat flow: uploading the screenshot then asking about it must
    never produce the lowercase 'File not found' error from the bug report."""

    def test_chat_analyze_uploaded_screenshot(self):
        from backend.agents.graph import app

        reset_artifact_tracking()
        result = app.invoke(
            {
                "input": f"Analyze uploaded file: {UPLOAD_FNAME}",
                "role": "admin",
                "selected_model": "mock",
            }
        )
        output = result.get("output", "")
        assert "No match found" not in output
        assert "File not found" not in output
        assert "screenshot_2026-09-03_15-43-10" not in output

    def test_chat_vision_analysis_of_uploaded_screenshot(self):
        from backend.agents.graph import app

        reset_artifact_tracking()
        result = app.invoke(
            {
                "input": f"Analyze the nameplate in the field photo {UPLOAD_FNAME}",
                "role": "admin",
                "selected_model": "mock",
            }
        )
        output = result.get("output", "")
        assert "No match found" not in output
        assert "File not found" not in output
        assert "no_match_found" not in output
        plan_tools = [s.get("tool") for s in result.get("plan", [])]
        assert "photo_analyzer" in plan_tools
