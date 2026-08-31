"""
Step 6 Tests: RBAC, Benchmarking, Confidence Fallback
"""

import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HARDWARE_TIER", "BUILD")

from backend.main import app as fastapi_app

tc = TestClient(fastapi_app)


# ---------- RBAC Tests ----------


class TestRBAC:
    """Test role-based access control on the RAG search and /chat endpoint."""

    def _ingest_restricted_chunk(self):
        """Helper: ingest a chunk into the restricted collection."""
        from backend.tools.rag_search import get_rag

        rag = get_rag()
        rag.ingest([
            {
                "text": "Top Secret Budget: Project X costs $50 million.",
                "metadata": {"collection": "financials_restricted", "source": "budget.txt"},
            }
        ])

    def test_engineer_denied_restricted_collection(self):
        """Engineer role should NOT see financials_restricted chunks in search."""
        self._ingest_restricted_chunk()
        from backend.tools.rag_search import get_rag
        rag = get_rag()
        # Engineer search should filter out restricted chunks
        results_eng = rag.search("Top Secret Budget", top_k=10, role="engineer")
        for r in results_eng:
            assert r.get("metadata", {}).get("collection") != "financials_restricted", (
                f"Engineer saw restricted chunk: {r}"
            )

    def test_manager_sees_restricted_collection(self):
        """Manager role should see all chunks, including restricted ones."""
        self._ingest_restricted_chunk()
        from backend.tools.rag_search import get_rag
        rag = get_rag()
        # Manager search should include restricted chunks
        results_mgr = rag.search("Top Secret Budget", top_k=10, role="manager")
        found_restricted = any(
            r.get("metadata", {}).get("collection") == "financials_restricted"
            for r in results_mgr
        )
        assert found_restricted, (
            f"Manager did not see restricted chunk. Results: {[r.get('text', '')[:60] for r in results_mgr]}"
        )

    def test_chat_endpoint_accepts_role(self):
        """The /chat endpoint should accept role parameter without error."""
        resp = tc.post(
            "/chat",
            json={"prompt": "hello"},
            params={"role": "manager"},
        )
        assert resp.status_code == 200

    def test_invalid_role_rejected(self):
        """Invalid role string should return 400."""
        resp = tc.post(
            "/chat",
            json={"prompt": "hello"},
            params={"role": "hacker"},
        )
        assert resp.status_code == 400
        assert "Invalid role" in resp.json().get("detail", "")

    def test_default_role_is_engineer(self):
        """get_role() defaults to 'engineer' when no role is provided."""
        from backend.core.auth import get_role
        from starlette.testclient import TestClient as _TC
        from fastapi import FastAPI, Depends

        _app = FastAPI()

        @_app.get("/test-role")
        async def _test(role: str = Depends(get_role)):
            return {"role": role}

        _tc = _TC(_app)
        resp = _tc.get("/test-role")
        assert resp.status_code == 200
        assert resp.json()["role"] == "engineer"


# ---------- Benchmark Tests ----------


class TestBenchmark:
    """Test the /benchmark endpoint."""

    def test_benchmark_returns_200(self):
        resp = tc.get("/benchmark")
        assert resp.status_code == 200
        data = resp.json()
        assert "handwriting_word_accuracy" in data, f"Missing key in: {data}"
        assert "pid_precision" in data, f"Missing key in: {data}"

    def test_benchmark_accuracy_is_numeric(self):
        resp = tc.get("/benchmark")
        data = resp.json()
        assert isinstance(data["handwriting_word_accuracy"], (int, float))
        assert isinstance(data["pid_precision"], (int, float))

    def test_benchmark_file_written(self):
        """Benchmark should write results to docs/benchmark_results.json."""
        resp = tc.get("/benchmark")
        assert resp.status_code == 200
        results_path = PROJECT_ROOT / "docs" / "benchmark_results.json"
        assert results_path.exists(), f"Benchmark file not created at {results_path}"
        content = json.loads(results_path.read_text(encoding="utf-8"))
        assert "handwriting_word_accuracy" in content


# ---------- Confidence Fallback Tests ----------


class TestConfidenceFallback:
    """Test that low confidence triggers a warning prefix."""

    def test_low_confidence_warning(self):
        """When MockVisionModel returns low confidence, output should contain warning."""
        from unittest.mock import patch
        from backend.tools.handwriting_triage import read_note
        from pathlib import Path

        SANDBOX_DIR = PROJECT_ROOT / "workspace" / "sandbox_files"
        SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
        test_img = SANDBOX_DIR / "confidence_test.jpg"

        # Create a test image if it doesn't exist
        if not test_img.exists():
            try:
                from PIL import Image
                Image.new("RGB", (100, 100), "white").save(str(test_img))
            except ImportError:
                test_img.write_bytes(b"\xff\xd8\xff\xe0")

        # Patch get_mock_confidence to return a low value
        with patch(
            "backend.core.model_manager.MockVisionModel.get_mock_confidence",
            return_value=0.4,
        ):
            result = read_note(str(test_img))
            text = result.get("text", "")
            assert "LOW CONFIDENCE" in text or "⚠️" in text, (
                f"Expected low-confidence warning in: {text[:200]}"
            )
            assert result["confidence"] == 0.4

    def test_high_confidence_no_warning(self):
        """When confidence >= 0.6, no warning prefix should appear."""
        from unittest.mock import patch
        from backend.tools.handwriting_triage import read_note
        from pathlib import Path

        SANDBOX_DIR = PROJECT_ROOT / "workspace" / "sandbox_files"
        test_img = SANDBOX_DIR / "confidence_test.jpg"

        if not test_img.exists():
            try:
                from PIL import Image
                Image.new("RGB", (100, 100), "white").save(str(test_img))
            except ImportError:
                test_img.write_bytes(b"\xff\xd8\xff\xe0")

        with patch(
            "backend.core.model_manager.MockVisionModel.get_mock_confidence",
            return_value=0.85,
        ):
            result = read_note(str(test_img))
            text = result.get("text", "")
            assert "LOW CONFIDENCE" not in text, (
                f"Unexpected low-confidence warning at confidence=0.85: {text[:200]}"
            )


# ---------- Auth Unit Tests ----------


class TestAuthModule:
    """Direct unit tests for the auth module."""

    def test_valid_roles(self):
        from backend.core.auth import VALID_ROLES
        assert "engineer" in VALID_ROLES
        assert "manager" in VALID_ROLES

    def test_restricted_collections_defined(self):
        from backend.core.auth import RESTRICTED_COLLECTIONS
        assert "financials_restricted" in RESTRICTED_COLLECTIONS

    def test_is_restricted_engineer(self):
        from backend.core.auth import is_restricted
        assert is_restricted("financials_restricted", "engineer") is True
        assert is_restricted("financials_restricted", "manager") is False
        assert is_restricted("general_kb", "engineer") is False
