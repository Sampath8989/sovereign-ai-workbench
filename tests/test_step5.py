"""
Step 5 Tests: Multimodal & Engineering Innovations
(P&ID Extractor, Handwriting Triage, Photo Analyzer)
"""

import json
import os
import sys
from pathlib import Path

import pytest
import requests

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SANDBOX_DIR = PROJECT_ROOT / "workspace" / "sandbox_files"
BASE_URL = "http://127.0.0.1:8000"


# ---------- Setup ----------

@pytest.fixture(scope="module", autouse=True)
def create_test_images():
    """Create dummy test images for the multimodal tools."""
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")

    # P&ID test image
    img = Image.new("RGB", (100, 100), "white")
    img.save(str(SANDBOX_DIR / "test_pid.png"))

    # Handwriting test image
    img = Image.new("RGB", (150, 100), (240, 240, 240))
    img.save(str(SANDBOX_DIR / "test_note.jpg"))

    # Photo test image
    img = Image.new("RGB", (200, 150), (200, 220, 255))
    img.save(str(SANDBOX_DIR / "test_photo.jpg"))

    yield

    # Cleanup
    for f in ["test_pid.png", "test_note.jpg", "test_photo.jpg",
              "_preprocessed_test_note.png", "_crop_10_10.png",
              "_crop_60_20.png", "_crop_140_10.png"]:
        p = SANDBOX_DIR / f
        if p.exists():
            p.unlink()


# ---------- Direct Tool Tests ----------

class TestPidExtractorDirect:
    """Test P&ID extractor directly (no server needed)."""

    def test_extract_topology(self):
        from backend.tools.pid_extractor import extract_topology

        result = extract_topology(str(SANDBOX_DIR / "test_pid.png"))
        assert "nodes" in result, f"Missing 'nodes' in result: {result}"
        assert "edges" in result, f"Missing 'edges' in result: {result}"
        assert len(result["nodes"]) > 0, "No nodes detected"
        # Check that mock YOLO returned known equipment classes
        node_types = [n["type"] for n in result["nodes"]]
        assert "valve" in node_types or "pump" in node_types, f"Unexpected types: {node_types}"

    def test_extract_topology_returns_json(self):
        from backend.tools.pid_extractor import extract_topology

        result = extract_topology(str(SANDBOX_DIR / "test_pid.png"))
        serialized = json.dumps(result)
        assert "V-101" in serialized or "valve" in serialized


class TestHandwritingDirect:
    """Test handwriting triage directly."""

    def test_read_note(self):
        from backend.tools.handwriting_triage import read_note

        result = read_note(str(SANDBOX_DIR / "test_note.jpg"))
        assert "text" in result, f"Missing 'text' in result: {result}"
        assert "confidence" in result, f"Missing 'confidence' in result: {result}"
        assert result["confidence"] >= 0.0
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0


class TestPhotoAnalyzerDirect:
    """Test photo analyzer directly."""

    def test_analyze_nameplate(self):
        from backend.tools.photo_analyzer import analyze_nameplate

        result = analyze_nameplate(str(SANDBOX_DIR / "test_photo.jpg"))
        assert "model" in result, f"Missing 'model' in result: {result}"
        assert "serial" in result, f"Missing 'serial' in result: {result}"
        assert result["model"] != ""
        assert result["serial"] != ""


# ---------- MockVisionModel Tests ----------

class TestMockVisionModel:
    """Test MockVisionModel directly."""

    def test_pid_prompt(self):
        from backend.core.model_manager import MockVisionModel
        vm = MockVisionModel()
        result = vm.analyze_image("test.png", "Extract topology from P&ID")
        assert "V-101" in result or "nodes" in result

    def test_handwriting_prompt(self):
        from backend.core.model_manager import MockVisionModel
        vm = MockVisionModel()
        result = vm.analyze_image("test.jpg", "Transcribe handwritten text")
        assert "Mock handwritten text" in result
        assert "Pressure" in result

    def test_nameplate_prompt(self):
        from backend.core.model_manager import MockVisionModel
        vm = MockVisionModel()
        result = vm.analyze_image("test.jpg", "Extract equipment nameplate data")
        assert "Model:" in result
        assert "Serial:" in result


# ---------- Server Integration Tests ----------

@pytest.fixture(scope="module")
def server_running():
    """Check if the FastAPI server is running."""
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            yield True
        else:
            pytest.skip("FastAPI server not running")
    except requests.ConnectionError:
        pytest.skip("FastAPI server not running")


@pytest.mark.usefixtures("server_running")
class TestPidViaChat:
    """Test P&ID extraction through /chat endpoint."""

    def test_chat_pid_extraction(self):
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"prompt": "Extract the topology from the P&ID at workspace/sandbox_files/test_pid.png."},
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        response_text = data["response"]
        # The response should reference V-101 or nodes (from MockVisionModel)
        assert "V-101" in response_text or "nodes" in response_text or "valve" in response_text


@pytest.mark.usefixtures("server_running")
class TestHandwritingViaChat:
    """Test handwriting triage through /chat endpoint."""

    def test_chat_handwriting(self):
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"prompt": "Read the handwriting at workspace/sandbox_files/test_note.jpg."},
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        response_text = data["response"]
        assert "Mock handwritten text" in response_text or "confidence" in response_text or "Pressure" in response_text


@pytest.mark.usefixtures("server_running")
class TestPhotoViaChat:
    """Test photo analysis through /chat endpoint."""

    def test_chat_photo_analysis(self):
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"prompt": "Analyze the nameplate in the field photo at workspace/sandbox_files/test_photo.jpg."},
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        response_text = data["response"]
        assert "Model:" in response_text or "X-200" in response_text or "nameplate" in response_text


@pytest.mark.usefixtures("server_running")
class TestUploadEndpoint:
    """Test the /upload endpoint."""

    def test_upload_file(self):
        # Create a temporary file to upload
        import io
        file_content = b"test upload content"
        files = {"file": ("upload_test.txt", io.BytesIO(file_content), "text/plain")}
        resp = requests.post(
            f"{BASE_URL}/upload",
            files=files,
            params={"target_filename": "upload_test.txt"},
            timeout=10,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "File uploaded"
        assert "upload_test.txt" in data["path"]

        # Verify file exists
        uploaded = SANDBOX_DIR / "upload_test.txt"
        assert uploaded.exists()
        assert uploaded.read_bytes() == file_content

        # Cleanup
        uploaded.unlink()

    def test_upload_traversal_blocked(self):
        import io
        file_content = b"evil content"
        files = {"file": ("evil.txt", io.BytesIO(file_content), "text/plain")}
        resp = requests.post(
            f"{BASE_URL}/upload",
            files=files,
            params={"target_filename": "../../etc/evil.txt"},
            timeout=10,
        )
        assert resp.status_code == 403
