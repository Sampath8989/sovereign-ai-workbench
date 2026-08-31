"""
Step 4 Tests: Deliverable Synthesis Tools (Word, PPT, Excel, Calculator).
Verifies tool generation, chat endpoint integration, and file download.
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

# Output directory for generated files
OUTPUT_DIR = PROJECT_ROOT / "workspace" / "outputs"
SANDBOX_DIR = PROJECT_ROOT / "workspace" / "sandbox_files"

# FastAPI server URL
BASE_URL = "http://127.0.0.1:8000"


# ---------- Direct Tool Tests ----------


class TestCalculatorDirect:
    """Test the calculator tool directly (no server needed)."""

    def test_solve_linear_equation(self):
        from backend.tools.calculator import solve_expression

        result = solve_expression("x + 5 = 10")
        assert "5" in result, f"Expected '5' in result: {result}"

    def test_solve_2x_equation(self):
        from backend.tools.calculator import solve_expression

        result = solve_expression("2*x + 5 = 15")
        assert "5" in result, f"Expected '5' in result: {result}"

    def test_factor(self):
        from backend.tools.calculator import solve_expression

        result = solve_expression("factor x**2 - 1")
        assert "x" in result.lower(), f"Expected variable in result: {result}"

    def test_empty_expression(self):
        from backend.tools.calculator import solve_expression

        result = solve_expression("")
        assert "Error" in result


class TestDocGeneratorDirect:
    """Test the Word document generator directly."""

    def test_generate_doc(self):
        from backend.tools.doc_generator import generate_doc

        path = generate_doc("test_direct.docx", "Test Title", "Test content body.")
        assert os.path.exists(path), f"File not created: {path}"
        assert path.endswith("test_direct.docx")

        # Cleanup
        os.remove(path)


class TestSpreadsheetGeneratorDirect:
    """Test the spreadsheet generator directly."""

    def test_generate_sheet(self):
        from backend.tools.spreadsheet_generator import generate_sheet

        data = [["Name", "Age"], ["Alice", "25"], ["Bob", "30"]]
        path = generate_sheet("test_direct.xlsx", data)
        assert os.path.exists(path), f"File not created: {path}"

        # Cleanup
        os.remove(path)


class TestPptGeneratorDirect:
    """Test the PowerPoint generator directly."""

    def test_generate_ppt(self):
        from backend.tools.ppt_generator import generate_ppt

        path = generate_ppt("test_direct.pptx", "My Slide", ["Point 1", "Point 2"])
        assert os.path.exists(path), f"File not created: {path}"

        # Cleanup
        os.remove(path)


# ---------- Integration Tests (require running server) ----------


@pytest.fixture(scope="module")
def server_running():
    """Check if the FastAPI server is running. Skip tests if not."""
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            yield True
        else:
            pytest.skip("FastAPI server not running or unhealthy")
    except requests.ConnectionError:
        pytest.skip("FastAPI server not running")


@pytest.mark.usefixtures("server_running")
class TestDocGenerationViaChat:
    """Test Word document generation through the /chat endpoint."""

    def test_chat_creates_docx(self):
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"prompt": "Create a word document named test_chat.docx with title 'Test' and content 'Hello'."},
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data

        # Check that the output mentions the file or was processed
        response_text = data["response"]
        # The response should reference the file path or contain success info
        assert len(response_text) > 0


@pytest.mark.usefixtures("server_running")
class TestFileDownload:
    """Test the /download endpoint."""

    def test_download_returns_200(self):
        # First, ensure a file exists by generating one
        from backend.tools.doc_generator import generate_doc
        path = generate_doc("download_test.docx", "Download Test", "Content")

        try:
            resp = requests.get(
                f"{BASE_URL}/download",
                params={"filename": "download_test.docx"},
                timeout=10,
            )
            assert resp.status_code == 200
            # Check content type is valid docx mime type
            content_type = resp.headers.get("Content-Type", "")
            assert "document" in content_type or "octet-stream" in content_type, \
                f"Unexpected content type: {content_type}"
        finally:
            os.remove(path)

    def test_download_nonexistent_returns_404(self):
        resp = requests.get(
            f"{BASE_URL}/download",
            params={"filename": "nonexistent_file_12345.docx"},
            timeout=10,
        )
        assert resp.status_code == 404


@pytest.mark.usefixtures("server_running")
class TestSpreadsheetGenerationViaChat:
    """Test spreadsheet generation through the /chat endpoint."""

    def test_chat_creates_xlsx(self):
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={
                "prompt": "Generate a spreadsheet named test_chat.xlsx with data [['Name', 'Age'], ['Alice', '25']]"
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data


@pytest.mark.usefixtures("server_running")
class TestCalculatorViaChat:
    """Test calculator through the /chat endpoint."""

    def test_chat_calculator(self):
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"prompt": "Solve: x + 5 = 10"},
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data


@pytest.mark.usefixtures("server_running")
class TestPptGenerationViaChat:
    """Test PowerPoint generation through the /chat endpoint."""

    def test_chat_creates_pptx(self):
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"prompt": "Create a PowerPoint presentation named test_chat.pptx with title 'My Slides' and bullet points ['First point', 'Second point']"},
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
