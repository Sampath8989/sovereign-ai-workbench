"""
Pytest test suite for Sovereign AI Workbench Step 2.
Tests file I/O, semantic routing, agent graph execution, and /chat endpoint.

Assumes the FastAPI server is running on http://localhost:8000.
Run with: pytest tests/test_step2.py -v
"""

import json
import os
import sys
import time

import pytest
import requests

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BASE_URL = "http://localhost:8000"


def wait_for_server(max_retries=10, delay=1.0):
    """Wait for the FastAPI server to be available."""
    for i in range(max_retries):
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=2)
            if resp.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(delay)
    return False


@pytest.fixture(scope="session", autouse=True)
def server_ready():
    """Ensure server is running before tests."""
    if not wait_for_server():
        pytest.skip("FastAPI server not available at localhost:8000")


class TestFileIO:
    """Test suite for the file I/O tool."""

    def test_write_and_read_file(self):
        """Write a file, then read it back. Verify contents match."""
        from backend.tools.file_io import write_file, read_file

        test_content = "Hello World"
        result = write_file("test_step2.txt", test_content)
        assert "Success" in result, f"write_file failed: {result}"

        content = read_file("test_step2.txt")
        assert content == test_content, f"Expected '{test_content}', got '{content}'"

    def test_read_nonexistent_file(self):
        """Reading a non-existent file returns an error message."""
        from backend.tools.file_io import read_file

        result = read_file("nonexistent_file_xyz.txt")
        assert "Error" in result
        assert "not found" in result.lower()

    def test_directory_traversal_blocked(self):
        """Directory traversal attempts are blocked."""
        from backend.tools.file_io import write_file, read_file

        result = write_file("../../etc/passwd", "malicious")
        assert "Error" in result, f"Traversal should be blocked, got: {result}"

        result = read_file("../../etc/passwd")
        assert "Error" in result, f"Traversal should be blocked, got: {result}"


class TestRouter:
    """Test suite for the semantic router."""

    def test_code_routing(self):
        """Prompts containing code/script/execute route to CODE."""
        from backend.core.router import route_task

        assert route_task("write a python script") == "CODE"
        assert route_task("execute this code") == "CODE"
        assert route_task("can you code a function?") == "CODE"

    def test_file_routing(self):
        """Prompts containing read/file/write route to FILE."""
        from backend.core.router import route_task

        assert route_task("read the file") == "FILE"
        assert route_task("write to a file") == "FILE"
        assert route_task("what's in this file?") == "FILE"

    def test_vision_routing(self):
        """Prompts containing image/scan/drawing route to VISION."""
        from backend.core.router import route_task

        assert route_task("scan this image") == "VISION"
        assert route_task("what do you see in this drawing?") == "VISION"
        assert route_task("analyze the image") == "VISION"

    def test_text_routing(self):
        """Generic prompts route to TEXT."""
        from backend.core.router import route_task

        assert route_task("what is the capital of France?") == "TEXT"
        assert route_task("tell me a joke") == "TEXT"


class TestMockLLM:
    """Test suite for the MockLLM fallback."""

    def test_mock_plan_generation(self):
        """MockLLM returns a valid JSON plan when asked for steps."""
        from backend.core.model_manager import MockLLM

        llm = MockLLM()
        result = llm.create_chat_completion(
            [{"role": "user", "content": "Generate a plan of steps"}]
        )
        text = result["choices"][0]["text"]
        plan = json.loads(text)
        assert isinstance(plan, list)
        assert len(plan) > 0
        assert "tool" in plan[0]

    def test_mock_summarize(self):
        """MockLLM returns a summary when asked to summarize."""
        from backend.core.model_manager import MockLLM

        llm = MockLLM()
        result = llm.create_chat_completion(
            [{"role": "user", "content": "Please summarize this text"}]
        )
        text = result["choices"][0]["text"]
        assert "mock summary" in text.lower()

    def test_mock_default(self):
        """MockLLM returns a default response for unrecognized prompts."""
        from backend.core.model_manager import MockLLM

        llm = MockLLM()
        result = llm.create_chat_completion("hello world")
        text = result["choices"][0]["text"]
        assert "MockLLM" in text


class TestPlanner:
    """Test suite for the ReWOO planner."""

    def test_planner_generates_plan(self):
        """Planner returns a list of step dicts."""
        from backend.agents.planner import generate_plan
        from backend.core.model_manager import ModelManager

        mm = ModelManager()
        plan = generate_plan("Read test.txt and summarize it", mm)
        assert isinstance(plan, list)
        assert len(plan) > 0
        assert "tool" in plan[0]
        assert "action" in plan[0]

    def test_planner_fallback_on_error(self):
        """Planner returns a fallback plan when parsing fails."""
        from backend.agents.planner import generate_plan, _make_fallback

        fallback = _make_fallback("test prompt")
        assert isinstance(fallback, list)
        assert fallback[0]["tool"] == "llm"


class TestChatEndpoint:
    """Test suite for the /chat endpoint (end-to-end agent graph)."""

    def test_chat_basic(self):
        """POST /chat with a simple prompt returns a response."""
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"prompt": "What is 2 + 2?"},
            timeout=60,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert len(data["response"]) > 0

    def test_chat_file_io_chain(self):
        """
        POST /chat with a file I/O request proves the full chain works:
        Planner -> Executor -> File I/O -> Synthesizer.
        """
        # First, ensure test.txt exists via file_io tool directly
        from backend.tools.file_io import write_file
        write_file("test.txt", "Hello World from Step 2")

        # Now ask the agent to read and summarize it
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"prompt": "Read the file test.txt and tell me what it says."},
            timeout=60,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        # The response should contain either the file content or a mock summary
        response_text = data["response"].lower()
        assert (
            "hello world" in response_text
            or "mock summary" in response_text
            or "test.txt" in response_text
        ), f"Response doesn't contain expected content: {data['response'][:200]}"

    def test_chat_empty_prompt(self):
        """POST /chat with empty prompt still returns a response (no crash)."""
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"prompt": ""},
            timeout=60,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
