#!/usr/bin/env python3
"""
Step 2 Test Suite: Agent Orchestration & Routing
Tests the LangGraph ReWOO orchestrator, semantic router, file I/O tool,
and end-to-end graph execution with MockLLM fallback.

Run with: pytest tests/test_step2.py -v
"""
import os
import sys
import json
import time

import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HARDWARE_TIER", "BUILD")


# ============================================================
#  FILE I/O TOOL TESTS
# ============================================================

class TestFileIO:
    """Test the sandboxed file I/O tool."""

    def test_write_and_read(self):
        """Write a file, then read it back."""
        from backend.tools.file_io import write_file, read_file

        result = write_file("test_step2.txt", "Hello World")
        assert "Success" in result, f"Write failed: {result}"

        content = read_file("test_step2.txt")
        assert content == "Hello World", f"Content mismatch: {content}"

        # Cleanup
        from backend.tools.file_io import BASE_DIR
        (BASE_DIR / "test_step2.txt").unlink(missing_ok=True)

    def test_read_nonexistent(self):
        """Reading a non-existent file returns an error message."""
        from backend.tools.file_io import read_file

        content = read_file("nonexistent_file_xyz.txt")
        assert "Error" in content, f"Expected error for missing file: {content}"

    def test_path_traversal_blocked(self):
        """Directory traversal attempts are blocked."""
        from backend.tools.file_io import write_file, read_file

        result = write_file("../../etc/passwd", "malicious")
        assert "Error" in result, f"Traversal should be blocked: {result}"

        content = read_file("../../etc/passwd")
        assert "Error" in content, f"Traversal should be blocked: {content}"

    def test_absolute_path_blocked(self):
        """Absolute paths are blocked."""
        from backend.tools.file_io import write_file

        result = write_file("/etc/passwd", "malicious")
        assert "Error" in result, f"Absolute path should be blocked: {result}"

    def test_write_returns_success_message(self):
        """Write returns a success message with the filename."""
        from backend.tools.file_io import write_file, BASE_DIR

        result = write_file("verify_msg.txt", "test")
        assert "Success" in result
        assert "verify_msg.txt" in result

        # Cleanup
        (BASE_DIR / "verify_msg.txt").unlink(missing_ok=True)


# ============================================================
#  SEMANTIC ROUTER TESTS
# ============================================================

class TestSemanticRouter:
    """Test the keyword-based semantic router."""

    def test_code_routing(self):
        """Prompts with code/script/execute keywords route to CODE."""
        from backend.core.router import route_task

        assert route_task("write a python script") == "CODE"
        assert route_task("execute this code") == "CODE"
        assert route_task("help me with code") == "CODE"

    def test_file_routing(self):
        """Prompts with read/file/write keywords route to FILE."""
        from backend.core.router import route_task

        assert route_task("read the file") == "FILE"
        assert route_task("write to a file") == "FILE"
        assert route_task("open this file") == "FILE"

    def test_vision_routing(self):
        """Prompts with image/scan/drawing keywords route to VISION."""
        from backend.core.router import route_task

        assert route_task("analyze this image") == "VISION"
        assert route_task("scan the document") == "VISION"
        assert route_task("look at this drawing") == "VISION"

    def test_text_default(self):
        """Prompts with no matching keywords route to TEXT."""
        from backend.core.router import route_task

        assert route_task("what is the capital of France") == "TEXT"
        assert route_task("tell me a joke") == "TEXT"

    def test_router_class(self):
        """SemanticRouter class delegates to route_task."""
        from backend.core.router import SemanticRouter

        router = SemanticRouter()
        assert router.route_task("write code") == "CODE"
        assert router.route_task("read file") == "FILE"

    def test_case_insensitive(self):
        """Routing is case-insensitive."""
        from backend.core.router import route_task

        assert route_task("WRITE A SCRIPT") == "CODE"
        assert route_task("Read The File") == "FILE"


# ============================================================
#  MOCKLLM FALLBACK TESTS
# ============================================================

class TestMockLLM:
    """Test the MockLLM fallback in ModelManager."""

    def test_mock_llm_returns_plan(self):
        """MockLLM returns a JSON plan when asked for planning."""
        from backend.core.model_manager import MockLLM

        llm = MockLLM()
        result = llm.create_chat_completion(
            [{"role": "system", "content": "Decompose into steps"}, {"role": "user", "content": "read test.txt"}]
        )
        text = result["choices"][0]["text"]
        plan = json.loads(text)
        assert isinstance(plan, list)
        assert len(plan) > 0
        assert "tool" in plan[0]

    def test_mock_llm_returns_summary(self):
        """MockLLM returns a summary when asked to summarize."""
        from backend.core.model_manager import MockLLM

        llm = MockLLM()
        result = llm.create_chat_completion(
            [{"role": "user", "content": "Please summarize this content"}]
        )
        text = result["choices"][0]["text"]
        assert "summary" in text.lower() or "mock" in text.lower()

    def test_mock_llm_default_response(self):
        """MockLLM returns a default response for unrecognized prompts."""
        from backend.core.model_manager import MockLLM

        llm = MockLLM()
        result = llm.create_chat_completion("What is 2+2?")
        text = result["choices"][0]["text"]
        assert len(text) > 0
        assert "MockLLM" in text

    def test_model_manager_uses_mock(self):
        """ModelManager falls back to MockLLM when model file is missing."""
        from backend.core.model_manager import ModelManager, MockLLM

        mgr = ModelManager(hardware_tier="BUILD", max_vram_gb=4.0)
        model = mgr.load_model("nonexistent_model_xyz.gguf")
        assert isinstance(model, MockLLM), f"Expected MockLLM, got {type(model)}"

    def test_model_manager_generate_with_mock(self):
        """ModelManager.generate() works with MockLLM fallback."""
        from backend.core.model_manager import ModelManager

        mgr = ModelManager(hardware_tier="BUILD", max_vram_gb=4.0)
        output = mgr.generate("nonexistent_model_xyz.gguf", "What is 2+2?")
        assert len(output) > 0


# ============================================================
#  PLANNER TESTS
# ============================================================

class TestPlanner:
    """Test the ReWOO task planner."""

    def test_planner_returns_plan(self):
        """Planner returns a list of steps."""
        from backend.agents.planner import generate_plan
        from backend.core.model_manager import ModelManager

        mgr = ModelManager(hardware_tier="BUILD", max_vram_gb=4.0)
        plan = generate_plan("Read test.txt and summarize it", mgr)

        assert isinstance(plan, list)
        assert len(plan) > 0
        assert "tool" in plan[0]
        assert "action" in plan[0]

    def test_planner_plan_has_valid_structure(self):
        """Each step in the plan has tool, action, and args."""
        from backend.agents.planner import generate_plan
        from backend.core.model_manager import ModelManager

        mgr = ModelManager(hardware_tier="BUILD", max_vram_gb=4.0)
        plan = generate_plan("read test.txt", mgr)

        for step in plan:
            assert "tool" in step, f"Step missing 'tool': {step}"
            assert "action" in step, f"Step missing 'action': {step}"
            assert "args" in step, f"Step missing 'args': {step}"

    def test_planner_fallback_on_bad_input(self):
        """Planner returns a fallback plan for unusual input."""
        from backend.agents.planner import generate_plan
        from backend.core.model_manager import ModelManager

        mgr = ModelManager(hardware_tier="BUILD", max_vram_gb=4.0)
        plan = generate_plan("", mgr)

        assert isinstance(plan, list)
        assert len(plan) > 0


# ============================================================
#  EXECUTOR TESTS
# ============================================================

class TestExecutor:
    """Test the tool dispatcher / executor."""

    def test_execute_file_read(self):
        """Executor can read a file via the file_io tool."""
        from backend.agents.executor import execute_step
        from backend.tools.file_io import write_file, BASE_DIR

        # Write a test file first
        write_file("executor_test.txt", "Executor can read this")

        step = {"tool": "file_io", "action": "read", "args": ["executor_test.txt"]}
        context = {}
        result = execute_step(step, context)

        assert "Executor can read this" in result
        assert "step_0_result" in context

        # Cleanup
        (BASE_DIR / "executor_test.txt").unlink(missing_ok=True)

    def test_execute_file_write(self):
        """Executor can write a file via the file_io tool."""
        from backend.agents.executor import execute_step
        from backend.tools.file_io import read_file, BASE_DIR

        step = {"tool": "file_io", "action": "write", "args": ["exec_write.txt", "written by executor"]}
        context = {}
        result = execute_step(step, context)

        assert "Success" in result
        content = read_file("exec_write.txt")
        assert content == "written by executor"

        # Cleanup
        (BASE_DIR / "exec_write.txt").unlink(missing_ok=True)

    def test_execute_llm_summarize(self):
        """Executor can call LLM for summarization."""
        from backend.agents.executor import execute_step
        from backend.core.model_manager import ModelManager

        mgr = ModelManager(hardware_tier="BUILD", max_vram_gb=4.0)
        step = {"tool": "llm", "action": "summarize", "args": ["This is test content to summarize"]}
        context = {}
        result = execute_step(step, context, mgr)

        assert len(result) > 0
        assert "step_0_result" in context

    def test_execute_unknown_tool(self):
        """Executor returns an error for unknown tools."""
        from backend.agents.executor import execute_step

        step = {"tool": "unknown_tool", "action": "do_something", "args": []}
        context = {}
        result = execute_step(step, context)

        assert "Error" in result

    def test_context_accumulation(self):
        """Executor accumulates results in context across multiple steps."""
        from backend.agents.executor import execute_step
        from backend.tools.file_io import write_file, BASE_DIR

        context = {}

        # Step 1: write a file
        step1 = {"tool": "file_io", "action": "write", "args": ["ctx_test.txt", "context data"]}
        execute_step(step1, context)

        # Step 2: read it back
        step2 = {"tool": "file_io", "action": "read", "args": ["ctx_test.txt"]}
        execute_step(step2, context)

        # Executor stores step_N_result, step_N_tool, step_N_action for each step
        result_keys = [k for k in context if k.endswith("_result")]
        assert len(result_keys) == 2, f"Expected 2 result keys, got {result_keys}"
        # The second result should contain the file content
        second_result = context[result_keys[1]]
        assert "context data" in second_result

        # Cleanup
        (BASE_DIR / "ctx_test.txt").unlink(missing_ok=True)


# ============================================================
#  LANGGRAPH STATE MACHINE TESTS
# ============================================================

class TestGraph:
    """Test the LangGraph state machine."""

    def test_graph_builds(self):
        """The graph compiles without errors."""
        from backend.agents.graph import build_graph

        app = build_graph()
        assert app is not None

    def test_graph_invocation(self):
        """The graph can be invoked with a simple prompt."""
        from backend.agents.graph import app

        result = app.invoke({"input": "Hello, what is 2+2?"})
        assert "output" in result
        assert len(result["output"]) > 0

    def test_graph_plan_execution_synthesis(self):
        """Full pipeline: plan -> execute -> synthesize."""
        from backend.agents.graph import app

        result = app.invoke({"input": "Summarize this text: The sky is blue"})
        assert "output" in result
        assert "plan" in result
        assert isinstance(result["plan"], list)
        assert len(result["plan"]) > 0


# ============================================================
#  END-TO-END INTEGRATION TESTS
# ============================================================

class TestEndToEnd:
    """End-to-end integration tests."""

    def test_file_io_through_graph(self):
        """Write a file, then ask the graph to read and summarize it."""
        from backend.tools.file_io import write_file, BASE_DIR
        from backend.agents.graph import app

        # Write test content
        write_file("e2e_test.txt", "The quick brown fox jumps over the lazy dog")

        # Ask the graph to read and summarize
        result = app.invoke({
            "input": "Read the file e2e_test.txt and tell me what it says."
        })

        assert "output" in result
        assert len(result["output"]) > 0
        # The output should contain information about the file content
        # (either the content itself or a summary)

        # Cleanup
        (BASE_DIR / "e2e_test.txt").unlink(missing_ok=True)

    def test_router_planner_executor_chain(self):
        """Router -> Planner -> Executor chain works end-to-end."""
        from backend.core.router import route_task
        from backend.agents.planner import generate_plan
        from backend.agents.executor import execute_step
        from backend.core.model_manager import ModelManager

        # Route
        category = route_task("read the file test.txt")
        assert category == "FILE"

        # Plan
        mgr = ModelManager(hardware_tier="BUILD", max_vram_gb=4.0)
        plan = generate_plan("read test.txt", mgr)
        assert len(plan) > 0

        # Execute first step
        context = {}
        result = execute_step(plan[0], context, mgr)
        assert len(result) > 0
