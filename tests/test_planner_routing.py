"""
Regression tests for the planner routing bug.

Bug: MockLLM's create_chat_completion combines system + user messages into
one string, then checks for 'plan' keyword. The planner system prompt contains
the word 'plan', so EVERY input triggers the file_io.read → llm.summarize plan,
regardless of what the user actually typed.

Fix: Added intent classifier (_is_greeting) that detects greetings/simple chat
and returns a direct response. Added is_direct_response() to detect this in
planner output. Added conditional edge in graph to skip pipeline for greetings.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.core.model_manager import MockLLM


class TestMockLLMGreetingDetection:
    """Test that MockLLM correctly detects greetings and returns direct responses."""

    def setup_method(self):
        self.llm = MockLLM()

    def test_hello_returns_direct_response(self):
        """'hello' should return a direct response, not a file_io plan."""
        messages = [
            {"role": "system", "content": "You are a task planner. ..."},
            {"role": "user", "content": "hello"},
        ]
        result = self.llm.create_chat_completion(messages)
        text = result["choices"][0]["text"]
        # Should NOT contain [MockLLM] plan JSON
        assert '{"mock":' not in text
        assert '"tool"' not in text or "file_io" not in text
        # Should be a conversational response
        assert "sovereign" in text.lower() or "workbench" in text.lower() or "hello" in text.lower()

    def test_hi_returns_direct_response(self):
        """'hi' should return a direct response."""
        messages = [
            {"role": "system", "content": "You are a task planner. You produce a plan with steps."},
            {"role": "user", "content": "hi"},
        ]
        result = self.llm.create_chat_completion(messages)
        text = result["choices"][0]["text"]
        assert '"tool"' not in text or "file_io" not in text

    def test_hey_returns_direct_response(self):
        """'hey' should return a direct response."""
        messages = [
            {"role": "system", "content": "You are a task planner. You produce a plan with steps."},
            {"role": "user", "content": "hey"},
        ]
        result = self.llm.create_chat_completion(messages)
        text = result["choices"][0]["text"]
        assert "file_io" not in text

    def test_what_can_you_do_returns_direct_response(self):
        """'what can you do' should return a direct response."""
        messages = [
            {"role": "system", "content": "You are a task planner. You produce a plan with steps."},
            {"role": "user", "content": "what can you do"},
        ]
        result = self.llm.create_chat_completion(messages)
        text = result["choices"][0]["text"]
        assert "file_io" not in text

    def test_who_are_you_returns_direct_response(self):
        """'who are you' should return a direct response."""
        messages = [
            {"role": "system", "content": "You are a task planner. You produce a plan with steps."},
            {"role": "user", "content": "who are you"},
        ]
        result = self.llm.create_chat_completion(messages)
        text = result["choices"][0]["text"]
        assert "file_io" not in text

    def test_good_morning_returns_direct_response(self):
        """'good morning' should return a direct response."""
        messages = [
            {"role": "system", "content": "You are a task planner. You produce a plan with steps."},
            {"role": "user", "content": "good morning"},
        ]
        result = self.llm.create_chat_completion(messages)
        text = result["choices"][0]["text"]
        assert "file_io" not in text

    def test_generate_word_document_still_triggers_plan(self):
        """Actual task inputs should still trigger plan generation."""
        messages = [
            {"role": "system", "content": "You are a task planner. You produce a plan with steps."},
            {"role": "user", "content": "generate a word document for approval"},
        ]
        result = self.llm.create_chat_completion(messages)
        text = result["choices"][0]["text"]
        # Should be a plan (wrapped format)
        assert '{"mock":' in text or '"doc_generator"' in text

    def test_calculate_still_triggers_plan(self):
        """Math inputs should still trigger plan generation."""
        messages = [
            {"role": "system", "content": "You are a task planner. You produce a plan with steps."},
            {"role": "user", "content": "calculate 2 + 2"},
        ]
        result = self.llm.create_chat_completion(messages)
        text = result["choices"][0]["text"]
        assert '{"mock":' in text or '"calculator"' in text


class TestIsDirectResponse:
    """Test the is_direct_response() helper."""

    def test_greeting_plan_is_direct(self):
        """A plan with a greeting response should be detected as direct."""
        from backend.agents.planner import is_direct_response

        plan = [{"tool": "llm", "action": "summarize", "args": ["Hello! I'm the Sovereign AI Workbench..."]}]
        assert is_direct_response(plan) is True

    def test_file_io_plan_is_not_direct(self):
        """A file_io plan should NOT be detected as direct."""
        from backend.agents.planner import is_direct_response

        plan = [
            {"tool": "file_io", "action": "read", "args": ["test.txt"]},
            {"tool": "llm", "action": "summarize", "args": []},
        ]
        assert is_direct_response(plan) is False

    def test_empty_plan_is_not_direct(self):
        """An empty plan should NOT be detected as direct."""
        from backend.agents.planner import is_direct_response

        assert is_direct_response([]) is False
        assert is_direct_response(None) is False

    def test_calculator_plan_is_not_direct(self):
        """A calculator plan should NOT be detected as direct."""
        from backend.agents.planner import is_direct_response

        plan = [{"tool": "calculator", "action": "solve", "args": ["2 + 2"]}]
        assert is_direct_response(plan) is False


class TestGraphRouting:
    """Test that the graph correctly routes greetings to direct response."""

    def test_hello_bypasses_pipeline(self):
        """Sending 'hello' through the graph should skip execute→retrieve→verify."""
        from backend.agents.graph import app as graph_app

        result = graph_app.invoke({"input": "hello", "role": "engineer"})
        output = result.get("output", "")

        # Should NOT reference file paths or permissions
        assert "file_io" not in output.lower()
        assert "permission" not in output.lower()
        assert "test.txt" not in output
        # Should be a conversational response
        assert len(output) > 10

    def test_what_can_you_do_bypasses_pipeline(self):
        """'what can you do' should produce a relevant response."""
        from backend.agents.graph import app as graph_app

        result = graph_app.invoke({"input": "what can you do", "role": "engineer"})
        output = result.get("output", "")

        # Should NOT reference file paths or permissions
        assert "file_io" not in output.lower()
        assert "test.txt" not in output
        assert len(output) > 10

    def test_generate_document_still_uses_pipeline(self):
        """Actual task inputs should still go through the full pipeline."""
        from backend.agents.graph import app as graph_app

        result = graph_app.invoke({"input": "generate a word document", "role": "engineer"})
        # The plan should contain doc_generator
        plan = result.get("plan", [])
        assert any(step.get("tool") == "doc_generator" for step in plan if isinstance(step, dict))
