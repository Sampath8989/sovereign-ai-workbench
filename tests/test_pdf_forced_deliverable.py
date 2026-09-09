"""
Regression tests for BUG 2 — file-output requests ("2-page PDF", "document",
explicit output-format instructions) silently degrade into free-text chat
with no file ever produced.

Expected orchestrator behavior these tests pin:
  1. file-output intent is detected from the prompt;
  2. the document body is generated in a step SEPARATE from the tool call that
     renders the file (a truncated "I will provide a concise outline..." chat
     reply can never become the PDF body);
  3. the render tool call is FORCED even when no plan/executor step produced a
     file, and the task is only marked done once a real file exists on disk;
  4. multi-page requests get a token budget large enough to finish content.
"""

import os
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents.executor import reset_artifact_tracking
from backend.agents.graph import (
    _generate_doc_content,
    compute_synthesis_token_budget,
    synthesize_node,
)
from backend.core.model_manager import MockLLM
from backend.tools.doc_generator import OUTPUT_DIR


@pytest.fixture(autouse=True)
def clean_state():
    """Ensure clean artifact tracking and no leftover outputs between tests."""
    reset_artifact_tracking()
    yield
    reset_artifact_tracking()
    # Remove deliverables written by these tests.
    for f in OUTPUT_DIR.iterdir():
        if f.is_file() and f.suffix in (".pdf", ".docx", ".pptx", ".xlsx"):
            try:
                f.unlink()
            except OSError:
                pass


def _pdf_header_ok(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(5) == b"%PDF-"


class TestTokenBudget:
    """Multi-page document requests must get an adequate content budget."""

    def test_two_page_pdf_gets_full_budget(self):
        budget = compute_synthesis_token_budget(
            "Create a 2-page PDF on Python basics", is_doc_gen=True
        )
        assert budget >= 2048, f"2-page PDF budget too small: {budget}"

    def test_three_page_pdf_gets_full_budget(self):
        budget = compute_synthesis_token_budget(
            "Create a 3-page guide on machine learning", is_doc_gen=True
        )
        assert budget >= 2048


class TestIntentDetectionAndSeparateContentStep:
    """File-output intent is detected and content is produced in its own step."""

    def test_doc_intent_detected(self):
        for prompt in [
            "Create a 2-page PDF titled 'Pump Maintenance' with sections.",
            "Please write a PDF document about corrosion inspection.",
            "Generate a report file and save it as a PDF.",
        ]:
            assert MockLLM._is_doc_generation_intent(prompt) is True, prompt
            assert MockLLM.detect_format(prompt) == "pdf", prompt

    def test_truncated_chat_reply_is_rejected_as_doc_content(self):
        """A model reply that promises an outline instead of writing content must
        never be accepted as the document body — the content step falls back to
        the deterministic composer."""

        class _ChattyModel:
            def generate_from_messages(self, model_name, messages, **kwargs):
                # Simulates a small local model that truncates mid-list.
                return (
                    "I will provide a concise outline of Python basics:\n"
                    "1. Introduction\n"
                    "2. Core syntax\n"
                    "3. Loops\n"
                )

        content = _generate_doc_content(
            state={"input": "Create a 2-page PDF on python basic all the syntax included"},
            context_text="",
            source_context="",
            model_manager=_ChattyModel(),
            model_name="qwen2.5-coder-7b",
            budget=2048,
            is_mock_model=False,
        )
        assert content and not content.lower().startswith("i will"), (
            f"Placeholder reply leaked into document content: {content[:120]!r}"
        )
        assert "Python" in content, "Deterministic fallback content missing"
        assert "Core Python Basics" in content


class TestForcedRenderToolCall:
    """Even with NO tool step in the plan/executor, a file-output request must
    end with a real file on disk — never with chat prose as the final answer."""

    def _no_tool_state(self, prompt: str) -> dict:
        return {
            "input": prompt,
            "context": {},  # nothing produced a file during execution
            "plan": [],  # planner hypothetically returned no doc_generator step
            "retrieved_sources": [],
            "retrieval_invoked": False,
            "selected_model": "mock",
        }

    def test_synthesize_forces_render_when_no_file_produced(self):
        result = synthesize_node(self._no_tool_state(
            "Create a 2-page PDF titled 'Pump Maintenance Guide' about "
            "inspection steps, lubrication, and safety."
        ))
        deliverables = result.get("deliverables", [])
        output = result.get("output", "")

        assert deliverables, f"No deliverable produced: {output[:200]}"
        pdf_name = deliverables[0]
        assert pdf_name.lower().endswith(".pdf")
        pdf_path = Path(OUTPUT_DIR) / pdf_name
        assert pdf_path.is_file(), f"PDF not on disk: {pdf_path}"
        assert _pdf_header_ok(pdf_path), "File is not a valid PDF (%PDF- header)"

        # The task is only 'done' once the file exists and is advertised.
        assert pdf_name.lower() in output.lower()
        assert "[Error]" not in output

    def test_output_never_claims_success_when_render_fails(self, monkeypatch):
        from backend.agents import graph as graph_mod

        def boom(*args, **kwargs):
            raise RuntimeError("render tool exploded")

        monkeypatch.setattr(graph_mod, "_generate_doc_content",
                            lambda *a, **k: "Real body content that would have been rendered.")
        # Patch doc_generator.generate_doc where synthesize_node imports it from.
        monkeypatch.setattr("backend.tools.doc_generator.generate_doc", boom)

        # synthesize_node imports generate_doc inside the function from
        # backend.tools.doc_generator; patch the module attribute used there.
        import backend.tools.doc_generator as doc_gen_mod
        monkeypatch.setattr(doc_gen_mod, "generate_doc", boom)

        result = synthesize_node(self._no_tool_state(
            "Create a 2-page PDF on python basics with all the syntax included."
        ))
        assert result.get("deliverables") == []
        output = result.get("output", "")
        assert "[Error]" in output, "Failure must be surfaced honestly"
        assert "Deliverable generated" not in output


class TestFullPipelinePdf:
    """End-to-end: a "2-page PDF" request through the whole graph yields a real,
    valid PDF whose body is the generated content."""

    def test_full_graph_python_pdf(self):
        from backend.agents.graph import app

        result = app.invoke(
            {
                "input": "create a pdf on python basic all the syntax included",
                "role": "admin",
                "selected_model": "mock",
            }
        )
        deliverables = result.get("deliverables", [])
        assert deliverables, f"No PDF produced. Output: {result.get('output','')[:200]}"
        pdfs = [d for d in deliverables if d.lower().endswith(".pdf")]
        assert pdfs, f"Expected PDF deliverable, got {deliverables}"

        pdf_path = Path(OUTPUT_DIR) / pdfs[0]
        assert pdf_path.is_file()
        assert _pdf_header_ok(pdf_path)

        from pypdf import PdfReader
        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages)
        assert "Python" in text, "Rendered PDF missing generated content"

        output = result.get("output", "")
        assert pdfs[0].lower() in output.lower(), "Output must reference the deliverable"

    def test_full_graph_detailed_two_page_spec(self):
        from backend.agents.graph import app

        prompt = (
            "Create a professional, beginner-friendly 2-page PDF titled "
            "'Python Basics: Introduction to Python' that includes:\n"
            "1. What is Python?\n"
            "2. Core Syntax & Data Types with code blocks\n"
            "3. Control Flow (if/else, loops)\n"
            "4. Functions & Next Steps"
        )
        result = app.invoke(
            {"input": prompt, "role": "admin", "selected_model": "mock"}
        )
        deliverables = result.get("deliverables", [])
        pdfs = [d for d in deliverables if d.lower().endswith(".pdf")]
        assert pdfs, f"No PDF produced for detailed spec. Output: {result.get('output','')[:200]}"

        pdf_path = Path(OUTPUT_DIR) / pdfs[0]
        assert pdf_path.is_file()
        assert _pdf_header_ok(pdf_path)

        from pypdf import PdfReader
        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages)
        assert "Python" in text and len(text) > 150
