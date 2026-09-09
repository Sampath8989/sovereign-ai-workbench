"""
Comprehensive regression test suite for:
1. Format Routing: PDF requests produce genuine .pdf files with valid %PDF- headers.
2. Detailed Specs Handling: Long, structured prompts do not silently skip deliverable generation.
3. Token Budget Dynamic Scaling: Synthesis budget scales up to 2048 tokens based on request complexity.
4. Model / VRAM Architecture Integrity: 14B models on BUILD tier cannot be pinned, VRAM reports live telemetry.
5. Ambiguous format requests default to prompting the user for format selection.
"""

import os
import re
import sys
import pytest
from pathlib import Path
from pypdf import PdfReader

# Ensure backend in import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.agents.graph import app as graph_app, compute_synthesis_token_budget
from backend.agents.executor import reset_artifact_tracking
from backend.core.model_manager import MockLLM, ModelManager, query_free_vram_gb, query_used_vram_gb, query_total_vram_gb
from backend.tools.doc_generator import OUTPUT_DIR, generate_pdf, generate_doc
from backend.config import MODEL_ROSTERS, get_tier


@pytest.fixture(autouse=True)
def setup_teardown():
    """Ensure clean artifact tracking and output directory before and after tests."""
    reset_artifact_tracking()
    yield
    reset_artifact_tracking()


class TestFormatRoutingAndPDFGeneration:
    """Test 1: PDF requests route to PDF generator and produce real, valid PDFs."""

    def test_direct_pdf_generator_creates_valid_pdf(self, tmp_path):
        """Verify reportlab PDF generation produces valid %PDF- header and readable content."""
        filename = "test_basics.pdf"
        title = "Python Basics Guide"
        content = (
            "Section 1: Introduction\nPython is a high-level programming language.\n\n"
            "Section 2: Syntax\nVariables and data types:\n"
            "```python\nx = 10\nname = 'Workbench'\nprint(f'{name}: {x}')\n```\n\n"
            "- Lists: [1, 2, 3]\n- Dictionaries: {'a': 1}\n- Tuples: (1, 2)"
        )
        pdf_path = generate_pdf(filename, title, content)
        assert os.path.exists(pdf_path)
        assert pdf_path.endswith(".pdf")

        # Check %PDF- header
        with open(pdf_path, "rb") as f:
            header = f.read(5)
            assert header == b"%PDF-", f"Expected %PDF- header, got {header}"

        # Read back using pypdf
        reader = PdfReader(pdf_path)
        assert len(reader.pages) >= 1
        full_text = "\n".join(page.extract_text() for page in reader.pages)
        assert "Python Basics Guide" in full_text
        assert "Introduction" in full_text

    def test_exact_prompt_create_pdf_on_python_basics(self):
        """
        User Prompt: 'create a pdf on python basic all the syntax included'
        Must produce a real .pdf deliverable, NOT a .docx file.
        """
        prompt = "create a pdf on python basic all the syntax included"

        # Verify MockLLM format detection
        fmt = MockLLM.detect_format(prompt)
        assert fmt == "pdf", f"Expected format 'pdf', got '{fmt}'"

        result = graph_app.invoke({
            "input": prompt,
            "role": "admin",
            "selected_model": "mock",
        })

        deliverables = result.get("deliverables", [])
        output = result.get("output", "")

        # Must have at least one deliverable
        assert len(deliverables) > 0, f"No deliverables generated. Result: {result}"

        # Must NOT be .docx
        docx_deliverables = [d for d in deliverables if d.lower().endswith(".docx")]
        assert len(docx_deliverables) == 0, f"Unexpected .docx deliverable found: {docx_deliverables}"

        # Must be a .pdf deliverable
        pdf_deliverables = [d for d in deliverables if d.lower().endswith(".pdf")]
        assert len(pdf_deliverables) >= 1, f"Expected .pdf deliverable, got: {deliverables}"

        target_pdf_name = pdf_deliverables[0]
        target_pdf_path = Path(OUTPUT_DIR) / target_pdf_name
        assert target_pdf_path.exists(), f"PDF file not found at {target_pdf_path}"

        # Validate PDF integrity with pypdf
        with open(target_pdf_path, "rb") as f:
            header = f.read(5)
            assert header == b"%PDF-", "File is not a valid PDF (%PDF- header missing)"

        reader = PdfReader(str(target_pdf_path))
        assert len(reader.pages) >= 1
        extracted = "\n".join(page.extract_text() for page in reader.pages)
        assert "Python" in extracted

        # Output text must reference the deliverable
        assert target_pdf_name.lower() in output.lower()

    def test_ambiguous_document_request_prompts_for_format(self):
        """
        When format cannot be determined (e.g. 'create a document on python basics'),
        the system must default to asking rather than silently picking docx.
        """
        prompt = "create a document on python basics"
        fmt = MockLLM.detect_format(prompt)
        assert fmt is None, f"Expected None for ambiguous format, got {fmt}"

        mock_llm = MockLLM()
        plan_res = mock_llm.create_chat_completion([
            {"role": "system", "content": "You are a task planner."},
            {"role": "user", "content": prompt},
        ])
        plan_text = plan_res["choices"][0]["text"]
        # Must ask for format preference
        assert "format" in plan_text.lower()
        assert "pdf" in plan_text.lower()
        assert "word" in plan_text.lower() or "docx" in plan_text.lower()


class TestDetailedSpecsHandling:
    """Test 2: Detailed specs prompts do not silently skip deliverable generation."""

    def test_detailed_spec_prompt_1(self):
        """
        User Prompt:
        Create a professional, beginner-friendly 2-page PDF titled “Python Basics: Introduction to Python” that includes:
        1. Cover / Title Block ...
        2. What is Python? ...
        3. Core Syntax & Data Types ...
        """
        prompt = (
            'Create a professional, beginner-friendly 2-page PDF titled “Python Basics: Introduction to Python” that includes:\n'
            '1. Cover / Title Block (title, subtitle, author, date, version)\n'
            '2. What is Python? (high-level overview, key features, use cases)\n'
            '3. Core Syntax & Data Types with code blocks (variables, strings, ints/floats, booleans, lists, dicts)\n'
            '4. Control Flow with code blocks (if/elif/else, for loops, while loops)\n'
            '5. Functions & Basic Error Handling (def, return, try/except)\n'
            '6. Best Practices & Next Steps (PEP 8, virtual environments, recommended resources)'
        )

        assert MockLLM._is_doc_generation_intent(prompt) is True
        assert MockLLM.detect_format(prompt) == "pdf"

        result = graph_app.invoke({
            "input": prompt,
            "role": "admin",
            "selected_model": "mock",
        })

        deliverables = result.get("deliverables", [])
        output = result.get("output", "")

        # Deliverable must exist and be PDF
        pdf_deliverables = [d for d in deliverables if d.lower().endswith(".pdf")]
        assert len(pdf_deliverables) >= 1, f"Expected PDF deliverable for detailed spec, got {deliverables}"

        pdf_path = Path(OUTPUT_DIR) / pdf_deliverables[0]
        assert pdf_path.exists()

        with open(pdf_path, "rb") as f:
            header = f.read(5)
            assert header == b"%PDF-"

        # Response must not cut off prematurely
        assert len(output) > 200
        assert "Python Basics" in output

    def test_detailed_spec_prompt_2(self):
        """
        User Prompt with bold markdown:
        Create a professional **2-page PDF** titled “Python Basics: Introduction to Python” for absolute beginners...
        """
        prompt = (
            'Create a professional **2-page PDF** titled “Python Basics: Introduction to Python” for absolute beginners, covering:\n'
            '- Section 1: Overview & Setup\n'
            '- Section 2: Variables, Data Types & Syntax\n'
            '- Section 3: Control Flow (if/else, loops)\n'
            '- Section 4: Functions & Modules\n'
            '- Section 5: Common Pitfalls & Quick Reference'
        )

        assert MockLLM._is_doc_generation_intent(prompt) is True
        assert MockLLM.detect_format(prompt) == "pdf"

        result = graph_app.invoke({
            "input": prompt,
            "role": "admin",
            "selected_model": "mock",
        })

        deliverables = result.get("deliverables", [])
        pdf_deliverables = [d for d in deliverables if d.lower().endswith(".pdf")]
        assert len(pdf_deliverables) >= 1, f"Expected PDF deliverable, got {deliverables}"


class TestTokenBudgetScaling:
    """Test 3: Token budget dynamically scales based on structure and length."""

    def test_multi_page_token_budget(self):
        budget_2page = compute_synthesis_token_budget("Create a 2-page PDF on Python", is_doc_gen=True)
        assert budget_2page >= 1200
        assert budget_2page <= 2048

        budget_3page = compute_synthesis_token_budget("Create a 3-page guide on machine learning", is_doc_gen=True)
        assert budget_3page >= 1800
        assert budget_3page <= 2048

    def test_comprehensive_syntax_token_budget(self):
        budget = compute_synthesis_token_budget("create a pdf on python basic all the syntax included", is_doc_gen=True)
        assert budget == 2048

    def test_simple_query_token_budget(self):
        budget = compute_synthesis_token_budget("What is Python?", is_doc_gen=False)
        assert budget <= 512


class TestModelManagerAndVRAMIntegrity:
    """Test 4: Hardware tier roster integrity, live VRAM telemetry, and pinning rejection."""

    def test_roster_phi4_14b_not_3_6_gb(self):
        """14B models cannot reside in 3.6 GB VRAM. Ensure BUILD tier roster is corrected."""
        build_roster = MODEL_ROSTERS.get("BUILD", {})
        phi4_vram = build_roster.get("phi4-14b.gguf")
        assert phi4_vram is not None
        assert phi4_vram >= 8.5, f"Expected phi4-14b to require >=8.5 GB VRAM, got {phi4_vram}"

    def test_pinning_oversized_model_rejected(self):
        """Pinning a 9.0 GB model on a BUILD tier (<=4.0 GB budget) must be rejected with ValueError."""
        mm = ModelManager(hardware_tier="BUILD", max_vram_gb=3.7)

        # Attempting to pin phi4-14b.gguf (9.0 GB) must raise ValueError
        with pytest.raises(ValueError) as excinfo:
            mm.pin_model("phi4-14b.gguf")

        err_msg = str(excinfo.value)
        assert "exceeds the effective budget" in err_msg
        assert mm.pinned_model is None

    def test_pinning_valid_model_and_lru_protection(self):
        """Pinning a model within budget succeeds, and LRU eviction protects the pinned model."""
        mm = ModelManager(hardware_tier="BUILD", max_vram_gb=4.0)

        # Pin small model
        res = mm.pin_model("qwen-0.5b.gguf")
        assert res["status"] == "pinned"
        assert mm.pinned_model == "qwen-0.5b.gguf"

        # Load another small model
        mm.load_model("qwen-1.5-4b.gguf", reject_oversized=False)

        # Evict LRU: must evict qwen-1.5-4b, NOT the pinned qwen-0.5b
        evicted = mm._evict_lru()
        assert evicted != "qwen-0.5b.gguf"
        assert "qwen-0.5b.gguf" in mm.resident_models

        # Unpin model
        unpin_res = mm.unpin_model()
        assert unpin_res["status"] == "unpinned"
        assert mm.pinned_model is None

    def test_get_status_contains_live_vram_fields(self):
        """get_status() must report live VRAM telemetry."""
        mm = ModelManager()
        status = mm.get_status()
        assert "live_free_vram_gb" in status
        assert "live_used_vram_gb" in status
        assert "live_total_vram_gb" in status
        assert "pinned_model" in status
        assert "effective_budget_gb" in status
