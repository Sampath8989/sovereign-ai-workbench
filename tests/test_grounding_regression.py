"""
Unified Grounding & Retrieval Regression Test Suite.

Verifies root cause fixes across all 3 categories in the same run:
1. Valid Grounded Queries: Retrieves relevant documents, answers with exact factual claim ("5mm"), and attaches citations.
2. Invalid Grounded Queries: Requests facts not in corpus (e.g. SOP-44 operating pressure) or nonexistent SOPs; strictly refuses via threshold gate.
3. Non-Retrieval Queries: Pure definitions ("What is a neural network?"), general knowledge ("capital of France"), conceptual comparisons ("supervised vs unsupervised learning"), and math; deterministically bypasses RAG and answers directly without false refusal or stray citations.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture(scope="module")
def client():
    """Create test client with initialized workbench."""
    with TestClient(app) as c:
        # Ensure SOP-44 is ingested into knowledge base
        ingest_resp = c.post("/ingest", json={"directory": "data/knowledge_base/sops"})
        assert ingest_resp.status_code == 200
        yield c


class TestUnifiedGroundingRegression:
    """Unified regression test covering all 3 categories in the same test run."""

    # -------------------------------------------------------------------------
    # Category A: Valid Grounded Queries
    # -------------------------------------------------------------------------

    def test_category_a_valid_grounded_sop44(self, client):
        """
        Query: 'What is the maximum allowable corrosion depth for pressure vessels per SOP-44?'
        Expectation:
        - Retrieves sop-44.txt with confidence >= 0.20
        - Response contains '5mm'
        - Response contains citation tag [Source: sop-44.txt
        - Does NOT falsely refuse
        """
        resp = client.post(
            "/chat",
            json={"prompt": "What is the maximum allowable corrosion depth for pressure vessels per SOP-44?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        resp_text = data.get("response", "")

        assert "5mm" in resp_text, f"Expected '5mm' in response: {resp_text}"
        assert "[Source: sop-44.txt" in resp_text, (
            f"Expected citation tag '[Source: sop-44.txt ...]' in response: {resp_text}"
        )
        assert "not found in available sources" not in resp_text.lower(), (
            f"False negative refusal detected on valid retrieval: {resp_text}"
        )

    def test_category_a_valid_grounded_inspection_frequency(self, client):
        """
        Query: 'What is the inspection frequency for pressure vessels per SOP-44?'
        Expectation:
        - Retrieves sop-44.txt
        - Response contains citation tag [Source: sop-44.txt
        - Does NOT falsely refuse
        """
        resp = client.post(
            "/chat",
            json={"prompt": "What is the inspection frequency for pressure vessels per SOP-44?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        resp_text = data.get("response", "")

        assert "sop-44" in resp_text.lower() or "quarterly" in resp_text.lower() or "inspected" in resp_text.lower(), (
            f"Expected inspection or sop-44 details in response: {resp_text}"
        )
        assert "[Source: sop-44.txt" in resp_text, (
            f"Expected citation tag '[Source: sop-44.txt ...]' in response: {resp_text}"
        )
        assert "not found in available sources" not in resp_text.lower(), (
            f"False negative refusal detected on valid retrieval: {resp_text}"
        )

    # -------------------------------------------------------------------------
    # Category B: Invalid Grounded Queries
    # -------------------------------------------------------------------------

    def test_category_b_invalid_grounded_unsupported_fact(self, client):
        """
        Query: 'What is the maximum operating pressure per SOP-44?'
        SOP-44 specifies corrosion limits (5mm) and inspection frequency, but NOT operating pressure.
        Expectation:
        - Confidence drops below threshold 0.20
        - System correctly refuses with 'not found in available sources'
        - Must NOT hallucinate an operating pressure or attach a false citation tag to it
        """
        resp = client.post(
            "/chat",
            json={"prompt": "What is the maximum operating pressure per SOP-44?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        resp_text = data.get("response", "")

        assert "not found in available sources" in resp_text.lower(), (
            f"Expected explicit refusal for unsupported fact in SOP-44: {resp_text}"
        )
        assert "[Source:" not in resp_text, (
            f"Stray citation marker found on refused query: {resp_text}"
        )

    def test_category_b_invalid_grounded_nonexistent_doc(self, client):
        """
        Query: 'What is the safety margin per SOP-99?'
        SOP-99 does not exist in the knowledge base.
        Expectation:
        - RAG search finds no match
        - System correctly refuses with 'not found in available sources'
        """
        resp = client.post(
            "/chat",
            json={"prompt": "What is the safety margin per SOP-99?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        resp_text = data.get("response", "")

        assert "not found in available sources" in resp_text.lower(), (
            f"Expected explicit refusal for nonexistent document SOP-99: {resp_text}"
        )

    # -------------------------------------------------------------------------
    # Category C: Non-Retrieval Queries (Bypasses RAG, Answers Directly)
    # -------------------------------------------------------------------------

    def test_category_c_non_retrieval_neural_network(self, client):
        """
        Query: 'What is a neural network?'
        Expectation:
        - Bypasses RAG completely (retrieval_invoked = False)
        - Does NOT refuse ('not found in available sources')
        - Answers with a valid explanation
        - Contains NO citation tags
        """
        resp = client.post(
            "/chat",
            json={"prompt": "What is a neural network?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        resp_text = data.get("response", "")

        assert "not found in available sources" not in resp_text.lower(), (
            f"False refusal on general knowledge definition: {resp_text}"
        )
        assert len(resp_text) > 30, f"Response too short: {resp_text}"
        assert any(term in resp_text.lower() for term in ["neural", "network", "algorithm", "layer", "model", "data"]), (
            f"Expected definition of neural network: {resp_text}"
        )
        assert "[Source:" not in resp_text, f"Stray citation tag on non-retrieval query: {resp_text}"

    def test_category_c_non_retrieval_capital_of_france(self, client):
        """
        Query: 'What is the capital of France?'
        Expectation:
        - Bypasses RAG completely
        - Answers Paris (or request response under MockLLM)
        - Does NOT refuse
        - Contains NO citation tags
        """
        resp = client.post(
            "/chat",
            json={"prompt": "What is the capital of France?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        resp_text = data.get("response", "")

        assert "not found in available sources" not in resp_text.lower(), (
            f"False refusal on general knowledge query: {resp_text}"
        )
        assert "paris" in resp_text.lower() or "capital of france" in resp_text.lower(), (
            f"Expected answer for capital of France: {resp_text}"
        )
        assert "[Source:" not in resp_text, f"Stray citation tag on non-retrieval query: {resp_text}"

    def test_category_c_non_retrieval_concept_comparison(self, client):
        """
        Query: 'supervised vs unsupervised learning'
        Expectation:
        - Bypasses RAG completely
        - Explains difference between supervised and unsupervised learning
        - Does NOT refuse
        - Contains NO citation tags
        """
        resp = client.post(
            "/chat",
            json={"prompt": "supervised vs unsupervised learning"},
        )
        assert resp.status_code == 200
        data = resp.json()
        resp_text = data.get("response", "")

        assert "not found in available sources" not in resp_text.lower(), (
            f"False refusal on conceptual comparison query: {resp_text}"
        )
        assert any(term in resp_text.lower() for term in ["supervised", "unsupervised", "learning", "label"]), (
            f"Expected conceptual comparison: {resp_text}"
        )
        assert "[Source:" not in resp_text, f"Stray citation tag on non-retrieval query: {resp_text}"

    def test_category_c_non_retrieval_math_problem(self, client):
        """
        Query: 'Solve: x + 5 = 10'
        Expectation:
        - Executes calculator tool or direct calculation
        - Returns answer containing '5'
        - Does NOT refuse
        - Contains NO citation tags
        """
        resp = client.post(
            "/chat",
            json={"prompt": "Solve: x + 5 = 10"},
        )
        assert resp.status_code == 200
        data = resp.json()
        resp_text = data.get("response", "")

        assert "not found in available sources" not in resp_text.lower(), (
            f"False refusal on math problem: {resp_text}"
        )
        assert "5" in resp_text, f"Expected '5' in math solution: {resp_text}"
        assert "[Source:" not in resp_text, f"Stray citation tag on math problem: {resp_text}"

    # -------------------------------------------------------------------------
    # Unified All-3-in-One Sequential Session Run
    # -------------------------------------------------------------------------

    def test_all_three_categories_unified_session(self, client):
        """
        Run all 3 categories sequentially in the exact same session:
        1. Valid grounded -> retrieves + answers 5mm + cites
        2. Invalid grounded -> correctly refuses
        3. Non-retrieval -> answers directly without citation or refusal
        """
        # 1. Valid Grounded
        r_valid = client.post(
            "/chat",
            json={"prompt": "What is the maximum allowable corrosion depth for pressure vessels per SOP-44?"},
        )
        assert r_valid.status_code == 200
        text_valid = r_valid.json().get("response", "")
        assert "5mm" in text_valid, f"Expected 5mm in: {text_valid}"
        assert "[Source: sop-44.txt" in text_valid, f"Expected citation in: {text_valid}"

        # 2. Invalid Grounded
        r_invalid = client.post(
            "/chat",
            json={"prompt": "What is the maximum operating pressure per SOP-44?"},
        )
        assert r_invalid.status_code == 200
        text_invalid = r_invalid.json().get("response", "")
        assert "not found in available sources" in text_invalid.lower(), f"Expected refusal in: {text_invalid}"
        assert "[Source:" not in text_invalid, f"Stray citation in: {text_invalid}"

        # 3. Non-Retrieval
        r_non_retrieval = client.post(
            "/chat",
            json={"prompt": "What is a neural network?"},
        )
        assert r_non_retrieval.status_code == 200
        text_non_retrieval = r_non_retrieval.json().get("response", "")
        assert "not found in available sources" not in text_non_retrieval.lower(), f"Unexpected refusal in: {text_non_retrieval}"
        assert len(text_non_retrieval) > 30
        assert "[Source:" not in text_non_retrieval
