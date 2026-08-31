"""
Step 3 Tests: Knowledge Base & Grounding
Tests Hybrid RAG, ingestion, citation tagging, and CoT verification.
"""

import os
import sys
import time
import pytest
import requests

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://localhost:8000"


class TestRAGSearch:
    """Test the Hybrid RAG search system."""

    def test_hybrid_rag_instantiation(self):
        """Test that HybridRAG can be instantiated with fallbacks."""
        from backend.tools.rag_search import HybridRAG, get_embedder, MockEmbedder

        rag = HybridRAG()
        status = rag.get_status()

        assert "qdrant_connected" in status
        assert "embedder_type" in status
        assert "total_chunks" in status
        print(f"RAG status: {status}")

    def test_embedder_fallback(self):
        """Test that MockEmbedder works when sentence-transformers unavailable."""
        from backend.tools.rag_search import get_embedder, MockEmbedder

        embedder = get_embedder()
        assert embedder is not None

        # Test encode
        vectors = embedder.encode(["test text", "another text"])
        assert vectors.shape[0] == 2
        assert vectors.shape[1] == 384  # MockEmbedder dim
        print(f"Embedder type: {type(embedder).__name__}, shape: {vectors.shape}")

    def test_ingest_and_search(self):
        """Test ingesting chunks and searching."""
        from backend.tools.rag_search import HybridRAG

        rag = HybridRAG()
        rag.clear()

        chunks = [
            {"text": "Corrosion limit is 5mm for pressure vessels.", "metadata": {"source": "sop-44.txt", "page": 1}},
            {"text": "Inspection must be done quarterly.", "metadata": {"source": "sop-44.txt", "page": 2}},
            {"text": "Safety meeting notes from November 2024.", "metadata": {"source": "memo-2024-11.msg"}},
        ]

        count = rag.ingest(chunks)
        assert count == 3

        results = rag.search("corrosion limit", top_k=2)
        assert len(results) > 0
        assert any("5mm" in r["text"] for r in results)
        print(f"Search results: {[r['text'][:50] for r in results]}")


class TestChunker:
    """Test the text chunking logic."""

    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        from backend.ingestion.chunker import chunk_text

        text = "This is sentence one. This is sentence two. This is sentence three."
        metadata = {"source": "test.txt", "doc_type": "Text"}

        chunks = chunk_text(text, metadata, chunk_size=50)
        assert len(chunks) > 0
        assert all("text" in c for c in chunks)
        assert all("metadata" in c for c in chunks)
        print(f"Chunks: {len(chunks)}")

    def test_chunk_metadata_preserved(self):
        """Test that metadata is preserved in chunks."""
        from backend.ingestion.chunker import chunk_text

        text = "Some content here."
        metadata = {"source": "test.txt", "page": 1, "doc_type": "PDF"}

        chunks = chunk_text(text, metadata)
        for chunk in chunks:
            assert chunk["metadata"]["source"] == "test.txt"
            assert chunk["metadata"]["page"] == 1
            assert chunk["metadata"]["doc_type"] == "PDF"


class TestCitationTagger:
    """Test the citation tagging system."""

    def test_tag_citations_basic(self):
        """Test basic citation tagging."""
        from backend.tools.citation_tagger import tag_citations

        text = "The corrosion limit is 5mm for pressure vessels."
        sources = [
            {"text": "Corrosion limit is 5mm for pressure vessels.", "metadata": {"source": "sop-44.txt", "page": 1}}
        ]

        tagged = tag_citations(text, sources)
        assert "[Source:" in tagged
        assert "sop-44.txt" in tagged
        print(f"Tagged: {tagged}")

    def test_tag_citations_no_match(self):
        """Test that non-matching text gets no citation."""
        from backend.tools.citation_tagger import tag_citations

        text = "This is completely unrelated content about weather."
        sources = [
            {"text": "Corrosion limit is 5mm.", "metadata": {"source": "sop-44.txt"}}
        ]

        tagged = tag_citations(text, sources)
        # Should not have citation for unrelated text
        assert "[Source:" not in tagged or tagged.count("[Source:") == 0


class TestCitationVerifier:
    """Test the CoT citation verifier."""

    def test_verifier_instantiation(self):
        """Test CitationVerifier can be instantiated."""
        from backend.agents.verifier import CitationVerifier

        verifier = CitationVerifier()
        assert verifier is not None

    def test_verifier_with_sources(self):
        """Test verification with matching sources."""
        from backend.agents.verifier import CitationVerifier

        verifier = CitationVerifier()
        text = "The corrosion limit is 5mm."
        sources = [
            {"text": "Corrosion limit is 5mm for pressure vessels.", "metadata": {"source": "sop-44.txt"}}
        ]

        result = verifier.verify(text, sources)
        assert "grounded" in result
        assert "reason" in result
        print(f"Verification: {result}")


class TestEndToEndIngestion:
    """Test end-to-end document ingestion via HTTP."""

    def test_ingest_endpoint(self):
        """Test POST /ingest with a directory of files."""
        from fastapi.testclient import TestClient
        from backend.main import app

        with TestClient(app) as client:
            response = client.post(
                "/ingest",
                json={"directory": "data/knowledge_base/sops"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "Ingestion complete"
            assert data["chunks_added"] > 0
            print(f"Ingest response: {data}")

    def test_ingest_nonexistent_directory(self):
        """Test POST /ingest with nonexistent directory."""
        from fastapi.testclient import TestClient
        from backend.main import app

        with TestClient(app) as client:
            response = client.post(
                "/ingest",
                json={"directory": "/nonexistent/path"},
            )
            assert response.status_code == 400


class TestEndToEndChat:
    """Test end-to-end chat with RAG grounding."""

    def test_chat_with_rag(self):
        """
        Spec test: ingest sop-44.txt, ask about corrosion limit,
        assert response contains '5mm' AND a citation tag like [Source: sop-44.txt].
        """
        from fastapi.testclient import TestClient
        from backend.main import app

        with TestClient(app) as client:
            # First ingest
            ingest_resp = client.post(
                "/ingest",
                json={"directory": "data/knowledge_base/sops"},
            )
            assert ingest_resp.status_code == 200
            ingest_data = ingest_resp.json()
            assert ingest_data["chunks_added"] > 0, f"Expected chunks, got {ingest_data}"

            # Then chat
            response = client.post(
                "/chat",
                json={"prompt": "What is the corrosion limit?"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "response" in data

            resp_text = data["response"]
            print(f"Chat response: {resp_text}")

            # Assert response contains the specific value from the document
            assert "5mm" in resp_text, f"Expected '5mm' in response: {resp_text}"

            # Assert response contains a citation tag referencing sop-44.txt
            assert "[Source: sop-44.txt" in resp_text, (
                f"Expected citation tag '[Source: sop-44.txt ...]' in response: {resp_text}"
            )

    def test_chat_without_rag(self):
        """Test /chat without prior ingestion."""
        from fastapi.testclient import TestClient
        from backend.main import app

        with TestClient(app) as client:
            response = client.post(
                "/chat",
                json={"prompt": "hi"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            print(f"Chat response: {data['response']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
