"""
Hybrid RAG Search: Combines BM25 sparse search with Qdrant dense vector search.
Falls back to MockEmbedder if sentence-transformers is unavailable.
Falls back to in-memory search if Qdrant is unreachable.
"""

import hashlib
import logging
import os
import re
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# --- Query-keyed relevance gate ---
#
# A chunk is only a real "match" for a query when it demonstrably relates to
# that query. Dense vector similarity alone (especially with the MockEmbedder)
# can surface unrelated corpus content from other sessions — that is how
# "file-search" fabricated results and bled unrelated data across contexts.
# Every result must therefore share at least one significant token with the
# query before it can be returned; otherwise the search reports NO MATCH.

_RELEVANCE_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "and", "or", "but", "if",
    "then", "else", "when", "at", "by", "for", "with", "to", "from",
    "in", "on", "of", "it", "this", "that", "these", "those", "i", "you",
    "he", "she", "we", "they", "me", "him", "her", "us", "them", "my",
    "your", "his", "its", "our", "their", "what", "which", "who", "whom",
    "where", "how", "not", "no", "nor", "also", "per", "each", "about",
    "please", "tell", "give", "show", "find", "look", "need", "want",
}


def _query_tokens(query: str) -> set:
    """Significant query tokens used for the relevance gate."""
    tokens = re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", query.lower())
    return {t for t in tokens if t and t not in _RELEVANCE_STOP_WORDS and len(t) > 1}


def _chunk_tokens(text: str) -> set:
    """Significant chunk tokens used for the relevance gate."""
    tokens = re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", text.lower())
    return {t for t in tokens if t and t not in _RELEVANCE_STOP_WORDS and len(t) > 1}


def _is_relevant_to_query(chunk_text: str, query: str) -> bool:
    """
    True when the chunk shares at least one significant token with the query.
    This keys search results to the CURRENT query, preventing unrelated corpus
    content (from other contexts/sessions) from bleeding into results.
    """
    q_tokens = _query_tokens(query)
    if not q_tokens:
        return True  # nothing to gate on — caller decides
    c_tokens = _chunk_tokens(chunk_text)
    return bool(q_tokens & c_tokens)


def _extract_specific_targets(text: str) -> set:
    """Extract specific document/standard/project identifiers from text."""
    if not text or not isinstance(text, str):
        return set()
    targets = set()
    # Standards/SOPs: SOP-44, ISO-9001, API-510, ASME VIII, OSHA-1910, etc.
    for m in re.finditer(r'\b(sop|iso|api|asme|iec|osha|astm)[-\s_]?([a-z0-9\-]+)', text, re.I):
        prefix, num = m.group(1).lower(), m.group(2).lower()
        if num in ("the", "a", "an", "and", "for", "in", "of", "to", "or", "per", "is"):
            continue
        targets.add(f"{prefix}-{num}")
        targets.add(f"{prefix}{num}")
        targets.add(f"{num}")
    # Named projects: Project Omega, Project Titan
    for m in re.finditer(r'\bproject\s+([a-z0-9_-]+)', text, re.I):
        pname = m.group(1).lower()
        targets.add(f"project {pname}")
        targets.add(pname)
    # Explicit filenames: sop-44.txt, titan_specs.pdf
    for m in re.finditer(r'\b([a-z0-9_\-]+)\.(?:txt|pdf|docx|xlsx|md)\b', text, re.I):
        fname = m.group(1).lower()
        targets.add(fname)
    return targets


def _matches_query_targets(query: str, chunk_text: str, metadata: Optional[dict] = None) -> bool:
    """
    If the query specifies target document/standard/project entities (e.g. SOP-44, Project Titan),
    ensure the candidate chunk or its metadata contains matching target identifiers.
    Prevents cross-entity/cross-document bleeding.
    """
    q_targets = _extract_specific_targets(query)
    if not q_targets:
        return True  # No specific target constraint in query

    meta = metadata or {}
    meta_parts = [
        str(meta.get("source", "")),
        str(meta.get("source_file", "")),
        str(meta.get("collection", "")),
        str(meta.get("title", ""))
    ]
    c_meta_text = f"{chunk_text} {' '.join(meta_parts)}"
    c_targets = _extract_specific_targets(c_meta_text)
    c_lower = c_meta_text.lower()
    return bool((q_targets & c_targets) or any(t in c_lower for t in q_targets))


def _token_stem(t: str) -> str:
    """Basic morphological stemming for singular/plural and verb inflections."""
    t = t.lower()
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("es") and len(t) > 3:
        return t[:-2]
    if t.endswith("s") and not t.endswith("ss") and len(t) > 2:
        return t[:-1]
    if t.endswith("ing") and len(t) > 4:
        return t[:-3]
    if t.endswith("ed") and len(t) > 3:
        return t[:-2]
    return t


_DOC_META_TOKENS = {
    "sop", "protocol", "standard", "procedure", "document", "spec",
    "specification", "section", "per", "for", "vessel", "vessels"
}


# --- Embedder (lazy-loaded to avoid blocking startup) ---

_ST_AVAILABLE = False
_st_model = None
_ST_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_USE_MOCK = os.getenv("USE_MOCK_EMBEDDER", "1") == "1"  # Default: MockEmbedder for fast startup


def _load_real_embedder():
    """Lazy-load the real sentence-transformers model on first use."""
    global _ST_AVAILABLE, _st_model
    if _ST_AVAILABLE:
        return _st_model
    if _USE_MOCK:
        logger.info("USE_MOCK_EMBEDDER=1, skipping real embedder load.")
        return None
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading sentence-transformers model: {_ST_MODEL_NAME} (this may take a moment)...")
        _st_model = SentenceTransformer(_ST_MODEL_NAME)
        _ST_AVAILABLE = True
        logger.info(f"Loaded sentence-transformers model: {_ST_MODEL_NAME}")
        return _st_model
    except Exception as e:
        _ST_AVAILABLE = False
        _st_model = None
        logger.warning(f"sentence-transformers not available ({e}). Using MockEmbedder.")
        return None


class MockEmbedder:
    """
    Deterministic mock embedder. Returns a fixed-size vector seeded by the
    text's SHA-256 hash. Produces consistent results for the same input.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            # Use hash bytes to seed a deterministic random vector
            seed = int.from_bytes(h[:4], "big")
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dim).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-8  # normalize
            vectors.append(vec)
        return np.array(vectors)


def get_embedder():
    """Return the active embedder (real or mock). Lazy-loads real model on first use."""
    real = _load_real_embedder()
    if real is not None:
        return real
    return MockEmbedder(dim=384)


# --- Qdrant Client ---

_QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
_QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
_COLLECTION = "sovereign_kb"


def _get_qdrant_client():
    """Create a Qdrant client. Returns None if unreachable."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=_QDRANT_HOST, port=_QDRANT_PORT, timeout=5)
        # Quick health check
        client.get_collections()
        logger.info(f"Connected to Qdrant at {_QDRANT_HOST}:{_QDRANT_PORT}")
        return client
    except Exception as e:
        logger.warning(f"Qdrant unreachable at {_QDRANT_HOST}:{_QDRANT_PORT}: {e}")
        return None


def _ensure_collection(client, dim: int = 384):
    """Create the sovereign_kb collection if it doesn't exist."""
    try:
        from qdrant_client.models import VectorParams, Distance
        collections = client.get_collections().collections
        names = [c.name for c in collections]
        if _COLLECTION not in names:
            client.create_collection(
                collection_name=_COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {_COLLECTION}")
    except Exception as e:
        logger.warning(f"Failed to ensure Qdrant collection: {e}")


# --- BM25 Index ---

try:
    from rank_bm25 import BM25Plus, BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    logger.warning("rank-bm25 not installed. BM25 search unavailable.")


# --- Sensitive content detection (defense-in-depth) ---
#
# Heuristic patterns that flag chunks likely containing financial/confidential
# data.  This is NOT a complete classifier — it's a second layer of defense
# intended to catch the common case where ingestion-time metadata tagging
# misclassifies a chunk.  A determined adversarial document could still
# evade these simple keyword/regex checks.
#
# NOTE: Word-based patterns like "\bX million\b" are intentionally excluded
# because they produce false positives on legitimate engineering cost estimates
# (e.g., "turbine costs $2.5 million").  Only keyword + currency patterns used.
#
# B/M/K suffix pattern (\$X.XB, \$X.XM, \$XK) catches abbreviated financial
# amounts that don't use comma formatting.  This closes a gap where the exact
# leak format (e.g., "$8.3B") was missed.  The pattern is intentionally broad
# on B/M/K suffixes because any such abbreviation in an engineering context
# signals a high-value figure worth flagging for human review.  False positives
# on legitimate cost estimates are acceptable here — this is defense-in-depth,
# not a hard gate.
_SENSITIVE_PATTERNS = re.compile(
    r'\b(?:confidential|restricted|secret|budget|salary|payroll|compensation|'
    r'financial\s+statement|earnings|revenue|profit|loss|quarterly\s+report|'
    r'merger\s+value|acquisition\s+price|ipo|valuation|stock\s+price|'
    r'insider\s+trading|non-disclosure|nda)\b'
    r'|\$\s*\d{1,3}(?:,\d{3}){2,}(?:\.\d{2})?'  # currency >= $1,000,000 (3+ comma groups)
    r'|\$\s*\d{1,3}(?:\.\d{1,3})?\s*[BMK]\b',  # abbreviated: $8.3B, $500K, $3.2M
    re.IGNORECASE,
)

def contains_sensitive_content(text: str) -> bool:
    """
    Heuristic check: does this text contain patterns commonly associated
    with sensitive financial or confidential data?

    This is a defense-in-depth layer, NOT a guarantee.  See module-level
    docstring on _SENSITIVE_PATTERNS for limitations.
    """
    return bool(_SENSITIVE_PATTERNS.search(text))


# --- Singleton instance ---
_rag_instance: Optional["HybridRAG"] = None


def get_rag() -> "HybridRAG":
    """Return the singleton HybridRAG instance, creating it if needed."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = HybridRAG()
    return _rag_instance


class HybridRAG:
    """
    Hybrid RAG combining BM25 sparse search with Qdrant dense vector search.
    Falls back gracefully if Qdrant or sentence-transformers are unavailable.
    Use get_rag() to get the singleton instance.
    """

    def __init__(self):
        self.qdrant = _get_qdrant_client()
        self.embedder = get_embedder()
        self.dim = getattr(self.embedder, "dim", 384)
        if _ST_AVAILABLE and hasattr(self.embedder, "get_sentence_embedding_dimension"):
            self.dim = self.embedder.get_sentence_embedding_dimension()

        # BM25 state
        self._bm25_corpus: List[str] = []
        self._bm25_metadata: List[dict] = []
        self._bm25 = None

        # In-memory fallback for dense search when Qdrant is down
        self._in_memory_vectors: List[np.ndarray] = []
        self._in_memory_metadata: List[dict] = []
        self._in_memory_texts: List[str] = []

        if self.qdrant:
            _ensure_collection(self.qdrant, self.dim)
            self._sync_from_qdrant()

    def _sync_from_qdrant(self) -> int:
        """Rehydrate BM25 corpus and in-memory fallback from Qdrant collection if present."""
        if not self.qdrant:
            return 0
        try:
            records, _ = self.qdrant.scroll(
                collection_name=_COLLECTION,
                limit=10000,
                with_payload=True,
                with_vectors=True,
            )
            count = 0
            for rec in records:
                payload = rec.payload or {}
                text = payload.get("text")
                metadata = payload.get("metadata", {})
                if text and text not in self._bm25_corpus:
                    self._bm25_corpus.append(text)
                    self._bm25_metadata.append(metadata)
                    count += 1
                if text and text not in self._in_memory_texts:
                    self._in_memory_texts.append(text)
                    self._in_memory_metadata.append(metadata)
                    if rec.vector is not None:
                        self._in_memory_vectors.append(np.array(rec.vector, dtype=np.float32))
            if _BM25_AVAILABLE and self._bm25_corpus:
                tokenized = [doc.lower().split() for doc in self._bm25_corpus]
                try:
                    self._bm25 = BM25Plus(tokenized)
                except Exception:
                    self._bm25 = BM25Okapi(tokenized)
            if count > 0:
                logger.info(f"Synchronized {count} chunks from Qdrant into BM25 index")
            return count
        except Exception as e:
            logger.warning(f"Failed to sync chunks from Qdrant: {e}")
            return 0

    def clear(self) -> None:
        """Clear all in-memory state and Qdrant collection if present."""
        self._bm25_corpus.clear()
        self._bm25_metadata.clear()
        self._bm25 = None
        self._in_memory_vectors.clear()
        self._in_memory_metadata.clear()
        self._in_memory_texts.clear()
        if self.qdrant:
            try:
                self.qdrant.delete_collection(_COLLECTION)
                _ensure_collection(self.qdrant, self.dim)
            except Exception as e:
                logger.warning(f"Failed to clear Qdrant collection: {e}")

    def ingest(self, chunks: List[Dict]) -> int:
        """
        Ingest chunks into both Qdrant (dense) and BM25 (sparse).

        Args:
            chunks: List of {"text": "...", "metadata": {...}} dicts.

        Returns:
            Number of chunks ingested.
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]

        # --- Ingestion-time content classification (defense-in-depth) ---
        # If a chunk contains sensitive content patterns but is NOT already
        # tagged as restricted, auto-retag it into financials_restricted.
        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            collection = meta.get("collection", "")
            if collection != "financials_restricted" and contains_sensitive_content(text):
                logger.warning(
                    f"Content classifier: chunk {i} from '{meta.get('source', '?')}' "
                    f"contains sensitive patterns; re-tagging from '{collection}' "
                    f"to 'financials_restricted'"
                )
                metadatas[i] = dict(meta)  # copy to avoid mutating input
                metadatas[i]["collection"] = "financials_restricted"
                metadatas[i]["_content_retagged"] = True

        # Embed
        vectors = self.embedder.encode(texts)

        # Upsert to Qdrant with deterministic SHA-256 point IDs (independent of PYTHONHASHSEED)
        if self.qdrant:
            try:
                from qdrant_client.models import PointStruct
                points = [
                    PointStruct(
                        id=int.from_bytes(hashlib.sha256(texts[i].encode("utf-8")).digest()[:8], "big") % (2**63),
                        vector=vectors[i].tolist(),
                        payload={"text": texts[i], "metadata": metadatas[i]},
                    )
                    for i in range(len(texts))
                ]
                self.qdrant.upsert(collection_name=_COLLECTION, points=points)
                logger.info(f"Upserted {len(points)} points to Qdrant")
            except Exception as e:
                logger.warning(f"Qdrant upsert failed: {e}")

        # In-memory fallback for dense search (deduplicated by text content)
        for i, text in enumerate(texts):
            if text not in self._in_memory_texts:
                self._in_memory_vectors.append(vectors[i])
                self._in_memory_metadata.append(metadatas[i])
                self._in_memory_texts.append(text)

        # Add to BM25 corpus (deduplicated by text content to prevent duplicate score skew)
        for i, text in enumerate(texts):
            if text not in self._bm25_corpus:
                self._bm25_corpus.append(text)
                self._bm25_metadata.append(metadatas[i])

        if _BM25_AVAILABLE and self._bm25_corpus:
            tokenized = [doc.lower().split() for doc in self._bm25_corpus]
            try:
                self._bm25 = BM25Plus(tokenized)
            except Exception:
                self._bm25 = BM25Okapi(tokenized)

        return len(texts)

    def _compute_confidence(
        self,
        query: str,
        chunk_text: str,
        sparse_sc: float,
        dense_sc: float,
        metadata: Optional[dict] = None,
    ) -> float:
        """
        Calibrate composite confidence score against lexical, semantic, and property coverage signals.
        """
        q_tokens = _query_tokens(query)
        if not q_tokens:
            return 0.5

        meta = metadata or {}
        meta_parts = [
            str(meta.get("source", "")),
            str(meta.get("source_file", "")),
            str(meta.get("collection", "")),
            str(meta.get("title", ""))
        ]
        c_tokens = _chunk_tokens(f"{chunk_text} {' '.join(meta_parts)}")
        c_stems = {_token_stem(t) for t in c_tokens}

        prop_tokens = {
            t for t in q_tokens
            if t not in _DOC_META_TOKENS and not (t.isdigit() and len(t) <= 2)
        }
        if not prop_tokens:
            prop_tokens = q_tokens

        missing_prop = {t for t in prop_tokens if _token_stem(t) not in c_stems}
        prop_cov = (len(prop_tokens) - len(missing_prop)) / max(len(prop_tokens), 1)

        is_mock_embedder = (
            getattr(self.embedder, "__class__", None).__name__ == "MockEmbedder"
            or _USE_MOCK
        )

        sparse_component = min(0.60, (sparse_sc / 15.0) * 0.60) if sparse_sc > 0 else 0.0

        if not is_mock_embedder and dense_sc > 0:
            dense_component = min(0.40, max(0.0, (dense_sc - 0.3) / 0.7) * 0.40)
        else:
            dense_component = 0.0

        coverage_component = prop_cov * 0.40

        if is_mock_embedder:
            raw_conf = (sparse_component * 0.6) + (coverage_component * 1.0)
        else:
            raw_conf = (sparse_component * 0.5) + (dense_component * 0.5) + (coverage_component * 0.5)

        # Strongly penalize when critical property/metric tokens in the query are missing from the chunk
        if missing_prop and prop_cov < 0.90:
            raw_conf = min(0.18, raw_conf * (prop_cov ** 2))

        conf = max(0.05, min(0.98, raw_conf))
        return round(conf, 3)

    def search(self, query: str, top_k: int = 3, role: str = "engineer") -> List[Dict]:
        """
        Hybrid search: dense (Qdrant or in-memory) + sparse (BM25).
        Merges results via reciprocal rank fusion.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            role: User role for RBAC filtering (default: "engineer").
                  Engineers cannot see restricted collections; managers see all.

        Returns:
            List of {"text": "...", "metadata": {...}, "score": float}.
        """
        if _BM25_AVAILABLE and self._bm25 is None and self.qdrant:
            self._sync_from_qdrant()

        dense_results = self._dense_search(query, top_k * 2)
        sparse_results = self._sparse_search(query, top_k * 2)

        # Reciprocal rank fusion with score tracking
        rrf_scores: Dict[int, float] = {}
        rrf_data: Dict[int, Dict] = {}

        for rank, r in enumerate(dense_results):
            key = hash(r["text"])
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (60 + rank)
            entry = dict(r)
            entry["dense_rank"] = rank
            rrf_data[key] = entry

        for rank, r in enumerate(sparse_results):
            key = hash(r["text"])
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (60 + rank)
            if key not in rrf_data:
                entry = dict(r)
                rrf_data[key] = entry
            else:
                rrf_data[key]["sparse_score"] = r.get("sparse_score", 0.0)
            rrf_data[key]["sparse_rank"] = rank

        # Sort by RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        # --- RBAC filtering + query-keyed relevance gate + confidence threshold ---
        from backend.core.auth import is_restricted
        results = []
        for key in sorted_keys:
            entry = dict(rrf_data[key])
            entry["score"] = rrf_scores[key]

            # Calibrate composite confidence score
            dense_sc = entry.get("dense_score", 0.0)
            sparse_sc = entry.get("sparse_score", 0.0)
            entry["confidence"] = self._compute_confidence(
                query, entry.get("text", ""), sparse_sc, dense_sc, entry.get("metadata", {})
            )

            collection = entry.get("metadata", {}).get("collection", "")
            if is_restricted(collection, role):
                logger.info(f"RBAC: filtered restricted collection '{collection}' for role '{role}'")
                continue
            # --- Query-keyed relevance gate (session/context isolation) ---
            # Drop any chunk that does not demonstrably relate to the current
            # query. This is what prevents unrelated data from other contexts
            # or sessions from appearing as "matches" for this query.
            if not _is_relevant_to_query(entry.get("text", ""), query):
                logger.info(
                    f"Relevance gate: dropping chunk from '{collection}' "
                    f"— no shared tokens with query '{query[:60]}'"
                )
                continue
            # --- Target entity / document constraint gate ---
            if not _matches_query_targets(query, entry.get("text", ""), entry.get("metadata", {})):
                logger.info(
                    f"Target gate: dropping chunk from '{collection}' "
                    f"— does not match target document/entity in query '{query[:60]}'"
                )
                continue
            # --- Confidence threshold gate ---
            if entry["confidence"] < 0.20:
                logger.info(
                    f"Confidence gate: dropping chunk from '{collection}' "
                    f"— confidence {entry['confidence']:.2f} below threshold 0.20"
                )
                continue
            # --- Retrieval-time content filtering (defense-in-depth) ---
            if role == "engineer" and contains_sensitive_content(entry.get("text", "")):
                logger.warning(
                    f"Content filter: excluding chunk from '{collection}' "
                    f"for role '{role}' — sensitive content detected in text"
                )
                continue
            results.append(entry)
            if len(results) >= top_k:
                break

        # Explicit no-match semantics: an empty list means "no match found".
        if not results:
            logger.info(f"RAG search: NO MATCH found for query '{query[:80]}'")
        return results

    def _dense_search(self, query: str, top_k: int) -> List[Dict]:
        """Dense vector search via Qdrant or in-memory fallback."""
        query_vec = self.embedder.encode([query])[0]

        # Try Qdrant first
        if self.qdrant:
            try:
                # qdrant-client >= 1.7 uses query_points; older versions use search
                if hasattr(self.qdrant, "query_points"):
                    from qdrant_client.models import Query, Filter
                    resp = self.qdrant.query_points(
                        collection_name=_COLLECTION,
                        query=query_vec.tolist(),
                        limit=top_k,
                    )
                    return [
                        {
                            "text": r.payload["text"],
                            "metadata": r.payload.get("metadata", {}),
                            "dense_score": float(getattr(r, "score", 0.0)),
                        }
                        for r in resp.points
                    ]
                else:
                    results = self.qdrant.search(
                        collection_name=_COLLECTION,
                        query_vector=query_vec.tolist(),
                        limit=top_k,
                    )
                    return [
                        {
                            "text": r.payload["text"],
                            "metadata": r.payload.get("metadata", {}),
                            "dense_score": float(getattr(r, "score", 0.0)),
                        }
                        for r in results
                    ]
            except Exception as e:
                logger.warning(f"Qdrant search failed, using in-memory: {e}")

        # In-memory fallback: cosine similarity
        if not self._in_memory_vectors:
            return []

        query_np = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        scores = []
        for i, vec in enumerate(self._in_memory_vectors):
            sim = float(np.dot(query_np, vec / (np.linalg.norm(vec) + 1e-8)))
            scores.append((sim, i))
        scores.sort(reverse=True)

        return [
            {
                "text": self._in_memory_texts[i],
                "metadata": self._in_memory_metadata[i],
                "dense_score": float(sim),
            }
            for sim, i in scores[:top_k]
        ]

    def _sparse_search(self, query: str, top_k: int) -> List[Dict]:
        """BM25 sparse search."""
        if not _BM25_AVAILABLE or self._bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "text": self._bm25_corpus[i],
                "metadata": self._bm25_metadata[i] if i < len(self._bm25_metadata) else {},
                "sparse_score": float(scores[i]),
            }
            for i in top_indices
            if scores[i] > 0
        ]

    def get_status(self) -> dict:
        """Return current status of the RAG system."""
        return {
            "qdrant_connected": self.qdrant is not None,
            "embedder_type": type(self.embedder).__name__,
            "bm25_available": _BM25_AVAILABLE and self._bm25 is not None,
            "total_chunks": len(self._bm25_corpus),
            "collection": _COLLECTION,
        }
