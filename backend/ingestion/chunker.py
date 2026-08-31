"""
Text Chunker: Splits text into overlapping chunks with metadata.
Used by ingestion pipelines for PDF and email processing.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    metadata: dict,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict]:
    """
    Split text into overlapping chunks and attach metadata to each.

    Args:
        text: The text to chunk.
        metadata: Dict with keys like source, page, doc_type, sender.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between adjacent chunks.

    Returns:
        List of dicts: {"text": "...", "metadata": {...}}.
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_len = len(text)

    # Ensure step is always positive to avoid infinite loop
    step = max(chunk_size - overlap, 1)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text_str = text[start:end].strip()

        if chunk_text_str:
            chunk_metadata = dict(metadata)  # shallow copy
            chunk_metadata["chunk_start"] = start
            chunk_metadata["chunk_end"] = end
            chunks.append({
                "text": chunk_text_str,
                "metadata": chunk_metadata,
            })

        # Advance with overlap
        start += step
        if start >= text_len:
            break

    logger.info(f"Chunked {len(chunks)} chunks from {len(text)} chars "
                f"(size={chunk_size}, overlap={overlap})")
    return chunks
