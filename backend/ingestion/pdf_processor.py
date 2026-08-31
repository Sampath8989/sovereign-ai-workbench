"""
PDF Processor: Extracts text from PDF files page by page.
Uses pypdf for extraction, then chunks via the chunker module.
"""

import logging
from pathlib import Path
from typing import List, Dict

from backend.ingestion.chunker import chunk_text

logger = logging.getLogger(__name__)


def process_pdf(file_path: str) -> List[Dict]:
    """
    Extract text from a PDF file page by page and chunk it.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of chunk dicts with text and metadata.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"PDF file not found: {file_path}")
        return []

    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf not installed. Cannot process PDF files.")
        return []

    try:
        reader = PdfReader(str(path))
    except Exception as e:
        logger.error(f"Failed to read PDF {file_path}: {e}")
        return []

    all_chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                metadata = {
                    "source": str(path.name),
                    "page": page_num,
                    "doc_type": "PDF",
                }
                chunks = chunk_text(page_text, metadata)
                all_chunks.extend(chunks)
        except Exception as e:
            logger.warning(f"Failed to extract page {page_num} from {file_path}: {e}")

    logger.info(f"Processed PDF {path.name}: {len(all_chunks)} chunks "
                f"from {len(reader.pages)} pages")
    return all_chunks
