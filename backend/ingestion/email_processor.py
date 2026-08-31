"""
Email Processor: Extracts text from .eml and .msg email files.
Uses Python's built-in email library for .eml files.
"""

import email
import logging
from pathlib import Path
from typing import List, Dict

from backend.ingestion.chunker import chunk_text

logger = logging.getLogger(__name__)


def process_email(file_path: str) -> List[Dict]:
    """
    Extract text from an email file (.eml) and chunk it.

    Args:
        file_path: Path to the email file.

    Returns:
        List of chunk dicts with text and metadata.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"Email file not found: {file_path}")
        return []

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            msg = email.message_from_file(f)
    except Exception as e:
        logger.error(f"Failed to read email {file_path}: {e}")
        return []

    # Extract headers
    subject = msg.get("Subject", "(no subject)")
    sender = msg.get("From", "(unknown sender)")
    date = msg.get("Date", "(no date)")

    # Extract body
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body += payload.decode("utf-8", errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode("utf-8", errors="replace")

    if not body.strip():
        logger.warning(f"No text body found in {file_path}")
        return []

    # Prepend headers to body for context
    full_text = f"Subject: {subject}\nFrom: {sender}\nDate: {date}\n\n{body}"

    metadata = {
        "source": str(path.name),
        "doc_type": "Email",
        "sender": sender,
        "subject": subject,
        "date": date,
    }

    chunks = chunk_text(full_text, metadata)
    logger.info(f"Processed email {path.name}: {len(chunks)} chunks")
    return chunks
