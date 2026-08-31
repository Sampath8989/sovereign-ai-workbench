"""
Citation Tagger: Appends source references to claims in generated text.
Uses keyword overlap to match sentences against retrieved sources.
"""

import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)


def tag_citations(text: str, sources: List[Dict]) -> str:
    """
    Tag claims in generated text with source citations.

    For each sentence, checks if it semantically matches any of the
    retrieved sources using keyword overlap. If a match is found,
    appends a citation tag.

    Args:
        text: The generated text to tag.
        sources: List of {"text": "...", "metadata": {...}} dicts from RAG search.

    Returns:
        Text with citation tags appended to matching sentences.
    """
    if not text or not sources:
        return text

    # Split text into sentences
    sentences = _split_sentences(text)

    tagged_sentences = []
    for sentence in sentences:
        tagged = sentence
        best_match = _find_best_match(sentence, sources)
        if best_match:
            metadata = best_match.get("metadata", {})
            source_name = metadata.get("source", "unknown")
            page = metadata.get("page")
            doc_type = metadata.get("doc_type", "")

            citation = f"[Source: {source_name}"
            if page:
                citation += f", Page {page}"
            citation += "]"

            tagged = sentence.rstrip() + " " + citation

        tagged_sentences.append(tagged)

    result = " ".join(tagged_sentences)
    return result


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences. Handles common sentence boundaries."""
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s for s in sentences if s.strip()]


def _find_best_match(sentence: str, sources: List[Dict]) -> Dict:
    """
    Find the best matching source for a sentence using keyword overlap.
    Returns the best matching source dict, or empty dict if no good match.
    """
    sentence_words = set(sentence.lower().split())
    # Remove common stop words
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "being", "have", "has", "had", "do", "does", "did", "will",
                  "would", "could", "should", "may", "might", "shall", "can",
                  "and", "or", "but", "if", "then", "else", "when", "at",
                  "by", "for", "with", "to", "from", "in", "on", "of", "it",
                  "this", "that", "these", "those", "i", "you", "he", "she",
                  "we", "they", "me", "him", "her", "us", "them", "my",
                  "your", "his", "its", "our", "their", "what", "which",
                  "who", "whom", "where", "how", "not", "no", "nor"}
    sentence_words -= stop_words

    if not sentence_words:
        return {}

    best_score = 0
    best_source = {}

    for source in sources:
        source_text = source.get("text", "")
        source_words = set(source_text.lower().split()) - stop_words

        if not source_words:
            continue

        overlap = len(sentence_words & source_words)
        # Normalize by sentence length
        score = overlap / len(sentence_words) if sentence_words else 0

        if score > best_score and score > 0.2:  # minimum 20% overlap threshold
            best_score = score
            best_source = source

    return best_source
