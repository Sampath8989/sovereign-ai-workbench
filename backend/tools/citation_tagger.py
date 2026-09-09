"""
Citation Tagger: Appends source references to claims in generated text.
Uses keyword overlap and strict numerical verification to match sentences against retrieved sources.
Preserves list formatting and markdown structure without collisions.
"""

import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

_CITATION_PATTERN = re.compile(r'\s*\[Source:\s*[^\]]+\]', re.IGNORECASE)


def strip_citations(text: str) -> str:
    """
    Remove all citation markup like [Source: ...] from text.
    Ensures that ungrounded content or output from turns where no retrieval
    occurred contains zero citation tags.
    """
    if not text:
        return text
    return _CITATION_PATTERN.sub('', text).strip()


def tag_citations(text: str, sources: List[Dict], retrieval_invoked: bool = True) -> str:
    """
    Tag claims in generated text with source citations.

    Citations must ONLY be emitted when retrieval was actually invoked AND the
    retrieved chunk demonstrably supports the claim. If no retrieval occurred,
    no citation markup may appear in output, under any circumstance.
    Preserves list formatting and markdown structure: citation markers are appended
    after content text, never before headings or numbers.

    Args:
        text: The generated text to tag.
        sources: List of {"text": "...", "metadata": {...}} dicts from RAG search.
        retrieval_invoked: Whether retrieval was actually invoked for this request.

    Returns:
        Text with citation tags appended to matching sentences, or text with all
        citations stripped if retrieval did not occur.
    """
    if not text:
        return ""

    # If retrieval never occurred or no sources exist, no citation markup may appear
    # under any circumstance — strip any model-hallucinated citations.
    if not retrieval_invoked or not sources:
        return strip_citations(text)

    # Process line by line to preserve markdown lists, headers, and paragraph structure
    lines = text.split("\n")
    tagged_lines = []
    for line in lines:
        if not line.strip():
            tagged_lines.append(line)
            continue
        tagged_lines.append(_tag_line(line, sources))

    return "\n".join(tagged_lines)


def _tag_line(line: str, sources: List[Dict]) -> str:
    """Tag citations within a single line, preserving list/heading prefixes."""
    clean_line = strip_citations(line)
    if not clean_line.strip():
        return clean_line

    # Match list item, bullet, or heading prefix e.g. "1. ", "- ", "### ", "1. **Corrosion Limits:** "
    prefix_match = re.match(
        r'^(\s*(?:[\*\-\+]|\d+[\.\)]|#{1,6}|Item\s+\d+:?|Step\s+\d+:?)\s+(?:\*\*[^*]+\*\*:\s*)?)(.*)$',
        clean_line,
        re.IGNORECASE,
    )
    if prefix_match:
        prefix, content = prefix_match.group(1), prefix_match.group(2)
    else:
        # Check for bold prefix without list marker, e.g. "**Corrosion Limits:** ..."
        bold_match = re.match(r'^(\s*\*\*[^*]+\*\*:\s*)(.*)$', clean_line)
        if bold_match:
            prefix, content = bold_match.group(1), bold_match.group(2)
        else:
            prefix, content = "", clean_line

    if not content.strip():
        return clean_line

    sentences = _split_sentences(content)
    tagged_sentences = []
    for sentence in sentences:
        clean_sentence = strip_citations(sentence)
        if not clean_sentence.strip():
            continue

        best_match = _find_best_match(clean_sentence, sources)
        if best_match:
            metadata = best_match.get("metadata", {})
            source_name = metadata.get("source", "unknown")
            page = metadata.get("page")

            citation = f"[Source: {source_name}"
            if page is not None:
                citation += f", Page {page}"
            citation += "]"

            tagged = clean_sentence.rstrip() + " " + citation
        else:
            tagged = clean_sentence

        tagged_sentences.append(tagged)

    if not tagged_sentences:
        return clean_line

    return prefix + " ".join(tagged_sentences)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences. Handles common sentence boundaries without breaking list formatting."""
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = []
    for piece in raw:
        piece = piece.strip()
        if not piece:
            continue
        # If the previous piece was just a list marker or abbreviation like '1.' or 'A.' or 'e.g.'
        if sentences and re.match(r'^(?:\d{1,3}|[A-Za-z]|e\.g|i\.e)\.?$', sentences[-1]):
            sentences[-1] = sentences[-1] + " " + piece
        else:
            sentences.append(piece)
    return sentences


_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "and", "or", "but", "if", "then", "else", "when", "at",
    "by", "for", "with", "to", "from", "in", "on", "of", "it",
    "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "me", "him", "her", "us", "them", "my",
    "your", "his", "its", "our", "their", "what", "which",
    "who", "whom", "where", "how", "not", "no", "nor", "also",
    "per", "each", "can", "must", "should", "will", "would"
}


def _significant_words(text: str) -> set:
    """Tokenize into significant (non-stopword) words, normalized."""
    tokens = re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", text.lower())
    return {t for t in tokens if t and t not in _STOP_WORDS and len(t) > 1}


def _is_specific_token(token: str) -> bool:
    """
    A token is "specific" evidence when it carries a concrete fact: a number
    with/without a unit (5mm, 150psi, 15%), or a distinctive term (4+ chars,
    not a generic verb/noun).
    """
    if any(ch.isdigit() for ch in token):
        return True
    return len(token) >= 5 and token not in {
        "document", "report", "summary", "content", "created", "generated",
        "should", "would", "could", "about", "based", "request", "following",
        "including", "according", "provide", "please", "answer", "question",
        "there", "their", "these", "those", "other", "another", "after",
        "before", "between", "through", "during", "within", "without",
    }


def _extract_numbers_and_metrics(text: str) -> List[tuple]:
    """
    Extract numbers, fractions, and metrics from text.
    Returns list of (full_phrase, numeric_val, unit) tuples.
    E.g. [('5mm', '5', 'mm'), ('1/8 inch', '1/8', 'inch'), ('150 psi', '150', 'psi')].
    """
    # Exclude leading list markers like "1." or "- 2."
    clean_text = re.sub(r'^\s*(?:[\*\-\+]|\d+[\.\)]|Item\s+\d+:?|Step\s+\d+:?)\s+', '', text, flags=re.IGNORECASE)
    raw = re.findall(r"\b\d+(?:[./]\d+)?(?:\s*[a-zA-Z%]+)?\b", clean_text)
    items = []
    for r in raw:
        r_str = r.strip().lower()
        m = re.match(r"^(\d+(?:[./]\d+)?)\s*([a-zA-Z%]*)$", r_str)
        if m:
            val, unit = m.groups()
            items.append((r_str, val, unit))
    return items


def _number_present_in_source(full: str, val: str, unit: str, src_lower: str) -> bool:
    """
    Check if a numeric claim or unit-qualified quantity exists in the source text.
    Prevents false matches where e.g. 50mm matches 5mm or 1/8 matches unrelated text.
    """
    # 1. Search for full phrase (e.g. "5mm", "1/8 inch", "150 psi")
    full_norm = re.sub(r"\s+", r"\\s*", re.escape(full))
    if re.search(r"(?<![0-9/])" + full_norm + r"(?![0-9a-zA-Z])", src_lower):
        return True

    # 2. Search for numeric value with boundary that avoids partial digit matches
    val_esc = re.escape(val)
    if re.search(r"(?<![0-9/])" + val_esc + r"(?![0-9/])", src_lower):
        return True

    return False


def _verify_sentence_numbers_in_source(sentence: str, source_text: str) -> bool:
    """
    Verify that ANY number, fraction, or metric claimed in the sentence actually
    appears in the candidate source chunk. If the sentence makes a numerical claim
    not found in the chunk, the chunk cannot ground the claim.
    """
    nums = _extract_numbers_and_metrics(sentence)
    if not nums:
        return True

    src_lower = source_text.lower()
    for full, val, unit in nums:
        if not _number_present_in_source(full, val, unit, src_lower):
            logger.debug(f"Grounding rejected: sentence number '{full}' not found in source chunk")
            return False
    return True


def _find_best_match(sentence: str, sources: List[Dict]) -> Dict:
    """
    Find the source that demonstrably supports a sentence.

    A citation is only emitted when:
      1. All numerical quantities in the sentence actually appear in the source chunk.
      2. There is concrete word evidence:
         * >= 2 shared significant words AND overlap ratio >= 0.5, or
         * >= 1 shared specific token (number/unit or distinctive term).

    Returns the best matching source dict, or an empty dict when no source
    demonstrably supports the sentence (in which case NO citation is attached).
    """
    sentence_words = _significant_words(sentence)

    if not sentence_words:
        return {}

    best_score = 0.0
    best_source = {}

    for source in sources:
        source_text = source.get("text", "")
        source_words = _significant_words(source_text)

        if not source_words:
            continue

        # Strict numerical grounding check: numbers in sentence must exist in source chunk
        if not _verify_sentence_numbers_in_source(sentence, source_text):
            continue

        overlap = sentence_words & source_words
        ratio = len(overlap) / len(sentence_words)
        specific_overlap = {w for w in overlap if _is_specific_token(w)}

        # Evidence bar: strong overlap ratio with >=2 shared significant words
        # AND (a concrete specific token shared with the source OR near-total
        # overlap).
        if len(overlap) >= 2 and ratio >= 0.5 and (specific_overlap or ratio >= 0.8):
            score = ratio + (0.1 * len(specific_overlap))
            if score > best_score:
                best_score = score
                best_source = source

    return best_source
