"""
Semantic Router: Classifies user prompts into task categories.
Deterministic keyword-based routing with regex word-boundary matching
and simple negation heuristics.

Known limitations (documented):
- Negation handling covers common patterns within a ~5-word window.
  Complex multi-clause negation is out of scope for a deterministic
  keyword router.
- Word-boundary matching uses \b regex anchors to prevent substring
  collisions (e.g., "encode" no longer triggers CODE).
"""

import re
import logging

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns with word-boundary anchors for each category.
# Using \b ensures "encode" doesn't match \bcode\b, "profile" doesn't
# match \bfile\b, etc.
CODE_PATTERN = re.compile(r'\b(code|script|execute|function|class|program|debug|refactor|html|css|js|sql|regex|docx|pptx|xlsx|spreadsheet|word document|powerpoint)\b')
FILE_PATTERN = re.compile(r'\b(read|file|write)\b')
VISION_PATTERN = re.compile(r'\b(image|scan|drawing|photo|diagram|picture|visual|ocr|p&id)\b')
REASONING_PATTERN = re.compile(r'\b(math|calculate|calculation|equation|solve|integral|derivative|algebra|proof|reason|reasoning|why|step-by-step|logic|verify|verification|audit|evaluate)\b')
SYNTHESIS_PATTERN = re.compile(r'\b(architecture|comprehensive|deep dive|strategic|executive summary|detailed analysis)\b')

# Negation markers — words/phrases that negate a following keyword.
# Checked within a ~3-word window before the keyword, but only within
# the same clause (no comma/period between marker and keyword).
NEGATION_MARKERS = re.compile(
    r'\b(do\s+not|don\'t|dont|never|avoid|without|no|skip|refrain\s+from)\b'
)

# Maximum word distance between negation marker and keyword to count
# as "negating" that keyword.
_NEGATION_WINDOW = 5


def _is_negated(lower: str, keyword_match: re.Match) -> bool:
    """
    Check if a keyword match is preceded by a negation marker within
    a ~3-word window AND within the same clause (no comma/period between
    marker and keyword). Simple heuristic, not full NLP negation scope.
    """
    match_start = keyword_match.start()
    # Look at the text before the keyword match
    preceding = lower[:match_start]
    
    # Find the nearest clause boundary (comma, period, semicolon, dash)
    # Only check negation within the same clause
    clause_start = 0
    for sep in [',', '.', ';', ' - ', ' — ']:
        idx = preceding.rfind(sep)
        if idx > clause_start:
            clause_start = idx + len(sep)
    
    clause_text = preceding[clause_start:]
    words_before = clause_text.split()
    # Check last _NEGATION_WINDOW words for negation markers
    window_words = words_before[-_NEGATION_WINDOW:] if len(words_before) >= _NEGATION_WINDOW else words_before
    window_text = " ".join(window_words)
    return bool(NEGATION_MARKERS.search(window_text))


def _has_keyword(pattern: re.Pattern, lower: str) -> bool:
    """
    Check if the pattern matches in the lowercased prompt,
    excluding matches that are negated by a preceding negation marker.
    Returns True only if at least one non-negated match exists.
    """
    for match in pattern.finditer(lower):
        if not _is_negated(lower, match):
            return True
    return False


class SemanticRouter:
    """Class-based semantic router that delegates to route_task()."""

    def route_task(self, prompt: str) -> str:
        """Route a user prompt to the appropriate task category."""
        return route_task(prompt)


def route_task(prompt: str) -> str:
    """
    Route a user prompt to the appropriate task category.

    Args:
        prompt: The user's input prompt.

    Returns:
        One of: "CODE", "FILE", "VISION", "TEXT"
    """
    lower = prompt.lower()

    # CODE routing — word-boundary match, negation-aware
    if _has_keyword(CODE_PATTERN, lower):
        logger.info(f"Routing prompt to CODE: {prompt[:60]}")
        return "CODE"

    # FILE routing — word-boundary match, negation-aware
    if _has_keyword(FILE_PATTERN, lower):
        logger.info(f"Routing prompt to FILE: {prompt[:60]}")
        return "FILE"

    # VISION routing — word-boundary match, negation-aware
    if _has_keyword(VISION_PATTERN, lower):
        logger.info(f"Routing prompt to VISION: {prompt[:60]}")
        return "VISION"

    # Default: TEXT
    logger.info(f"Routing prompt to TEXT: {prompt[:60]}")
    return "TEXT"


def auto_select_model(prompt: str) -> str:
    """
    Intelligently select the best local model file on disk based on the prompt's intent.
    Routes to:
      - DeepSeek R1 7B for math, complex logic, and step-by-step reasoning
      - Qwen 2.5 Coder 7B for coding, script execution, and deliverable synthesis
      - LLaVA 7B for visual reasoning, OCR, and diagram analysis
      - Phi-4 14B for deep architecture analysis and comprehensive synthesis
      - Qwen 2.5 7B Instruct / Qwen 1.5 4B for general conversational queries
    """
    from backend.config import _model_file_valid, get_coder_model, get_router_model
    lower = prompt.lower()

    # 1. Math & Step-by-Step Reasoning -> DeepSeek R1 7B
    if _has_keyword(REASONING_PATTERN, lower):
        if _model_file_valid("deepseek-r1-7b.gguf"):
            logger.info("Auto-selected DeepSeek R1 7B for reasoning/math task")
            return "deepseek-r1-7b.gguf"
        if _model_file_valid("phi4-14b.gguf"):
            logger.info("Auto-selected Phi-4 14B for reasoning task (fallback)")
            return "phi4-14b.gguf"

    # 2. Vision & Diagram / OCR Tasks -> LLaVA 7B
    if _has_keyword(VISION_PATTERN, lower):
        if _model_file_valid("llava-7b.gguf"):
            logger.info("Auto-selected LLaVA 7B for vision task")
            return "llava-7b.gguf"

    # 3. Coding & Deliverable Generation -> Qwen 2.5 Coder 7B
    if _has_keyword(CODE_PATTERN, lower):
        if _model_file_valid("qwen2.5-coder-7b-instruct-q3_k_m.gguf"):
            logger.info("Auto-selected Qwen 2.5 Coder 7B for coding/deliverable task")
            return "qwen2.5-coder-7b-instruct-q3_k_m.gguf"
        if _model_file_valid("qwen2.5-coder-3b-instruct-q4_k_m.gguf"):
            logger.info("Auto-selected Qwen 2.5 Coder 3B for coding task")
            return "qwen2.5-coder-3b-instruct-q4_k_m.gguf"

    # 4. Deep Synthesis & Comprehensive Analysis -> Phi-4 14B
    if _has_keyword(SYNTHESIS_PATTERN, lower):
        if _model_file_valid("phi4-14b.gguf"):
            logger.info("Auto-selected Phi-4 14B for deep synthesis task")
            return "phi4-14b.gguf"

    # 5. General Chat & QA -> Qwen 2.5 7B Instruct -> Qwen 2.5 7B -> Qwen 1.5 4B -> Router
    if _model_file_valid("qwen2.5-7b-instruct-q3_k_m.gguf"):
        logger.info("Auto-selected Qwen 2.5 7B Instruct for general task")
        return "qwen2.5-7b-instruct-q3_k_m.gguf"
    if _model_file_valid("qwen2.5-7b.gguf"):
        logger.info("Auto-selected Qwen 2.5 7B for general task")
        return "qwen2.5-7b.gguf"
    if _model_file_valid("qwen1_5-4b-chat-q4_k_m.gguf"):
        logger.info("Auto-selected Qwen 1.5 4B for general task")
        return "qwen1_5-4b-chat-q4_k_m.gguf"

    return get_router_model()

