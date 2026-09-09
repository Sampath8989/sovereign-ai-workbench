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

# ---------------------------------------------------------------------------
# Intent / complexity classification (deterministic, runs BEFORE model
# selection and BEFORE retrieval invocation).
#
# Greetings and small talk are NOT tasks: they must never be escalated to a
# large model and must never trigger retrieval. This classifier is purely
# rule-based so the behavior is identical on every call — no LLM involvement.
# ---------------------------------------------------------------------------

# Exact-match greetings (normalized: lowercased, trailing punctuation stripped)
_GREETING_EXACT = {
    "hello", "hi", "hey", "howdy", "greetings", "sup", "yo", "hiya",
    "good morning", "good afternoon", "good evening", "good day", "morning",
    "afternoon", "evening", "welcome", "hi there", "hello there",
    "hey there", "hi workbench", "hello workbench", "hey workbench",
    "greetings workbench", "greetings assistant", "hello assistant", "hi assistant",
    "good morning workbench", "good afternoon workbench", "good evening workbench",
    "good morning assistant", "good afternoon assistant", "good evening assistant",
}

# Short phrasal greetings, e.g. "hello there", "hey everyone", "good morning bot"
_GREETING_SHORT_RE = re.compile(
    r'^(?:hi|hey|hello|yo|sup|hiya|howdy|good\s+(?:morning|afternoon|evening|day)|greetings|welcome)'
    r'(?:\s+(?:there|workbench|assistant|bot|ai|all|everyone|guys|folks|friend))?[\s!.,?]*$',
    re.IGNORECASE,
)

# Small talk / conversational openers that are not tasks (no retrieval, no
# tool execution, smallest model or template response).
_SMALLTALK_EXACT = {
    "what can you do", "who are you", "what are you", "tell me about yourself",
    "what do you do", "how are you", "how are you doing", "how's it going",
    "how is it going", "what's up", "whats up", "what is up", "are you there",
    "are you awake", "you there", "can you hear me", "test", "testing",
    "thanks", "thank you", "thanks a lot", "thank you very much", "ty", "thx",
    "help", "help me", "can you help me", "can you help", "good job", "nice",
    "ok", "okay", "cool", "great", "sounds good", "introduce yourself",
    "what is this workbench", "about workbench", "what is this",
}

_SMALLTALK_PREFIX_RE = re.compile(
    r'^(?:thanks?|thank you|ty|thx|ok|okay|hiya|yo|hey|hello|cool|great|sounds good|nice)[\s!.,?]*$',
    re.IGNORECASE,
)

_SMALLTALK_PATTERNS = re.compile(
    r'^(?:how\s+are\s+you(?:\s+doing)?|how(?:\'s|\s+is)\s+it\s+going|what(?:\'s|\s+is)\s+up|'
    r'what\s+can\s+you\s+do|who\s+are\s+you|what\s+are\s+you|tell\s+me\s+about\s+yourself|'
    r'what\s+do\s+you\s+do|can\s+you\s+help(?:\s+me)?|help\s+me|introduce\s+yourself)[\s!.,?]*$',
    re.IGNORECASE,
)

# Bare filename regex (Bug 4)
_BARE_FILENAME_RE = re.compile(
    r'^\s*[\'"\u201c\u2018]?(?:[a-zA-Z]:[\\/])?(?:[a-zA-Z0-9_\-./\\]+[\\/])?[a-zA-Z0-9_\-.]+\.(?:pdf|txt|docx|xlsx|csv|json|md|py|log|png|jpg|jpeg|pptx)[\'"\u201d\u2019]?\s*$',
    re.IGNORECASE,
)


def is_bare_filename(prompt: str) -> bool:
    """Check if prompt is just a plain-text filename or path."""
    if not isinstance(prompt, str):
        return False
    return bool(_BARE_FILENAME_RE.match(prompt.strip()))


def _normalize_for_intent(prompt: str) -> str:
    """Lowercase and strip trailing punctuation for deterministic matching."""
    if not isinstance(prompt, str):
        return ""
    cleaned = re.sub(r'[!.,?]+\s*$', '', prompt.strip().lower())
    return re.sub(r'\s+', ' ', cleaned).strip()


def classify_intent(prompt: str) -> str:
    """
    Classify a user prompt as ``"GREETING"``, ``"SMALLTALK"``, ``"BARE_FILENAME"``, or ``"TASK"``.

    Deterministic, rule-based, and cheap. Call this BEFORE model selection and
    BEFORE retrieval so trivial inputs can never be escalated to a large model
    or sent through the RAG pipeline.

    Returns:
        One of: "GREETING", "SMALLTALK", "BARE_FILENAME", "TASK"
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return "TASK"  # empty input is not a greeting; let downstream handle it

    # Check bare filename first
    if is_bare_filename(prompt):
        return "BARE_FILENAME"

    normalized = _normalize_for_intent(prompt)
    if not normalized:
        return "TASK"

    # Pure greetings
    if normalized in _GREETING_EXACT or _GREETING_SHORT_RE.match(normalized):
        return "GREETING"

    # Pure small talk
    if (
        normalized in _SMALLTALK_EXACT
        or _SMALLTALK_PREFIX_RE.match(normalized)
        or _SMALLTALK_PATTERNS.match(normalized)
    ):
        return "SMALLTALK"

    # Compound greeting + small talk (e.g. "hi, how are you", "hello! what can you do?")
    # Check if text starts with greeting prefix and remainder is smalltalk
    greeting_prefix_match = re.match(
        r'^(?:hi|hey|hello|yo|howdy|greetings|good\s+(?:morning|afternoon|evening|day))'
        r'(?:\s+(?:there|workbench|assistant|bot|friend))?[\s,!.:;?]+(.*)$',
        normalized,
        re.IGNORECASE,
    )
    if greeting_prefix_match:
        remainder = greeting_prefix_match.group(1).strip()
        remainder = re.sub(r'[!.,?]+$', '', remainder).strip()
        if not remainder or remainder in _SMALLTALK_EXACT or _SMALLTALK_PATTERNS.match(remainder) or _GREETING_SHORT_RE.match(remainder):
            return "GREETING"

    return "TASK"


def is_trivial_query(prompt: str) -> bool:
    """Return True when a prompt is greeting/small talk or bare filename (not a task)."""
    return classify_intent(prompt) in ("GREETING", "SMALLTALK", "BARE_FILENAME")


def get_direct_response(prompt: str) -> str:
    """
    Return a deterministic template response for greeting/small-talk prompts.
    Used by the synthesizer so trivial queries never invoke a model or RAG.
    """
    intent = classify_intent(prompt)
    if intent == "BARE_FILENAME":
        return (
            "I cannot open or process files from a plain-text filename. "
            "Please use the file upload control (paperclip icon) next to the chat box "
            "to upload your file to the sandbox first, then ask me to analyze it."
        )
    if intent == "GREETING":
        return (
            "Hello! I'm the Sovereign AI Workbench — an air-gapped, locally-hosted AI assistant. "
            "I can help you with: generating documents (Word, PowerPoint, Excel), "
            "analyzing P&ID diagrams, reading handwritten notes, calculating values, "
            "and more — all running entirely on your local hardware with no external network access. "
            "What would you like me to help you with?"
        )
    if intent == "SMALLTALK":
        return (
            "I'm the Sovereign AI Workbench, your local air-gapped AI assistant. "
            "I can generate Word/PowerPoint/Excel deliverables, answer questions from the "
            "knowledge base, read handwritten notes, and analyze P&ID diagrams. "
            "What would you like me to do?"
        )
    return ""


# Patterns indicating retrieval intent against the internal knowledge base / domain corpus
_DOMAIN_RETRIEVAL_PATTERNS = [
    # Standards, SOPs, specifications, protocols
    re.compile(r"\b(?:sop[-\s]?\d+|iso[-\s]?\d+|asme|astm|api[-\s]?\d+|osha|ieee|iec)\b", re.IGNORECASE),
    re.compile(r"\b(?:sop|standard\s+operating\s+procedure|protocols?|specifications?|specs?|guidelines?|policies|policy|manuals?|handbooks?|bulletins?)\b", re.IGNORECASE),
    # Industrial / engineering equipment & domain metrics
    re.compile(r"\b(?:corrosion|corrosion\s+limit|corrosion\s+depth|inspection\s+limit|inspection\s+frequency|ultrasonic\s+thickness)\b", re.IGNORECASE),
    re.compile(r"\b(?:pressure\s+vessels?|storage\s+tanks?|heat\s+exchangers?|boilers?|piping|maintenance\s+logs?)\b", re.IGNORECASE),
    # Corporate / financial domain data
    re.compile(r"\b(?:q[1-4]\s+budget|project\s+omega|financial\s+data|financial\s+statements?|confidential\s+budget|earnings\s+report)\b", re.IGNORECASE),
    re.compile(r"\b(?:engineering\s+procedures?|facility\s+procedures?|safety\s+procedures?)\b", re.IGNORECASE),
    # Explicit search / lookup directives
    re.compile(r"\b(?:search\s+(?:the\s+)?(?:kb|knowledge\s+base|documents?|archive|records?)|in\s+(?:the\s+)?(?:kb|knowledge\s+base|document|manual|protocol|sop)|per\s+sop|according\s+to\s+(?:the\s+)?(?:sop|document|manual|protocol|spec))\b", re.IGNORECASE),
    # Document filenames
    re.compile(r"\b[a-zA-Z0-9_\-]+\.(?:pdf|txt|docx|xlsx|csv|eml|msg)\b", re.IGNORECASE),
]

# Patterns for non-retrieval tasks (math, coding, tool generation, vision)
_NON_RETRIEVAL_PATTERNS = [
    re.compile(r"^(?:calculate|solve:?|math\b|\d+\s*[\+\-\*\/])", re.IGNORECASE),
    re.compile(r"\b(?:create\s+a\s+(?:word\s+document|powerpoint|presentation)|generate\s+a\s+spreadsheet)\b", re.IGNORECASE),
    re.compile(r"\b(?:extract\s+(?:the\s+)?topology|read\s+the\s+handwriting|analyze\s+the\s+nameplate)\b", re.IGNORECASE),
]


def should_invoke_retrieval(prompt: str) -> bool:
    """
    Determine if a prompt requires retrieval from the sovereign knowledge base.

    Returns False for:
    - Greetings, small talk, bare filenames
    - Pure general knowledge / definitions ('What is a neural network?', 'capital of France')
    - General concept comparisons ('supervised vs unsupervised learning')
    - Math / calculations / code generation / vision tool invocations

    Returns True for:
    - Domain standards, SOPs, protocols, specs (SOP-44, ISO, ASME)
    - Plant/engineering domain metrics (corrosion limits, inspection intervals)
    - Corporate/financial data (Project Omega, Q4 budget)
    - Explicit document/KB search requests
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return False
    if is_trivial_query(prompt):
        return False
    for pat in _NON_RETRIEVAL_PATTERNS:
        if pat.search(prompt):
            return False
    for pat in _DOMAIN_RETRIEVAL_PATTERNS:
        if pat.search(prompt):
            return True
    return False


# Pre-compiled regex patterns with word-boundary anchors for each category.
# Using \b ensures "encode" doesn't match \bcode\b, "profile" doesn't
# match \bfile\b, etc.
CODE_PATTERN = re.compile(r'\b(code|script|execute|function|class|program|debug|refactor|html|css|js|sql|regex|docx|pptx|xlsx|spreadsheet|word document|powerpoint)\b')
FILE_PATTERN = re.compile(r'\b(read|file|write)\b')
VISION_PATTERN = re.compile(r'\b(image|scan|drawing|photo|diagram|picture|visual|ocr|p&id|png|jpg|jpeg|webp|bmp|gif)\b')
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

    has_image_ref = bool(re.search(r'\.(?:png|jpg|jpeg|webp|bmp|gif|tiff|tif)\b', lower))

    # VISION routing — word-boundary match or image file reference, negation-aware
    if _has_keyword(VISION_PATTERN, lower) or has_image_ref:
        logger.info(f"Routing prompt to VISION: {prompt[:60]}")
        return "VISION"

    # FILE routing — word-boundary match, negation-aware
    if _has_keyword(FILE_PATTERN, lower):
        logger.info(f"Routing prompt to FILE: {prompt[:60]}")
        return "FILE"

    # Default: TEXT
    logger.info(f"Routing prompt to TEXT: {prompt[:60]}")
    return "TEXT"


def auto_select_model(prompt: str) -> str:
    """
    Intelligently select the best local model file on disk based on the prompt's intent.
    Routes to:
      - DeepSeek R1 7B or 3B for math, complex logic, and step-by-step reasoning
      - Qwen 2.5 Coder 7B or 3B for coding, script execution, and deliverable synthesis
      - LLaVA 7B for visual reasoning, OCR, and diagram analysis
      - Phi-4 14B or 3B for deep architecture analysis and comprehensive synthesis
      - Llama 3.2 3B / Qwen 2.5 3B / Qwen 2.5 7B for general conversational queries
    """
    import os
    from backend.config import _model_file_valid, get_coder_model, get_router_model
    lower = prompt.lower()
    prefer_3b = os.getenv("PREFER_3B_MODELS", "false").lower() in ("true", "1", "yes")

    # 0. Greetings / small talk — deterministic, BEFORE all other routing.
    #    Trivial queries must never be escalated to a 7B/14B model. Route them
    #    to the smallest available model (0.5B fallback -> 3B -> router default).
    if is_trivial_query(prompt):
        for m in [
            "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "qwen2.5-3b-instruct-q4_k_m.gguf",
            "qwen2.5-3b-instruct-q5_k_m.gguf",
            "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
            "llama-3.2-3b-instruct-q4_k_m.gguf",
        ]:
            if _model_file_valid(m):
                logger.info(
                    f"Auto-selected {m} for {classify_intent(prompt).lower()} "
                    "(smallest available model, retrieval bypassed)"
                )
                return m
        # No small model on disk: fall through to the tier router default,
        # which itself prefers small models before larger ones.
        logger.info("No small model on disk for greeting; using router default")
        return get_router_model()

    # 1. Math & Step-by-Step Reasoning
    if _has_keyword(REASONING_PATTERN, lower):
        if prefer_3b:
            if _model_file_valid("qwen2.5-3b-instruct-q4_k_m.gguf"):
                logger.info("Auto-selected Qwen 2.5 3B for reasoning task (3B mode)")
                return "qwen2.5-3b-instruct-q4_k_m.gguf"
            if _model_file_valid("llama-3.2-3b-instruct-q4_k_m.gguf"):
                logger.info("Auto-selected Llama 3.2 3B for reasoning task (3B mode)")
                return "llama-3.2-3b-instruct-q4_k_m.gguf"
        if _model_file_valid("deepseek-r1-7b.gguf"):
            logger.info("Auto-selected DeepSeek R1 7B for reasoning/math task")
            return "deepseek-r1-7b.gguf"
        if _model_file_valid("phi4-14b.gguf"):
            logger.info("Auto-selected Phi-4 14B for reasoning task (fallback)")
            return "phi4-14b.gguf"
        if _model_file_valid("qwen2.5-3b-instruct-q4_k_m.gguf"):
            return "qwen2.5-3b-instruct-q4_k_m.gguf"
        if _model_file_valid("llama-3.2-3b-instruct-q4_k_m.gguf"):
            return "llama-3.2-3b-instruct-q4_k_m.gguf"

    # 2. Vision & Diagram / OCR Tasks -> LLaVA 7B (or 3B in 3B mode)
    if _has_keyword(VISION_PATTERN, lower) or re.search(r'\.(?:png|jpg|jpeg|webp|bmp|gif|tiff|tif)\b', lower):
        if prefer_3b:
            if _model_file_valid("llama-3.2-3b-instruct-q4_k_m.gguf"):
                logger.info("Auto-selected Llama 3.2 3B for vision task (3B mode)")
                return "llama-3.2-3b-instruct-q4_k_m.gguf"
            if _model_file_valid("qwen2.5-3b-instruct-q4_k_m.gguf"):
                logger.info("Auto-selected Qwen 2.5 3B for vision task (3B mode)")
                return "qwen2.5-3b-instruct-q4_k_m.gguf"
        if _model_file_valid("llava-7b.gguf"):
            logger.info("Auto-selected LLaVA 7B for vision task")
            return "llava-7b.gguf"

    # 3. Coding & Deliverable Generation -> Qwen 2.5 Coder 7B / 3B
    if _has_keyword(CODE_PATTERN, lower):
        if prefer_3b:
            if _model_file_valid("qwen2.5-coder-3b-instruct-q4_k_m.gguf"):
                logger.info("Auto-selected Qwen 2.5 Coder 3B for coding task (3B mode)")
                return "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
            if _model_file_valid("qwen2.5-coder-3b-instruct-q5_k_m.gguf"):
                return "qwen2.5-coder-3b-instruct-q5_k_m.gguf"
        if _model_file_valid("qwen2.5-coder-7b-instruct-q3_k_m.gguf"):
            logger.info("Auto-selected Qwen 2.5 Coder 7B for coding/deliverable task")
            return "qwen2.5-coder-7b-instruct-q3_k_m.gguf"
        if _model_file_valid("qwen2.5-coder-3b-instruct-q4_k_m.gguf"):
            logger.info("Auto-selected Qwen 2.5 Coder 3B for coding task")
            return "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
        if _model_file_valid("qwen2.5-coder-3b-instruct-q5_k_m.gguf"):
            return "qwen2.5-coder-3b-instruct-q5_k_m.gguf"

    # 4. Deep Synthesis & Comprehensive Analysis -> Phi-4 14B / 3B
    if _has_keyword(SYNTHESIS_PATTERN, lower):
        if prefer_3b:
            if _model_file_valid("qwen2.5-3b-instruct-q4_k_m.gguf"):
                return "qwen2.5-3b-instruct-q4_k_m.gguf"
            if _model_file_valid("llama-3.2-3b-instruct-q4_k_m.gguf"):
                return "llama-3.2-3b-instruct-q4_k_m.gguf"
        if _model_file_valid("phi4-14b.gguf"):
            logger.info("Auto-selected Phi-4 14B for deep synthesis task")
            return "phi4-14b.gguf"
        if _model_file_valid("qwen2.5-3b-instruct-q4_k_m.gguf"):
            return "qwen2.5-3b-instruct-q4_k_m.gguf"
        if _model_file_valid("llama-3.2-3b-instruct-q4_k_m.gguf"):
            return "llama-3.2-3b-instruct-q4_k_m.gguf"

    # 5. General Chat & QA
    if prefer_3b:
        if _model_file_valid("llama-3.2-3b-instruct-q4_k_m.gguf"):
            logger.info("Auto-selected Llama 3.2 3B for general task (3B mode)")
            return "llama-3.2-3b-instruct-q4_k_m.gguf"
        if _model_file_valid("qwen2.5-3b-instruct-q4_k_m.gguf"):
            logger.info("Auto-selected Qwen 2.5 3B for general task (3B mode)")
            return "qwen2.5-3b-instruct-q4_k_m.gguf"

    if _model_file_valid("qwen2.5-7b-instruct-q3_k_m.gguf"):
        logger.info("Auto-selected Qwen 2.5 7B Instruct for general task")
        return "qwen2.5-7b-instruct-q3_k_m.gguf"
    if _model_file_valid("qwen2.5-7b.gguf"):
        logger.info("Auto-selected Qwen 2.5 7B for general task")
        return "qwen2.5-7b.gguf"
    if _model_file_valid("llama-3.2-3b-instruct-q4_k_m.gguf"):
        logger.info("Auto-selected Llama 3.2 3B for general task")
        return "llama-3.2-3b-instruct-q4_k_m.gguf"
    if _model_file_valid("qwen2.5-3b-instruct-q4_k_m.gguf"):
        logger.info("Auto-selected Qwen 2.5 3B for general task")
        return "qwen2.5-3b-instruct-q4_k_m.gguf"
    if _model_file_valid("qwen1_5-4b-chat-q4_k_m.gguf"):
        logger.info("Auto-selected Qwen 1.5 4B for general task")
        return "qwen1_5-4b-chat-q4_k_m.gguf"

    return get_router_model()

