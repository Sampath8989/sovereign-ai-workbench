"""
Model Manager: VRAM allocator & llama.cpp hot-swap engine.
Loads, evicts, and generates from GGUF models with VRAM-aware scheduling.

MockLLM fallback: If model files are missing or llama-cpp-python fails to load,
a deterministic MockLLM is used so the agent graph can still be tested end-to-end.
"""

import hashlib
import json
import os
import re
import subprocess
import threading
import time
import logging
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Union

from backend.config import get_model_path, get_model_roster, get_max_vram_gb, get_tier
from backend.core.audit_log import AuditLogger

logger = logging.getLogger(__name__)


class ModelInferenceError(Exception):
    """
    Typed inference failure (e.g. CUDA out-of-memory on a real GGUF model)
    carrying a stable machine-readable ``code`` and a human-readable
    ``message``. The API layer maps this to a structured JSON error body
    (``{"error": code, "detail": message}``) instead of a bare 500, and the
    frontend displays the message.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _looks_like_cuda_oom(exc: BaseException) -> bool:
    """
    Heuristic: does this exception look like a CUDA/VRAM out-of-memory failure?
    llama.cpp surfaces these as RuntimeError with messages like
    ``CUDA error: out of memory`` / ``ggml_cuda...failed to allocate``.
    """
    try:
        text = " ".join(str(a) for a in exc.args)
        if not text:
            text = str(exc)
        text = text.lower()
    except Exception:
        return False
    markers = (
        "out of memory", "cuda error", "cudamalloc", "cuda malloc",
        "failed to allocate", "allocation failed", "alloc failed",
        "ggml_cuda", "cublas", "insufficient vram", "out-of-memory",
    )
    return any(m in text for m in markers)


def _looks_like_context_overflow(exc: BaseException) -> bool:
    """
    Detect llama.cpp context-window overflow, surfaced as e.g.
    ``ValueError: Requested tokens (31520) exceed context window of 2048``.
    Usually caused by feeding binary/large content into the prompt.
    """
    try:
        text = " ".join(str(a) for a in exc.args)
        if not text:
            text = str(exc)
        text = text.lower()
    except Exception:
        return False
    markers = (
        "exceed context window", "requested tokens", "context window of",
        "n_ctx", "too many tokens", "prompt is too long", "input is too long",
        "context length", "sequence length",
    )
    return any(m in text for m in markers)


def _overflow_error_for(model_name: str, exc: BaseException) -> ModelInferenceError:
    """Structured error for a prompt that exceeds the model context window."""
    return ModelInferenceError(
        "context_overflow",
        f"{model_name}: the request is too long for the model context window "
        f"(2048 tokens). Uploaded images must be analyzed with the image tools "
        "(nameplate/handwriting/P&ID) rather than read as text, and large files "
        "are summarized in truncated form. Ask in smaller steps. "
        f"(underlying error: {str(exc)[:200]})",
    )


# Negation markers shared with backend/core/router.py for consistency.
# Words/phrases that negate a following keyword within a short window.
_NEGATION_MARKERS_RE = re.compile(
    r'\b(do\s+not|don\'t|dont|never|avoid|without|no|skip|refrain\s+from)\b'
)
_NEGATION_WINDOW = 5


def _is_keyword_negated(lower_text: str, keyword: str) -> bool:
    """
    Check if a keyword in lower_text is preceded by a negation marker
    within a ~5-word window and within the same clause.
    Reuses the same heuristic as backend/core/router.py._is_negated.
    """
    idx = lower_text.find(keyword)
    if idx < 0:
        return False
    preceding = lower_text[:idx]
    # Find nearest clause boundary
    clause_start = 0
    for sep in [',', '.', ';', ' - ', ' \u2014 ']:
        pos = preceding.rfind(sep)
        if pos > clause_start:
            clause_start = pos + len(sep)
    clause_text = preceding[clause_start:]
    words = clause_text.split()
    window = words[-_NEGATION_WINDOW:] if len(words) >= _NEGATION_WINDOW else words
    return bool(_NEGATION_MARKERS_RE.search(" ".join(window)))


# Try importing llama_cpp
try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning(
        "\n"
        "=" * 70 + "\n"
        "CRITICAL: llama_cpp not installed! Model loading will use MockLLM.\n"
        "Install with: pip install llama-cpp-python\n"
        "For CUDA GPU support: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python --force-reinstall\n"
        "=" * 70
    )

# Try importing pynvml for live GPU queries
_PYNVML_AVAILABLE = False
try:
    import pynvml
    pynvml.nvmlInit()
    _PYNVML_AVAILABLE = True
    logger.info("pynvml available for live GPU VRAM queries.")
except Exception:
    logger.info("pynvml not available. Will try nvidia-smi fallback.")


def query_free_vram_gb() -> Optional[float]:
    """
    Query currently free GPU VRAM in GB.
    Returns None if no GPU or query fails.
    """
    if _PYNVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gb = mem.free / (1024 ** 3)
            logger.debug(f"pynvml: free VRAM = {free_gb:.2f} GB")
            return free_gb
        except Exception as e:
            logger.warning(f"pynvml query failed: {e}")

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            free_mb = float(result.stdout.strip().split("\n")[0])
            free_gb = free_mb / 1024
            logger.debug(f"nvidia-smi: free VRAM = {free_gb:.2f} GB")
            return free_gb
    except Exception as e:
        logger.warning(f"nvidia-smi query failed: {e}")

    return None


def query_total_vram_gb() -> Optional[float]:
    """Query total GPU VRAM in GB."""
    if _PYNVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return mem.total / (1024 ** 3)
        except Exception:
            pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0]) / 1024
    except Exception:
        pass

    return None


def query_used_vram_gb() -> Optional[float]:
    """Query currently used GPU VRAM in GB."""
    if _PYNVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return mem.used / (1024 ** 3)
        except Exception:
            pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0]) / 1024
    except Exception:
        pass

    return None



MOCK_PREFIX = "[MockLLM] "

# A filename / file path token as it appears in the ORIGINAL (un-lowercased)
# user text. Captures preserve the exact case so lookups stay exact-match
# against the real file on disk (uploads keep names like
# "Screenshot_2026-09-03_15-43-10.png"). Optional directory prefix allowed.
_REFERENCED_PATH_RE = re.compile(
    r"(?:^|[\s,;:(=\[\u201c\u2018\"'])([A-Za-z0-9_./\\-]+\.(?:pdf|docx?|xlsx|pptx|csv|json|md|txt|log|png|jpe?g|bmp|gif|webp|tif{1,2}|heic))(?=$|[\s,;:).\u201d\u2019\"'!?])",
    re.IGNORECASE,
)

# Image extensions routed to the vision tools (photo/handwriting/P&ID).
_VISION_IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".heic",
}


class MockVisionModel:
    """
    Deterministic mock vision model for testing without real VL weights.
    Returns hardcoded OCR text based on the prompt content.

    NOTE: This is a mock/demo stub — NOT a real vision model. The confidence
    values and text it returns are deterministic artifacts for testing, not
    calibrated accuracy. Do not mistake mock outputs for real model performance.
    """

    @staticmethod
    def _image_hash_seed(image_path: str) -> int:
        """
        Derive a deterministic integer seed from the image file's actual bytes.
        Used so that different images produce different (but reproducible)
        mock confidence values, preventing the optics problem of "two different
        images, identical confidence score."

        Falls back to a hash of the path string if the file cannot be read.
        """
        try:
            with open(image_path, "rb") as f:
                data = f.read(4096)  # read first 4 KB for efficiency
            return int(hashlib.md5(data).hexdigest()[:8], 16)
        except (OSError, IOError):
            return int(hashlib.md5(image_path.encode()).hexdigest()[:8], 16)

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """
        Analyze an image with a text prompt. Returns deterministic fake OCR text.

        Args:
            image_path: Path to the image file.
            prompt: The analysis prompt.

        Returns:
            A string with the (mock) analysis result.
        """
        lower = prompt.lower()

        # P&ID topology extraction
        if any(kw in lower for kw in ["topology", "p&id", "pid", "equipment tag"]):
            return json.dumps({
                "nodes": [{"id": "V-101", "type": "valve"}],
                "edges": [{"from": "V-101", "to": "P-101"}]
            })

        # Handwriting transcription
        if "handwriting" in lower or "transcribe" in lower:
            return "Mock handwritten text: Pressure 5bar, Temperature 120C, Flow rate OK"

        # Nameplate / photo analysis
        if any(kw in lower for kw in ["nameplate", "photo", "model", "serial"]):
            return "Model: X-200, Serial: 12345, Manufacturer: Acme Corp, Type: Centrifugal Pump"

        # Generic fallback
        return f"MockVisionModel analysis of {Path(image_path).name}: {prompt[:100]}"

    def get_mock_confidence(self, image_path: str) -> float:
        """
        Return a deterministic but image-responsive mock confidence score.

        The score is derived from a hash of the image file's actual bytes,
        so different images produce different values (solving the "two images,
        identical confidence" demo optics problem). The range is [0.50, 0.95]
        to look plausible without claiming real accuracy.

        This is MOCK/DEMO behavior only — not calibrated model confidence.
        """
        seed = self._image_hash_seed(image_path)
        # Map the 32-bit hash seed to [0.50, 0.95]
        confidence = 0.50 + (seed % 4500) / 10000.0
        return round(confidence, 3)


class _StubModel:
    """Stub model for when llama.cpp is unavailable or model file is missing."""

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name
        self._loaded_at = time.time()

    def create_completion(self, prompt: str, **kwargs) -> dict:
        return {
            "choices": [
                {"text": f"[StubModel:{self.model_name}] Response to: {prompt[:80]}"}
            ]
        }

    def close(self):
        pass


class MockLLM(_StubModel):
    """
    Deterministic mock LLM for testing without model weights.
    Returns hardcoded responses based on the prompt content.
    
    All responses are routed through _wrap_response() to ensure the
    [MockLLM] disclosure prefix is present on every output path.
    """

    def __init__(self, model_name: str = "mock"):
        super().__init__(model_name=model_name)

    def _wrap_response(self, text: str) -> dict:
        """
        Wrap text in the standard llama-cpp-python response format,
        prepending the [MockLLM] disclosure prefix.
        
        Every public method that returns a response MUST call this.
        """
        return {"choices": [{"text": f"{MOCK_PREFIX}{text}"}]}

    def _wrap_plan(self, plan: list) -> dict:
        """
        Return a plan response. Since prepending text to a JSON array
        would break parsing, we wrap the plan in a dict that includes
        a 'mock' indicator field. The planner.py parser handles both
        raw arrays and this wrapped format.
        """
        wrapped = json.dumps({"mock": True, "plan": plan})
        return {"choices": [{"text": wrapped}]}

    @staticmethod
    def _extract_user_message(messages: Union[str, List[dict]]) -> str:
        """
        Extract only the user's message content from a message list.
        This avoids combining system prompts with user input, which causes
        false keyword matches (e.g., system prompt containing 'plan').
        """
        raw = ""
        if isinstance(messages, str):
            raw = messages
        elif isinstance(messages, list):
            # Last user message is the actual user input
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role") == "user":
                    raw = m.get("content", "")
                    break
            # Fallback: last message content
            if not raw and messages:
                m = messages[-1]
                raw = m.get("content", "") if isinstance(m, dict) else str(m)
        else:
            raw = str(messages)

        # Handle multimodal list content (e.g. [{"type": "text", "text": ...}, {"type": "image_url", ...}])
        if isinstance(raw, list):
            extracted_text = []
            for part in raw:
                if isinstance(part, dict) and part.get("type") == "text":
                    extracted_text.append(part.get("text", ""))
                elif isinstance(part, str):
                    extracted_text.append(part)
            raw = " ".join(extracted_text)

        # Strip synthesize/prompt wrappers to isolate the actual user query
        if "User request:" in raw:
            m_req = re.search(r"User request:\s*(.+?)(?:\n\nExecution results:|$)", raw, re.DOTALL)
            if m_req:
                return m_req.group(1).strip()
        elif "Sub-question:" in raw:
            m_req = re.search(r"Sub-question:\s*(.+?)(?:\n\nExecution results:|$)", raw, re.DOTALL)
            if m_req:
                return m_req.group(1).strip()
        return raw.strip()

    @staticmethod
    def _is_greeting(text: str) -> bool:
        """
        Detect simple greetings and conversational openers.
        These should be handled as direct chat, not routed through
        the full plan→execute→retrieve→synthesize pipeline.
        """
        lower = text.lower().strip()
        cleaned = re.sub(r'[!.,?]+$', '', lower).strip()
        # Exact match
        greetings = {
            'hello', 'hi', 'hey', 'howdy', 'greetings', 'sup', 'yo', 'hiya',
            'good morning', 'good afternoon', 'good evening', 'good day',
        }
        if cleaned in greetings:
            return True
        # Short inputs that are just greetings
        if len(lower) <= 25 and re.match(r'^(hi|hey|hello|yo|sup|hiya)(\s+(there|workbench|assistant|all|everyone))?[\s!.,]*$', lower):
            return True
        # Simple conversational openers
        if re.match(r'^(what can you do|who are you|what are you|tell me about yourself|what do you do|help)$', cleaned):
            return True
    @staticmethod
    def _is_doc_generation_intent(prompt: str) -> bool:
        """
        Determine if a prompt is requesting document/report creation (triggering doc_generator)
        vs a conversational question (triggering pure RAG/chat).
        Handles full formatting specs, adjectives, page counts, and design constraints.
        """
        p = prompt.strip().lower()

        # Negative check: pure question patterns that do not request file creation
        is_pure_question = (
            bool(re.match(r'^(what|how|why|when|where|who|which|is\s+there|are\s+there|can\s+you\s+explain|explain|tell\s+me|describe|lookup|search)\b', p))
            and not any(act in p for act in ['create', 'generate', 'write', 'make', 'draft', 'export', 'prepare', 'save to', 'download', 'produce'])
        )
        if is_pure_question:
            return False

        # Exclude read/analyze operations on existing files
        if any(kw in p for kw in ['analyze uploaded file', 'read this', 'read the', 'read file', 'analyze file', 'open file', 'summarize file', 'inspect file']):
            return False

        # Flexible regex matching action verbs and document nouns with up to 100 chars of intervening modifiers/specs
        # (e.g., "Create a professional, beginner-friendly 2-page PDF", "Create a professional **2-page PDF**")
        action_re = r'(?:create|generate|write|make|draft|export|produce|prepare|build|compile)'
        noun_re = r'(?:pdf|docx?|documents?|docs?|reports?|briefs?|memos?|whitepapers?|spreadsheets?|sheets?|excel|xlsx|presentations?|slides?|slide\s+deck|powerpoint|pptx)'
        if re.search(rf'\b{action_re}\b[^\.\n\?!;]{{0,100}}\b{noun_re}\b', p, re.IGNORECASE):
            return True

        # Check for explicit page/slide count specifications followed by document types (e.g. "2-page PDF")
        if re.search(r'\b(?:\d+|one|two|three|four|five)[\s\-]*(?:pages?|slides?)\b[^\.\n\?!;]{0,50}\b(?:pdf|doc|docx|report|presentation|slides?)\b', p, re.IGNORECASE):
            return True

        # Check for explicit file extensions in generation context
        if any(ext in p for ext in ['.pdf', '.docx', '.pptx', '.xlsx']):
            return True

        if any(phrase in p for phrase in [
            'in this document', 'in this report', 'in this memo',
            'word document', 'word doc', 'pdf report', 'pdf document',
            'approval note', 'status report', 'project report', 'quarterly report',
            'safety report', 'summary report', 'technical report', 'compliance report',
        ]):
            return True

        return False

    @staticmethod
    def detect_format(prompt: str) -> Optional[str]:
        """
        Explicit format-keyword detection upstream of tool invocation.
        Returns 'pdf', 'docx', 'pptx', 'xlsx', or None if format cannot be determined.
        """
        p = prompt.lower()
        if re.search(r'\b(pdf|\.pdf)\b', p):
            return "pdf"
        if re.search(r'\b(docx?|\.docx|word(?:\s+doc(?:ument)?)?)\b', p):
            return "docx"
        if re.search(r'\b(pptx?|\.pptx|powerpoint|slides?|presentation|slide\s+deck)\b', p):
            return "pptx"
        if re.search(r'\b(xlsx?|\.xlsx|excel|spreadsheets?)\b', p):
            return "xlsx"
        return None

    @staticmethod
    def _extract_doc_title(prompt: str) -> str:
        """Extract a readable title from document generation prompt."""
        p = prompt.strip()

        # Explicit "titled 'X'" / "titled “X”" / "with title 'X'" / "title: X"
        m = re.search(
            r'(?:with\s+title|titled|title\s*(?::|is|=))\s*[\'"\u201c\u2018]?([^\'"\u201d\u2019\n]{1,80}?)[\'"\u201d\u2019]?(?=\.|\n|,|\s+and|\s+with|\s+for|$)',
            p,
            re.IGNORECASE,
        )
        if m and m.group(1).strip():
            return m.group(1).strip()[:80]

        m = re.search(r'(?:summarizing|about|for|on|regarding|discussing)\s+(.+?)(?:\.|\n|$)', p, re.IGNORECASE)
        if m:
            raw_title = m.group(1).strip()
            if len(raw_title) > 60:
                raw_title = raw_title[:57] + "..."
            return raw_title[:1].upper() + raw_title[1:]
        if len(p) <= 40:
            return p
        return "Engineering Document Summary"

    @staticmethod
    def _extract_doc_filename(prompt: str) -> str:
        """
        Extract the user-specified output filename from a doc request, e.g.
        "Create a word document named test.docx ..." -> "test.docx".
        Supports .pdf, .docx, .pptx, .xlsx.
        """
        p = prompt.strip()
        m = re.search(
            r'(?:named?|called|filename(?:\s+of)?|file\s+name)\s+(?:as\s+)?[\'\"\u201c\u2018]?([A-Za-z0-9_\-./]+\.(?:docx?|pdf|pptx?|xlsx?))',
            p,
            re.IGNORECASE,
        )
        if m:
            return m.group(1)
        m = re.search(r'save\s+(?:as|to)\s+[\'\"\u201c\u2018]?([A-Za-z0-9_\-./]+\.(?:docx?|pdf|pptx?|xlsx?))', p, re.IGNORECASE)
        if m:
            return m.group(1)
        m = re.search(r'\b([A-Za-z0-9_\-./]+\.(?:docx?|pdf|pptx?|xlsx?))\b', p, re.IGNORECASE)
        if m:
            return m.group(1)
        return ""

    @staticmethod
    def _extract_referenced_path(prompt: str, allowed_exts: Optional[set] = None) -> str:
        """
        Extract the first file/path token referenced in a user prompt,
        PRESERVING the original letter case. Matches against the un-lowercased
        text so names like ``Screenshot_2026-09-03_15-43-10.png`` survive the
        plan untouched and resolve exactly against the sandbox directory.

        Args:
            prompt: Original-case user text.
            allowed_exts: Optional set of lowercased extensions to filter on
                (e.g. ``{".png", ".jpg"}`` for vision routing). None = any.

        Returns:
            The exact-case token (directory prefix included if present), or "".
        """
        if not prompt or not isinstance(prompt, str):
            return ""
        for m in _REFERENCED_PATH_RE.finditer(prompt):
            token = m.group(1).strip("\"'\u201c\u201d\u2018\u2019")
            ext = os.path.splitext(token)[1].lower()
            if allowed_exts is None or ext in allowed_exts:
                return token
        return ""

    @staticmethod
    def _qualify_sandbox_path(token: str, default: str = "") -> str:
        """
        Turn a bare filename into a workspace sandbox path. If the token already
        carries a directory prefix (or is absolute), it is used as-is; a bare
        filename is resolved against ``workspace/sandbox_files/``.
        """
        token = (token or "").strip()
        if not token:
            return default
        if token.startswith("workspace/") or token.startswith("workspace\\") \
                or token.startswith("sandbox_files/") or "/" in token or "\\" in token:
            return token
        return f"workspace/sandbox_files/{token}"

    @staticmethod
    def _extract_doc_content(prompt: str) -> str:
        """
        Extract explicit body content from a doc request when provided.
        """
        p = prompt.strip()
        m = re.search(
            r'(?:with\s+content|and\s+content|content\s*(?::|is|=|\s+of))\s*[\'"\u201c\u2018](.+?)[\'"\u201d\u2019](?=\.|,|$)',
            p,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
        m = re.search(r'(?:with|and)\s+content\s+([^\n]{1,200}?)(?:\.|$)', p, re.IGNORECASE)
        if m and m.group(1).strip():
            return m.group(1).strip()
        return ""

    @staticmethod
    def _compose_doc_body(prompt: str) -> str:
        """
        Compose the document body from the user's request.
        For Python Basics / Introduction requests, synthesizes a complete 2-page academic curriculum guide.
        """
        explicit = MockLLM._extract_doc_content(prompt)
        if explicit:
            return explicit

        p_lower = prompt.lower()
        if "python" in p_lower and ("basic" in p_lower or "introduction" in p_lower or "syntax" in p_lower):
            return (
                "# Page 1: Introduction to Python\n\n"
                "### What is Python?\n"
                "Python is a versatile, high-level, general-purpose programming language developed by Guido van Rossum. "
                "It emphasizes code readability with simple English-like syntax and minimal boilerplate.\n\n"
                "### Key Features of Python\n"
                "- **Simple and Readable**: Easy for beginners to understand and write quickly.\n"
                "- **Interpreted**: Executes instructions directly without previous compilation.\n"
                "- **Dynamically Typed**: Variable types are bound to values at runtime rather than declarations.\n"
                "- **Cross-Platform**: Seamless execution across Linux, Windows, macOS, and embedded hardware.\n"
                "- **Extensive Standard Library**: Rich batteries-included modules for web, data analysis, and OS scripting.\n\n"
                "### Why Learn Python?\n"
                "- **AI and Machine Learning**: Dominant language powering PyTorch, TensorFlow, and Hugging Face.\n"
                "- **Data Science and Analytics**: Backed by high-performance libraries like NumPy, Pandas, and SciPy.\n"
                "- **Automation and Scripting**: Rapid development of system maintenance utilities and pipeline tooling.\n"
                "- **Web and Software Development**: Robust enterprise backends using FastAPI, Django, and Flask.\n\n"
                "### Basic Python Syntax and Indentation\n"
                "Python uses indentation (4 spaces) rather than curly braces to define statement blocks:\n\n"
                "```python\n"
                "# First Python program\n"
                "print(\"Hello, World!\")\n"
                "```\n\n"
                "### Variables and Fundamental Data Types\n"
                "Variables are created when assigned a value:\n\n"
                "```python\n"
                "student_id = 1042           # int\n"
                "gpa = 3.85                  # float\n"
                "course_title = \"Intro Python\" # str\n"
                "is_passed = True            # bool\n"
                "```\n\n"
                "### Input / Output and Type Conversion\n"
                "```python\n"
                "name = input(\"Enter student name: \")\n"
                "year_str = input(\"Enter birth year: \")\n"
                "birth_year = int(year_str)  # Convert string to int\n"
                "age = 2026 - birth_year\n"
                "print(f\"Student {name} is {age} years old.\")\n"
                "```\n\n"
                "---\n\n"
                "# Page 2: Core Python Basics\n\n"
                "### Operators\n"
                "- **Arithmetic**: `+`, `-`, `*`, `/` (float div), `//` (floor div), `%` (modulo), `**` (power)\n"
                "- **Comparison**: `==`, `!=`, `>`, `<`, `>=`, `<=`\n"
                "- **Logical**: `and`, `or`, `not`\n\n"
                "### Conditional Statements\n"
                "```python\n"
                "score = 88\n"
                "if score >= 90:\n"
                "    grade = \"A\"\n"
                "elif score >= 75:\n"
                "    grade = \"B\"\n"
                "else:\n"
                "    grade = \"C\"\n"
                "print(f\"Calculated Grade: {grade}\")\n"
                "```\n\n"
                "### Loops: For and While\n"
                "```python\n"
                "# For loop\n"
                "for count in range(1, 4):\n"
                "    print(f\"Step {count}\")\n\n"
                "# While loop\n"
                "remaining = 3\n"
                "while remaining > 0:\n"
                "    print(f\"Countdown: {remaining}\")\n"
                "    remaining -= 1\n"
                "```\n\n"
                "### Basic Collections\n"
                "- **List**: Ordered, mutable sequence: `items = [\"valves\", \"pumps\", \"gauges\"]`\n"
                "- **Tuple**: Ordered, immutable sequence: `coords = (12.5, 45.8)`\n"
                "- **Set**: Unordered unique collection: `unique_tags = {101, 102, 103}`\n"
                "- **Dictionary**: Key-value pairs: `sensor = {\"id\": \"P-101\", \"pressure\": 4.5}`\n\n"
                "### Functions\n"
                "```python\n"
                "def calculate_flow(diameter: float, velocity: float) -> float:\n"
                "    \"\"\"Calculate volumetric flow rate.\"\"\"\n"
                "    import math\n"
                "    area = math.pi * (diameter / 2) ** 2\n"
                "    return area * velocity\n\n"
                "flow = calculate_flow(0.1, 2.5)\n"
                "print(f\"Flow rate: {flow:.4f} m3/s\")\n"
                "```\n\n"
                "### Python Basics Cheat Sheet\n"
                "- `print(obj)`: Print object to console\n"
                "- `input(prompt)`: Read input string\n"
                "- `len(sequence)`: Get number of elements\n"
                "- `type(variable)`: Return object type\n"
                "- `range(start, stop[, step])`: Generate arithmetic progression\n"
                "- `def func(args): return val`: Reusable logic block\n"
            )

        title = MockLLM._extract_doc_title(prompt)
        body = re.sub(
            r'(?:^|\s)(?:create|generate|write|make|draft|export|produce|prepare|build)\s+(?:a\s+|an\s+|the\s+)?(?:word\s+|pdf\s+)?(?:document|doc|report|file|brief|memo)\s*',
            ' ',
            prompt,
            flags=re.IGNORECASE,
        )
        body = re.sub(r'\s+named?\s+[A-Za-z0-9_\-./]+\.(?:docx?|pdf|pptx?|xlsx?)', '', body, flags=re.IGNORECASE)
        body = re.sub(r'\s+with\s+title\s+[\'"\u201c\u2018][^\'"\u201d\u2019]*[\'"\u201d\u2019]', '', body, flags=re.IGNORECASE)
        body = re.sub(r'\s+(?:with|and)\s+content\s+[\'"\u201c\u2018][^\'"\u201d\u2019]*[\'"\u201d\u2019]', '', body, flags=re.IGNORECASE)
        body = re.sub(r'[.\s]+$', '', body.strip())
        if not body:
            body = f"Requested document: {title}"
        return f"{title}\n\n{body.capitalize()}"

    def summarize(self, text: str) -> str:
        """
        Deterministic summarization used by the llm/summarize tool step.

        Never returns a tool plan (unlike create_chat_completion, which is
        intent-classified for PLANNING). The summarize tool must always return
        plain text so a nested tool plan can never leak into the pipeline.
        """
        if not text or not text.strip():
            return "No content provided to summarize."
        if text.strip().lower().startswith("error"):
            return text.strip()
        snippet = re.sub(r'\s+', ' ', text).strip()
        return f"This is a mock summary of the provided content: {snippet[:400]}"

    def create_chat_completion(self, messages: Union[str, List[dict]], **kwargs) -> dict:
        """
        Mock chat completion. Inspects the messages to return appropriate responses.

        Args:
            messages: Either a string prompt or a list of message dicts.

        Returns:
            A dict with 'choices' key mimicking llama-cpp-python output.
        """
        # Extract the full combined text for keyword matching
        has_multimodal_image = False
        if isinstance(messages, str):
            text = messages
        elif isinstance(messages, list):
            text_parts = []
            for m in messages:
                if isinstance(m, dict):
                    c = m.get("content", "")
                    if isinstance(c, list):
                        for sub in c:
                            if isinstance(sub, dict):
                                if sub.get("type") == "text":
                                    text_parts.append(sub.get("text", ""))
                                elif sub.get("type") == "image_url":
                                    has_multimodal_image = True
                    elif isinstance(c, str):
                        text_parts.append(c)
                    else:
                        text_parts.append(str(c))
                else:
                    text_parts.append(str(m))
            text = " ".join(text_parts)
        else:
            text = str(messages)

        lower = text.lower()

        # --- Intent classifier: greetings and simple chat ---
        # Extract only the user's message for intent detection.
        # This avoids false matches from system prompts (e.g., planner prompt
        # containing the word 'plan' which would trigger the planner branch
        # for every input, including simple greetings).
        user_msg = self._extract_user_message(messages)
        if self._is_greeting(user_msg):
            return self._wrap_response(
                "Hello! I'm the Sovereign AI Workbench — an air-gapped, locally-hosted AI assistant. "
                "I can help you with: generating documents (Word, PowerPoint, Excel), "
                "analyzing P&ID diagrams, reading handwritten notes, calculating values, "
                "and more — all running entirely on your local hardware with no external network access. "
                "What would you like me to help you with?"
            )

        # --- Verification requests are text responses, never tool plans ---
        # The verifier must ALWAYS get a VERDICT, never a nested tool plan, so
        # the check runs before any plan-generating branch below.
        if 'verdict' in lower or 'citation verif' in lower or 'grounded' in lower:
            has_source_evidence = self._check_grounding(text)
            if has_source_evidence:
                return self._wrap_response(
                    "VERDICT: YES\nREASON: All claims in the text are present in the provided sources."
                )
            return self._wrap_response(
                "VERDICT: NO\nREASON: Some claims in the text are not found in the provided sources."
            )

        # --- Bare-filename rejection (Bug 4) ---
        # A filename pasted into the chat box is NOT an upload or a read
        # request, and it must never be treated as a task (which would send it
        # through photo/pid/file analysis and fail with "not found"). Reject it
        # deterministically with upload guidance. Runs before EVERY
        # plan-generating branch.
        _user_lower = user_msg.lower()
        _bare_filename = re.fullmatch(
            r'\s*(?:[a-z0-9_\-\./]+\.(?:pdf|txt|docx|xlsx|csv|json|md|py|log|png|jpg|jpeg))\s*',
            _user_lower,
        )
        if _bare_filename:
            return self._wrap_response(
                "I can't open a file from a plain-text filename. Use the paperclip "
                "(attach) button next to the input box to upload the file to the "
                "sandbox, then ask me to analyze it."
            )

        # Multimodal vision completions always return vision analysis
        if has_multimodal_image:
            return self._wrap_response(
                "[MockLLM Vision] Image analyzed successfully: The visual inspection confirms "
                "the presence of the expected industrial components, equipment markings, and diagram structures."
            )

        # Check if called in a synthesis context vs planning context

        # --- Deliverable synthesis tool triggers ---
        # Document / Report / PDF / Word generation
        if self._is_doc_generation_intent(user_msg):
            fmt = self.detect_format(user_msg)
            if fmt is None:
                # Upstream format-keyword detection: if format cannot be determined,
                # default to asking rather than silently picking docx.
                return self._wrap_response(
                    "Which format would you like me to generate? Please specify whether you would prefer "
                    "a PDF (.pdf), Word document (.docx), PowerPoint presentation (.pptx), or Excel spreadsheet (.xlsx)."
                )

            body_content = self._compose_doc_body(user_msg)
            if is_synthesis:
                return self._wrap_response(body_content)

            uid = uuid.uuid4().hex[:8]
            title = self._extract_doc_title(user_msg)
            user_fname = self._extract_doc_filename(user_msg)

            if fmt == "pdf":
                if user_fname and user_fname.lower().endswith(".pdf"):
                    fname = user_fname
                elif user_fname:
                    fname = f"{Path(user_fname).stem}.pdf"
                else:
                    slug = re.sub(r'[^a-zA-Z0-9_-]+', '_', title.lower()).strip('_')[:30]
                    fname = f"{slug or 'report'}_{uid}.pdf"

                plan = [
                    {"tool": "doc_generator", "action": "generate",
                     "args": [fname, title, body_content, "pdf"]}
                ]
                return self._wrap_plan(plan)

            elif fmt == "docx":
                if user_fname and user_fname.lower().endswith(".docx"):
                    fname = user_fname
                elif user_fname:
                    fname = f"{Path(user_fname).stem}.docx"
                else:
                    slug = re.sub(r'[^a-zA-Z0-9_-]+', '_', title.lower()).strip('_')[:30]
                    fname = f"{slug or 'report'}_{uid}.docx"

                plan = [
                    {"tool": "doc_generator", "action": "generate",
                     "args": [fname, title, body_content, "docx"]}
                ]
                return self._wrap_plan(plan)

            elif fmt == "pptx":
                if user_fname and user_fname.lower().endswith(".pptx"):
                    fname = user_fname
                elif user_fname:
                    fname = f"{Path(user_fname).stem}.pptx"
                else:
                    fname = f"slides_{uid}.pptx"

                bullets = [line.strip("- *• ") for line in body_content.splitlines() if line.strip().startswith(("-", "*", "•"))][:6]
                if not bullets:
                    bullets = ["Overview", "Key Details", "Summary & Next Steps"]

                plan = [
                    {"tool": "ppt_generator", "action": "generate",
                     "args": [fname, title, bullets]}
                ]
                return self._wrap_plan(plan)

            elif fmt == "xlsx":
                if user_fname and user_fname.lower().endswith(".xlsx"):
                    fname = user_fname
                elif user_fname:
                    fname = f"{Path(user_fname).stem}.xlsx"
                else:
                    fname = f"data_{uid}.xlsx"

                plan = [
                    {"tool": "spreadsheet_generator", "action": "generate",
                     "args": [fname, [["Title", title], ["Summary", body_content[:100]]]]}
                ]
                return self._wrap_plan(plan)

        # PowerPoint / slides generation
        if any(kw in lower for kw in ["powerpoint", "slides", "pptx"]):
            uid = uuid.uuid4().hex[:8]
            plan = [
                {"tool": "ppt_generator", "action": "generate",
                 "args": [f"slides_{uid}.pptx", "Presentation", ["Slide 1", "Slide 2"]]}
            ]
            return self._wrap_plan(plan)

        # Spreadsheet generation
        if any(kw in lower for kw in ["spreadsheet", "xlsx", "excel"]):
            # Try to extract data from the prompt
            data_match = re.search(r"data\s*(?:=|:\s*)\s*(\[\[.*?\]\])", text, re.DOTALL)
            data = [["Name", "Value"], ["Item", "1"]]
            if data_match:
                try:
                    data = __import__("json").loads(data_match.group(1))
                except Exception:
                    pass

            # Try to extract filename, fall back to unique name.
            # Capture from the original-case text so "Data_Q3.xlsx" stays
            # exact-case (Bug 1).
            fname_match = re.search(r"named?\s+(\S+\.xlsx)", user_msg)
            if fname_match:
                fname = fname_match.group(1)
            else:
                uid = uuid.uuid4().hex[:8]
                fname = f"data_{uid}.xlsx"

            plan = [
                {"tool": "spreadsheet_generator", "action": "generate",
                 "args": [fname, data]}
            ]
            return self._wrap_plan(plan)

        # P&ID / topology extraction (negation-aware). If the user references an
        # actual uploaded drawing by name, analyze THAT file (exact case
        # preserved); only fall back to the demo image when no file is named.
        pid_keywords = ["p&pid", "topology", "pid extractor", "extract topology"]
        if any(kw in lower and not _is_keyword_negated(lower, kw) for kw in pid_keywords):
            pid_file = MockLLM._qualify_sandbox_path(
                MockLLM._extract_referenced_path(user_msg, _VISION_IMAGE_EXTS),
                "workspace/sandbox_files/test_pid.png",
            )
            plan = [
                {"tool": "pid_extractor", "action": "extract", "args": [pid_file]}
            ]
            return self._wrap_plan(plan)

        # Handwriting triage (negation-aware)
        hw_keywords = ["handwriting", "handwritten", "read note", "field note"]
        if any(kw in lower and not _is_keyword_negated(lower, kw) for kw in hw_keywords):
            note_file = MockLLM._qualify_sandbox_path(
                MockLLM._extract_referenced_path(user_msg, _VISION_IMAGE_EXTS),
                "workspace/sandbox_files/test_note.jpg",
            )
            plan = [
                {"tool": "handwriting_triage", "action": "read", "args": [note_file]}
            ]
            return self._wrap_plan(plan)

        # Photo / nameplate analysis (negation-aware)
        photo_keywords = ["photo", "nameplate", "field photo", "equipment photo"]
        if any(kw in lower and not _is_keyword_negated(lower, kw) for kw in photo_keywords):
            photo_file = MockLLM._qualify_sandbox_path(
                MockLLM._extract_referenced_path(user_msg, _VISION_IMAGE_EXTS),
                "workspace/sandbox_files/test_photo.jpg",
            )
            plan = [
                {"tool": "photo_analyzer", "action": "analyze", "args": [photo_file]}
            ]
            return self._wrap_plan(plan)

        # Calculator / math
        if any(kw in lower for kw in ["calculate", "solve", "math", "equation"]):
            # Extract the expression from the prompt
            expr = "x + 5 = 10"  # default
            eq_match = re.search(r'(?:solve|calculate|compute|math)[:\s]+(.+)', lower)
            if eq_match:
                expr = eq_match.group(1).strip()
            else:
                # Try to find an equation-like pattern in the full text
                eq_match = re.search(r'([a-z0-9\s\+\-\*/\^\=\.]+(?:=\s*[a-z0-9\s\+\-\*/\^\.]+)?)', text)
                if eq_match:
                    expr = eq_match.group(1).strip()

            plan = [
                {"tool": "calculator", "action": "solve",
                 "args": [expr]}
            ]
            return self._wrap_plan(plan)

        # File I/O triggers. Keyword detection is case-insensitive, but the
        # captured filename MUST come from the original-case user message:
        # matching the capture regex against user_msg.lower() lowercases
        # uploaded filenames ("Screenshot_2026-09-03_15-43-10.png" ->
        # "screenshot_...") which then fail lookup on a case-sensitive
        # filesystem. (Bug 1)
        user_lower = user_msg.lower()
        file_match = re.search(
            r'(?:read|write|open|load|analyze|inspect|summarize|explain)\s+(?:the\s+)?(?:uploaded\s+)?(?:file\s*:?\s*)?([a-zA-Z0-9_\-\./]+\.[a-zA-Z0-9]+)',
            user_msg,
            re.IGNORECASE,
        )
        if not file_match:
            file_match = re.search(r'\b([a-zA-Z0-9_\-\./]+\.(?:pdf|txt|docx|xlsx|csv|json|md|py|log))\b', user_msg, re.IGNORECASE)

        if (file_match or
                any(kw in user_lower for kw in ["read test.txt", "read the file", "file_io", "read file", "uploaded file", "this pdf", "this document"]) or
                re.search(r'\bplan\b', user_lower) or re.search(r'\bsteps\b', user_lower) or re.search(r'\bdecompos', user_lower)):
            fname = file_match.group(1) if file_match else "test.txt"
            if Path(fname).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif"}:
                plan = [
                    {"tool": "llm", "action": "summarize", "args": []}
                ]
            else:
                plan = [
                    {"tool": "file_io", "action": "read", "args": [fname]},
                    {"tool": "llm", "action": "summarize", "args": []}
                ]
            return self._wrap_plan(plan)

        # Code execution triggers
        if any(kw in user_lower for kw in ["execute code", "run code", "python code", "execute script"]):
            plan = [
                {"tool": "code", "action": "execute", "args": ["print('Hello')"]}
            ]
            return self._wrap_plan(plan)

        # Check if called in a planning context (e.g. from planner.py or system prompt asking for steps)
        system_content = ""
        if isinstance(messages, list):
            for m in messages:
                if isinstance(m, dict) and m.get("role") == "system":
                    system_content += " " + m.get("content", "")
        is_plan_request = any(
            kw in system_content.lower()
            for kw in ["task planner", "json array of steps", "decompose into steps"]
        )

        if is_plan_request:
            # If user explicitly asked to summarize or input is empty, produce an llm summarize plan
            if not user_msg.strip() or "summar" in user_lower:
                plan = [
                    {"tool": "llm", "action": "summarize", "args": [user_msg]}
                ]
                return self._wrap_plan(plan)
            # For general Q&A, math, reasoning, or queries not requiring prior tool execution,
            # return an empty plan [] so execute_node doesn't waste CPU time running a
            # redundant 7B summarization of the user prompt before synthesis.
            return self._wrap_plan([])

        # If source content is present in the prompt, prioritize returning
        # grounded source text (content only — never the internal section marker).
        source_section = self._extract_sources(text)
        if source_section:
            return self._wrap_response(source_section)

        # If asked to summarize, return a mock summary
        if "summar" in user_lower:
            return self._wrap_response("This is a mock summary of the provided content.")

        # If user message indicates document generation intent, compose comprehensive doc body
        if self._is_doc_generation_intent(user_msg):
            return self._wrap_response(self._compose_doc_body(user_msg))

        # Final fallback: respond to the USER message only. Never echo the
        # combined prompt text — it contains internal scratchpad state
        # ("Step step_0_result: ...", section markers) that must not leak.
        if user_msg.strip():
            return self._wrap_response(
                f"I received your request: \"{user_msg.strip()[:200]}\". "
                "I can help with document generation, knowledge-base questions, "
                "math, code, and more — all running locally on this air-gapped workbench."
            )
        return self._wrap_response(
            "Hello! I'm the Sovereign AI Workbench. Send me a task or question and I'll help."
        )

    @staticmethod
    def _check_grounding(text: str) -> bool:
        """Heuristic for MockLLM: extract the generated-text section and source-text
        section from the verifier prompt, and check whether:
        1. Specific numbers, quantities, or metrics in generated text actually appear in sources.
        2. Key content tokens in generated text are grounded in sources."""
        gen_start = text.lower().find('generated text:')
        src_start = text.lower().find('sources:')
        if gen_start < 0 or src_start < 0:
            return False
        generated = text[gen_start:src_start].lower()
        sources = text[src_start:].lower()

        # Check numerical / metric contradictions:
        # Extract all numbers/metrics from generated text
        gen_numbers = set(re.findall(r'\b\d+(?:\.\d+)?(?:mm|cm|m|km|psi|bar|kg|g|%|c|f)?\b', generated))
        src_numbers = set(re.findall(r'\b\d+(?:\.\d+)?(?:mm|cm|m|km|psi|bar|kg|g|%|c|f)?\b', sources))

        # If generated text contains specific numbers/metrics not in sources -> NOT grounded
        if gen_numbers and not gen_numbers.issubset(src_numbers):
            return False

        # Extract meaningful tokens from sources and generated text (skip common words)
        stop = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'or', 'from',
            'and', 'of', 'for', 'to', 'if', 'be', 'been', 'not', 'this', 'that',
            'these', 'those', 'it', 'its', 'as', 'by', 'at', 'on', 'with',
            'generated', 'text', 'sources', 'user', 'request', 'what', 'how'
        }
        gen_tokens = {w for w in re.split(r'\W+', generated) if len(w) > 2 and w not in stop}
        src_tokens = {w for w in re.split(r'\W+', sources) if len(w) > 2 and w not in stop}

        if not gen_tokens or not src_tokens:
            return False

        # Check what percentage of generated content tokens exist in sources
        overlap = len(gen_tokens & src_tokens)
        grounded_ratio = overlap / len(gen_tokens)

        return grounded_ratio >= 0.7  # at least 70% of generated content tokens must be in sources

    @staticmethod
    def _extract_sources(text: str) -> str:
        """
        If the prompt contains 'Retrieved sources:' or 'Sources:', return the
        source CONTENT (marker prefix stripped) so the mock response echoes
        back the retrieved context without leaking the internal marker.
        """
        for marker in ["Retrieved sources:", "Sources:", "sources:"]:
            idx = text.find(marker)
            if idx >= 0:
                return text[idx + len(marker):].strip()
        return ""

    def create_completion(self, prompt: str, **kwargs) -> dict:
        """Alias for create_chat_completion with a plain string."""
        return self.create_chat_completion(prompt, **kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate from a plain string prompt (used by verifier)."""
        output = self.create_chat_completion(prompt, **kwargs)
        return output["choices"][0]["text"]

    def close(self):
        pass


def _model_file_size_gb(model_name: str) -> Optional[float]:
    """On-disk size of the GGUF model file in GB, if resolvable."""
    try:
        return os.path.getsize(get_model_path(model_name)) / 1e9
    except (OSError, TypeError):
        return None


def _oom_error_for(model_name: str, exc: BaseException) -> ModelInferenceError:
    """
    Build a structured ModelInferenceError for a CUDA/VRAM out-of-memory
    failure, naming the model and how much VRAM is actually free so the user
    sees a real message instead of a bare 500.
    """
    code = "vision_model_oom" if "llava" in model_name.lower() else "model_oom"
    file_gb = _model_file_size_gb(model_name)
    free_gb = query_free_vram_gb()
    size_hint = f"~{file_gb:.2f} GB on disk" if file_gb else "its model file"
    free_hint = f"only ~{free_gb:.2f} GB VRAM free" if free_gb else "insufficient VRAM free"
    return ModelInferenceError(
        code,
        f"{model_name} could not run: CUDA out of memory while generating. "
        f"Model needs {size_hint}; {free_hint}. "
        "Unload other models (or use a smaller/quantized variant) and retry. "
        f"(underlying error: {exc})",
    )


class ModelManager:
    """
    Manages loading and eviction of GGUF models with VRAM budgeting.
    Uses live GPU queries + static tier ceiling to determine effective budget.
    Falls back to MockLLM if model files are missing.
    """

    def __init__(self, hardware_tier: str = None, max_vram_gb: float = None):
        self.hardware_tier = hardware_tier or get_tier()
        self.static_max_vram_gb = max_vram_gb if max_vram_gb is not None else get_max_vram_gb()
        self.model_roster = get_model_roster()
        self.resident_models: OrderedDict = OrderedDict()
        self.vram_usage: Dict[str, float] = {}
        self.audit = AuditLogger()
        self._total_vram_used: float = 0.0
        self._gpu_query_failures: int = 0
        self._mock_llm = MockLLM()
        self._infer_lock = threading.Lock()
        self.pinned_model: Optional[str] = None

        # Compute effective VRAM budget (live free vs static ceiling)
        self.max_vram_gb = self._compute_effective_budget()
        logger.info(
            f"ModelManager initialized: tier={self.hardware_tier}, "
            f"static_ceiling={self.static_max_vram_gb} GB, "
            f"effective_budget={self.max_vram_gb} GB"
        )

    def _compute_effective_budget(self) -> float:
        free_vram = query_free_vram_gb()

        if free_vram is not None:
            effective = min(self.static_max_vram_gb, free_vram)
            logger.info(
                f"Live GPU VRAM: {free_vram:.2f} GB free. "
                f"Effective budget: {effective:.2f} GB "
                f"(min of tier={self.static_max_vram_gb}, free={free_vram:.2f})"
            )
            self._gpu_query_failures = 0
            return effective
        else:
            self._gpu_query_failures += 1
            logger.warning(
                f"GPU VRAM query failed (attempt {self._gpu_query_failures}). "
                f"Falling back to static tier budget: {self.static_max_vram_gb} GB. "
                f"WARNING: This may OVER-allocate if other processes are using VRAM."
            )
            return self.static_max_vram_gb

    def refresh_vram_budget(self) -> float:
        self.max_vram_gb = self._compute_effective_budget()
        return self.max_vram_gb

    def _estimate_model_vram(self, model_name: str) -> float:
        return self.model_roster.get(model_name, 1.0)

    def _evict_lru(self) -> Optional[str]:
        if not self.resident_models:
            return None

        # Find first non-pinned model from resident_models (LRU order)
        candidate = None
        for name in self.resident_models:
            if name != self.pinned_model:
                candidate = name
                break

        if candidate is None:
            logger.warning("Cannot evict: only pinned model is in resident memory.")
            return None

        model_handle = self.resident_models.pop(candidate)
        vram_freed = self.vram_usage.pop(candidate, 0.0)
        self._total_vram_used -= vram_freed

        if hasattr(model_handle, "close"):
            try:
                model_handle.close()
            except Exception as e:
                logger.warning(f"Error closing model {candidate}: {e}")

        self.audit.log_event(
            "MODEL_EVICTION",
            {
                "model_name": candidate,
                "vram_freed_gb": vram_freed,
                "remaining_vram_gb": self._total_vram_used,
                "tier": self.hardware_tier,
            },
        )
        logger.info(f"Evicted model: {candidate} (freed {vram_freed} GB)")
        return candidate

    def load_model(self, model_name: str, reject_oversized: bool = True):
        """
        Load a model by name. Uses LRU eviction if VRAM is insufficient.
        Falls back to MockLLM if model file does not exist on disk.
        """
        # Return if already loaded
        if model_name in self.resident_models:
            self.resident_models.move_to_end(model_name)
            return self.resident_models[model_name]

        estimated_vram = self._estimate_model_vram(model_name)
        self.refresh_vram_budget()

        # Check if model exceeds total budget
        if estimated_vram > self.max_vram_gb:
            msg = (
                f"Model {model_name} requires {estimated_vram} GB but total "
                f"VRAM budget is only {self.max_vram_gb} GB. "
                f"Static tier ceiling: {self.static_max_vram_gb} GB."
            )
            if reject_oversized:
                self.audit.log_event(
                    "MODEL_LOAD_REJECTED",
                    {
                        "model_name": model_name,
                        "vram_required_gb": estimated_vram,
                        "vram_budget_gb": self.max_vram_gb,
                        "reason": "exceeds_total_budget",
                    },
                )
                raise ValueError(msg)
            else:
                logger.warning(msg + " Loading anyway (reject_oversized=False).")

        # Evict until we have enough VRAM
        while (self._total_vram_used + estimated_vram) > self.max_vram_gb:
            if not self.resident_models:
                if reject_oversized:
                    msg = (
                        f"Cannot free enough VRAM for {model_name} "
                        f"({estimated_vram} GB needed, {self._total_vram_used} GB used, "
                        f"{self.max_vram_gb} GB budget). No more models to evict."
                    )
                    self.audit.log_event(
                        "MODEL_LOAD_REJECTED",
                        {
                            "model_name": model_name,
                            "vram_required_gb": estimated_vram,
                            "vram_used_gb": self._total_vram_used,
                            "vram_budget_gb": self.max_vram_gb,
                            "reason": "insufficient_vram_no_eviction_candidates",
                        },
                    )
                    raise ValueError(msg)
                else:
                    logger.warning(
                        f"Model {model_name} ({estimated_vram} GB) exceeds "
                        f"VRAM budget ({self.max_vram_gb} GB). Loading anyway."
                    )
                    break
            evicted = self._evict_lru()
            if evicted is None:
                break

        model_path = get_model_path(model_name)

        # Try to load real model, fall back to MockLLM if unavailable or USE_MOCK_LLM is set
        use_mock = os.getenv("USE_MOCK_LLM", "").lower() in ("1", "true", "yes")
        if not use_mock and LLAMA_CPP_AVAILABLE and os.path.exists(model_path):
            # Determine GPU support via llama_supports_gpu_offload() or CUDA backend
            _cuda_available = False
            try:
                if hasattr(llama_cpp, "llama_supports_gpu_offload"):
                    _cuda_available = llama_cpp.llama_supports_gpu_offload()
                if not _cuda_available:
                    from llama_cpp import llama_cpp as _lc
                    _cuda_available = hasattr(_lc, "ggml_backend_cuda_init")
            except Exception:
                pass

            # Calculate safe GPU layers for 4GB VRAM. On CUDA builds the
            # planned offload is scaled down to what the CURRENTLY free VRAM
            # can actually hold (weights + compute buffers). This prevents
            # loading e.g. the 3.9 GB Q4 llava-7b with 26 GPU layers when only
            # ~3.7 GB is free, which would otherwise die with a CUDA OOM 500.
            if _cuda_available:
                clean = model_name.lower()
                if "14b" in clean:
                    planned = 12  # Hybrid GPU/CPU offload for 14B models on 4GB VRAM
                elif "7b" in clean:
                    planned = 26  # Substantial GPU offload for 7B models on 4GB VRAM
                else:
                    planned = -1  # Full GPU offload for <=4B models
                n_gpu = self._fit_gpu_layers(model_path, planned, query_free_vram_gb())
                backend = f"CUDA GPU ({n_gpu} layers)"
            else:
                n_gpu = 0
                backend = "CPU only"

            n_threads = max(1, min(8, (os.cpu_count() or 4) - 2))

            # Check if this is a vision/multimodal model and load companion mmproj
            chat_handler = None
            clean = model_name.lower()
            if any(v in clean for v in ["llava", "vl", "vision"]):
                mmproj_candidates = [
                    str(Path(model_path).parent / f"{Path(model_path).stem}-mmproj.gguf"),
                    str(Path(model_path).parent / "llava-7b-mmproj.gguf"),
                    str(Path(__file__).parent.parent.parent / "models" / "llava-7b-mmproj.gguf"),
                ]
                mmproj_path = None
                for cand in mmproj_candidates:
                    if os.path.exists(cand):
                        mmproj_path = cand
                        break
                if mmproj_path:
                    try:
                        from llama_cpp.llama_chat_format import Llava15ChatHandler
                        logger.info(
                            f"Initializing Llava15ChatHandler for {model_name} with mmproj {mmproj_path}"
                        )
                        chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path, verbose=False)
                    except Exception as e:
                        logger.error(
                            f"Failed to initialize Llava15ChatHandler with {mmproj_path}: {e}",
                            exc_info=True,
                        )
                else:
                    logger.warning(f"No mmproj companion file found for vision model {model_name}")

            try:
                logger.info(
                    f"Loading model {model_name} from {model_path} "
                    f"(n_gpu_layers={n_gpu}, backend={backend}, n_threads={n_threads}, chat_handler={'yes' if chat_handler else 'no'})"
                )
                t_load_start = time.time()
                model_handle = llama_cpp.Llama(
                    model_path=model_path,
                    chat_handler=chat_handler,
                    n_ctx=2048,
                    n_gpu_layers=n_gpu,
                    n_threads=n_threads,
                    n_batch=512,
                    verbose=False,
                )
                t_load = time.time() - t_load_start
                logger.info(
                    f"Model {model_name} loaded in {t_load:.2f}s ({backend})"
                )
            except Exception as e:
                # Log the FULL traceback: a real load failure here (CUDA OOM,
                # corrupt GGUF, missing dependency...) is otherwise invisible —
                # the app silently degrades to MockLLM and the user never sees
                # why the real model did not come up.
                logger.warning(
                    f"Failed to load model {model_name} from {model_path}: {e}. "
                    f"Falling back to MockLLM.",
                    exc_info=True,
                )
                model_handle = self._mock_llm
        else:
            if not LLAMA_CPP_AVAILABLE:
                logger.warning(
                    f"Model file found at {model_path} but llama_cpp not installed. "
                    f"Using MockLLM fallback."
                )
            else:
                logger.warning(
                    f"Model file not found at {model_path}. Using MockLLM fallback."
                )
            model_handle = self._mock_llm

        self.resident_models[model_name] = model_handle
        self.vram_usage[model_name] = estimated_vram
        self._total_vram_used += estimated_vram

        self.audit.log_event(
            "MODEL_LOAD",
            {
                "model_name": model_name,
                "model_path": model_path,
                "vram_allocated_gb": estimated_vram,
                "total_vram_used_gb": self._total_vram_used,
                "vram_budget_gb": self.max_vram_gb,
                "resident_count": len(self.resident_models),
                "tier": self.hardware_tier,
                "using_mock": isinstance(model_handle, MockLLM),
            },
        )
        logger.info(
            f"Loaded model: {model_name} ({estimated_vram} GB) "
            f"[{'MockLLM' if isinstance(model_handle, MockLLM) else 'real'}]. "
            f"Total: {self._total_vram_used}/{self.max_vram_gb} GB"
        )
        return model_handle

    @staticmethod
    def _fit_gpu_layers(
        model_path: str,
        planned_layers: int,
        free_vram_gb: Optional[float],
    ) -> int:
        """
        Cap the number of GPU-offloaded layers so the model weights plus the
        CUDA/compute headroom fit in the currently-free VRAM.

        - ``planned_layers == -1`` means "full offload" (small models).
        - Returns 0 when even one layer would not fit — the model then runs on
          CPU instead of crashing with a CUDA out-of-memory 500.
        - Layer count per architecture is approximated from the model name
          (7B => ~32, 3B => ~28, 0.5B => ~24, 14B => ~40); the exact value
          only shifts the estimate slightly and is safe because we clamp.
        """
        if free_vram_gb is None or free_vram_gb <= 0:
            return planned_layers
        try:
            file_gb = os.path.getsize(model_path) / 1e9
        except OSError:
            return planned_layers
        if file_gb <= 0:
            return planned_layers

        base = os.path.basename(model_path).lower()
        if "14b" in base:
            total_layers = 40
        elif "7b" in base:
            total_layers = 32
        elif "3b" in base:
            total_layers = 28
        elif "4b" in base:
            total_layers = 32
        elif "0.5b" in base:
            total_layers = 24
        else:
            total_layers = 32

        # Headroom for the CUDA context, KV cache (n_ctx=2048) and compute
        # buffers that live on the GPU alongside the offloaded weights.
        reserve_gb = 1.0
        weights_budget_gb = free_vram_gb - reserve_gb
        if weights_budget_gb <= 0:
            return 0

        per_layer_gb = file_gb / total_layers
        max_fit = int(weights_budget_gb // per_layer_gb)

        if planned_layers == -1:
            # Full offload: put as many layers on GPU as fit (<= total).
            return max(0, min(total_layers, max_fit))
        return max(0, min(planned_layers, max_fit))

    def generate(self, model_name: str, prompt: str, **kwargs) -> str:
        """Generate text using the specified model."""
        model = self.load_model(model_name, reject_oversized=False)

        if isinstance(model, MockLLM):
            output = model.create_completion(prompt, **kwargs)
            return output["choices"][0]["text"]

        if isinstance(model, _StubModel):
            return f"[StubResponse] Input: {prompt[:100]}"

        with self._infer_lock:
            try:
                default_stops = [
                    "<|im_end|>", "<|im_start|>", "<|endoftext|>",
                    "<|eot_id|>", "<|end_of_text|>", "<｜end of sentence｜>", "<|end|>",
                    "user:", "\nuser:"
                ]
                output = model.create_completion(
                    prompt,
                    max_tokens=kwargs.get("max_tokens", 256),
                    temperature=kwargs.get("temperature", 0.7),
                    repeat_penalty=kwargs.get("repeat_penalty", 1.2),
                    stop=kwargs.get("stop", default_stops),
                )
                return output["choices"][0]["text"].strip()
            except ModelInferenceError:
                raise
            except Exception as e:
                # Log the FULL traceback so the real cause (OOM, bad path,
                # dependency error...) is never swallowed into a bare 500.
                logger.error(f"Generation error on {model_name}: {e}", exc_info=True)
                if _looks_like_cuda_oom(e):
                    raise _oom_error_for(model_name, e) from e
                if _looks_like_context_overflow(e):
                    raise _overflow_error_for(model_name, e) from e
                raise

    def generate_from_messages(
        self, model_name: str, messages: List[dict], **kwargs
    ) -> str:
        """
        Generate text from a list of chat messages.
        Uses MockLLM's create_chat_completion for mock mode.
        """
        model = self.load_model(model_name, reject_oversized=False)

        # Check for multimodal image content
        has_image = False
        image_details = []
        if isinstance(messages, list):
            for m in messages:
                if isinstance(m, dict):
                    content = m.get("content")
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "image_url":
                                has_image = True
                                url = item.get("image_url", {}).get("url", "")
                                prefix = url[:30]
                                image_details.append(f"format='{prefix}...', len={len(url)}")

        if has_image:
            logger.info(
                f"[MULTIMODAL PAYLOAD OUTGOING] Model: {model_name}, "
                f"Images attached: {len(image_details)}, Details: {'; '.join(image_details)}"
            )

        if isinstance(model, MockLLM):
            output = model.create_chat_completion(messages, **kwargs)
            return output["choices"][0]["text"]

        if isinstance(model, _StubModel):
            return f"[StubResponse] Input: {str(messages)[:100]}"

        if (
            has_image
            and LLAMA_CPP_AVAILABLE
            and isinstance(model, llama_cpp.Llama)
            and getattr(model, "chat_handler", None) is None
        ):
            raise ModelInferenceError(
                "image_attach_failed",
                f"Model '{model_name}' does not have a multimodal vision chat handler configured."
            )

        # For real llama.cpp models, use create_chat_completion to leverage
        # the model's native GGUF chat template (with <|im_start|>/<|im_end|>)
        try:
            with self._infer_lock:
                default_stops = [
                    "<|im_end|>", "<|im_start|>", "<|endoftext|>",
                    "<|eot_id|>", "<|end_of_text|>", "<｜end of sentence｜>", "<|end|>"
                ]
                output = model.create_chat_completion(
                    messages=messages,
                    max_tokens=kwargs.get("max_tokens", 256),
                    temperature=kwargs.get("temperature", 0.7),
                    repeat_penalty=kwargs.get("repeat_penalty", 1.2),
                    stop=kwargs.get("stop", default_stops),
                )
                choice = output["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"].strip()
                elif "text" in choice:
                    return choice["text"].strip()
                return str(output)
        except ModelInferenceError:
            raise
        except Exception as e:
            # OOM / context-overflow must surface immediately as structured
            # errors — retrying the identical request (on CPU or as a prompt) is
            # pointless and doubles the wait. Everything else falls back.
            logger.error(
                f"create_chat_completion failed on {model_name}: {e}",
                exc_info=True,
            )
            if _looks_like_cuda_oom(e):
                raise _oom_error_for(model_name, e) from e
            if _looks_like_context_overflow(e):
                raise _overflow_error_for(model_name, e) from e
            if has_image:
                # Do NOT silently fall back to prompt text generation when an image is present!
                # That drops the image data and causes the model to respond with generic non-answers.
                raise
            logger.warning(
                f"create_chat_completion failed ({e}), falling back to prompt completion"
            )
            prompt = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in messages
                if isinstance(m, dict)
            )
            return self.generate(model_name, prompt, **kwargs)

    def pin_model(self, model_name: str) -> dict:
        """
        Pin a model into resident memory.
        Validates whether the model's required VRAM fits within the tier's effective budget.
        Rejects pinning if model exceeds tier ceiling or available budget.
        """
        estimated_vram = self._estimate_model_vram(model_name)
        self.refresh_vram_budget()
        if estimated_vram > self.max_vram_gb:
            err_msg = (
                f"Cannot pin {model_name}: requires {estimated_vram:.1f} GB VRAM, "
                f"which exceeds the effective budget of {self.max_vram_gb:.1f} GB "
                f"(Tier ceiling: {self.static_max_vram_gb:.1f} GB)."
            )
            self.audit.log_event(
                "MODEL_PIN_REJECTED",
                {
                    "model_name": model_name,
                    "vram_required_gb": estimated_vram,
                    "vram_budget_gb": self.max_vram_gb,
                    "tier": self.hardware_tier,
                },
            )
            logger.warning(err_msg)
            raise ValueError(err_msg)

        # Load the model into memory
        self.load_model(model_name, reject_oversized=True)
        self.pinned_model = model_name
        self.audit.log_event(
            "MODEL_PINNED",
            {
                "model_name": model_name,
                "vram_allocated_gb": estimated_vram,
                "tier": self.hardware_tier,
            },
        )
        logger.info(f"Pinned model: {model_name} ({estimated_vram:.1f} GB)")
        return {
            "status": "pinned",
            "pinned_model": model_name,
            "vram_gb": estimated_vram,
            "resident_models": list(self.resident_models.keys()),
        }

    def unpin_model(self) -> dict:
        """Unpin the currently pinned model, allowing normal LRU eviction."""
        prev = self.pinned_model
        self.pinned_model = None
        if prev:
            self.audit.log_event("MODEL_UNPINNED", {"model_name": prev})
            logger.info(f"Unpinned model: {prev}")
        return {"status": "unpinned", "previous_pinned": prev}

    def unload_all(self) -> None:
        self.pinned_model = None
        while self.resident_models:
            self._evict_lru()

    def get_status(self) -> dict:
        free_vram = query_free_vram_gb()
        used_vram = query_used_vram_gb()
        total_vram = query_total_vram_gb()
        return {
            "tier": self.hardware_tier,
            "max_vram_gb": self.static_max_vram_gb,
            "static_ceiling_gb": self.static_max_vram_gb,
            "effective_budget_gb": self.max_vram_gb,
            "live_free_vram_gb": free_vram,
            "live_used_vram_gb": used_vram,
            "live_total_vram_gb": total_vram,
            "total_vram_used_gb": self._total_vram_used,
            "pinned_model": self.pinned_model,
            "resident_models": {
                name: {
                    "vram_gb": self.vram_usage.get(name, 0),
                    "type": type(handle).__name__,
                    "pinned": (name == self.pinned_model),
                }
                for name, handle in self.resident_models.items()
            },
        }


_global_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Return the global ModelManager singleton instance."""
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = ModelManager()
    return _global_model_manager
