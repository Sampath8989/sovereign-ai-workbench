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
import time
import logging
import uuid
from collections import OrderedDict
from typing import Dict, List, Optional, Union

from backend.config import get_model_path, get_model_roster, get_max_vram_gb, get_tier
from backend.core.audit_log import AuditLogger

logger = logging.getLogger(__name__)

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


MOCK_PREFIX = "[MockLLM] "


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


class MockLLM:
    """
    Deterministic mock LLM for testing without model weights.
    Returns hardcoded responses based on the prompt content.
    
    All responses are routed through _wrap_response() to ensure the
    [MockLLM] disclosure prefix is present on every output path.
    """

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
        if isinstance(messages, str):
            return messages
        if isinstance(messages, list):
            # Last user message is the actual user input
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role") == "user":
                    return m.get("content", "")
            # Fallback: last message content
            if messages:
                m = messages[-1]
                return m.get("content", "") if isinstance(m, dict) else str(m)
        return ""

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
        return False

    def create_chat_completion(self, messages: Union[str, List[dict]], **kwargs) -> dict:
        """
        Mock chat completion. Inspects the messages to return appropriate responses.

        Args:
            messages: Either a string prompt or a list of message dicts.

        Returns:
            A dict with 'choices' key mimicking llama-cpp-python output.
        """
        # Extract the full combined text for keyword matching
        if isinstance(messages, str):
            text = messages
        elif isinstance(messages, list):
            text = " ".join(
                m.get("content", "") if isinstance(m, dict) else str(m)
                for m in messages
            )
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

        # --- Deliverable synthesis tool triggers ---
        # Word document generation
        if any(kw in lower for kw in ["word document", "approval note", "docx"]):
            uid = uuid.uuid4().hex[:8]
            plan = [
                {"tool": "doc_generator", "action": "generate",
                 "args": [f"approval_{uid}.docx", "Approval Note", "This is safe."]}
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

            # Try to extract filename, fall back to unique name
            fname_match = re.search(r"named?\s+(\S+\.xlsx)", lower)
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

        # P&ID / topology extraction (negation-aware)
        pid_keywords = ["p&pid", "topology", "pid extractor", "extract topology"]
        if any(kw in lower and not _is_keyword_negated(lower, kw) for kw in pid_keywords):
            plan = [
                {"tool": "pid_extractor", "action": "extract",
                 "args": ["workspace/sandbox_files/test_pid.png"]}
            ]
            return self._wrap_plan(plan)

        # Handwriting triage (negation-aware)
        hw_keywords = ["handwriting", "handwritten", "read note", "field note"]
        if any(kw in lower and not _is_keyword_negated(lower, kw) for kw in hw_keywords):
            plan = [
                {"tool": "handwriting_triage", "action": "read",
                 "args": ["workspace/sandbox_files/test_note.jpg"]}
            ]
            return self._wrap_plan(plan)

        # Photo / nameplate analysis (negation-aware)
        photo_keywords = ["photo", "nameplate", "field photo", "equipment photo"]
        if any(kw in lower and not _is_keyword_negated(lower, kw) for kw in photo_keywords):
            plan = [
                {"tool": "photo_analyzer", "action": "analyze",
                 "args": ["workspace/sandbox_files/test_photo.jpg"]}
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

        # File I/O triggers
        user_lower = user_msg.lower()
        file_match = re.search(r'(?:read|write|open|load)\s+(?:the\s+)?(?:file\s+)?([a-zA-Z0-9_\-\./]+\.[a-zA-Z0-9]+)', user_lower)
        if (file_match or
                any(kw in user_lower for kw in ["read test.txt", "read the file", "file_io", "read file"]) or
                re.search(r'\bplan\b', user_lower) or re.search(r'\bsteps\b', user_lower) or re.search(r'\bdecompos', user_lower)):
            fname = file_match.group(1) if file_match else "test.txt"
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
            # Fallback JSON plan for non-greeting, unrecognized prompts
            plan = [
                {"tool": "llm", "action": "summarize", "args": [user_msg]}
            ]
            return self._wrap_plan(plan)

        # If asked to verify / citation check, return a deterministic verdict
        if 'verdict' in lower or 'citation verif' in lower or 'grounded' in lower:
            # Check whether source text actually appears in the generated text
            has_source_evidence = self._check_grounding(text)
            if has_source_evidence:
                return self._wrap_response(
                    "VERDICT: YES\nREASON: All claims in the text are present in the provided sources."
                )
            else:
                return self._wrap_response(
                    "VERDICT: NO\nREASON: Some claims in the text are not found in the provided sources."
                )

        # If source content is present in the prompt, prioritize returning grounded source text
        source_section = self._extract_sources(text)
        if source_section:
            return self._wrap_response(source_section)

        # If asked to summarize, return a mock summary
        if "summar" in user_lower:
            return self._wrap_response("This is a mock summary of the provided content.")

        return self._wrap_response(f"Processed: {text[:500]}")

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
        """If the prompt contains 'Retrieved sources:' or 'Sources:', return the
        full source text so the mock response echoes back the retrieved context."""
        for marker in ["Retrieved sources:", "Sources:", "sources:"]:
            idx = text.find(marker)
            if idx >= 0:
                return text[idx:]
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

        model_name, model_handle = self.resident_models.popitem(last=False)
        vram_freed = self.vram_usage.pop(model_name, 0.0)
        self._total_vram_used -= vram_freed

        if hasattr(model_handle, "close"):
            try:
                model_handle.close()
            except Exception as e:
                logger.warning(f"Error closing model {model_name}: {e}")

        self.audit.log_event(
            "MODEL_EVICTION",
            {
                "model_name": model_name,
                "vram_freed_gb": vram_freed,
                "remaining_vram_gb": self._total_vram_used,
                "tier": self.hardware_tier,
            },
        )
        logger.info(f"Evicted model: {model_name} (freed {vram_freed} GB)")
        return model_name

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

        # Try to load real model, fall back to MockLLM if unavailable
        if LLAMA_CPP_AVAILABLE and os.path.exists(model_path):
            # Determine GPU layers: use -1 (all) if CUDA compiled in, else 0 (CPU only)
            _cuda_available = False
            try:
                from llama_cpp import llama_cpp as _lc
                _cuda_available = hasattr(_lc, "ggml_backend_cuda_init")
            except Exception:
                pass
            n_gpu = -1 if _cuda_available else 0
            backend = "CUDA GPU" if _cuda_available else "CPU only"

            try:
                logger.info(
                    f"Loading model {model_name} from {model_path} "
                    f"(n_gpu_layers={n_gpu}, backend={backend})"
                )
                t_load_start = time.time()
                model_handle = llama_cpp.Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_gpu_layers=n_gpu,
                    verbose=False,
                )
                t_load = time.time() - t_load_start
                logger.info(
                    f"Model {model_name} loaded in {t_load:.2f}s ({backend})"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load model {model_name} from {model_path}: {e}. "
                    f"Using MockLLM fallback."
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

    def generate(self, model_name: str, prompt: str, **kwargs) -> str:
        """Generate text using the specified model."""
        model = self.load_model(model_name)

        if isinstance(model, MockLLM):
            output = model.create_chat_completion(prompt, **kwargs)
            return output["choices"][0]["text"]

        if isinstance(model, _StubModel):
            return f"[StubResponse] Input: {prompt[:100]}"

        try:
            output = model.create_completion(
                prompt,
                max_tokens=kwargs.get("max_tokens", 256),
                temperature=kwargs.get("temperature", 0.7),
                stop=kwargs.get("stop", None),
            )
            return output["choices"][0]["text"]
        except Exception as e:
            logger.error(f"Generation error on {model_name}: {e}")
            raise

    def generate_from_messages(
        self, model_name: str, messages: List[dict], **kwargs
    ) -> str:
        """
        Generate text from a list of chat messages.
        Uses MockLLM's create_chat_completion for mock mode.
        """
        model = self.load_model(model_name)

        if isinstance(model, MockLLM):
            output = model.create_chat_completion(messages, **kwargs)
            return output["choices"][0]["text"]

        if isinstance(model, _StubModel):
            return f"[StubResponse] Input: {str(messages)[:100]}"

        # For real llama.cpp models, flatten messages into a prompt
        prompt = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in messages
            if isinstance(m, dict)
        )
        return self.generate(model_name, prompt, **kwargs)

    def unload_all(self) -> None:
        while self.resident_models:
            self._evict_lru()

    def get_status(self) -> dict:
        free_vram = query_free_vram_gb()
        return {
            "tier": self.hardware_tier,
            "static_ceiling_gb": self.static_max_vram_gb,
            "effective_budget_gb": self.max_vram_gb,
            "live_free_vram_gb": free_vram,
            "total_vram_used_gb": self._total_vram_used,
            "resident_models": {
                name: {
                    "vram_gb": self.vram_usage.get(name, 0),
                    "type": type(handle).__name__,
                }
                for name, handle in self.resident_models.items()
            },
        }


class _StubModel:
    """Stub model for when llama.cpp is unavailable or model file is missing."""

    def __init__(self, model_name: str):
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
