"""
Model Manager: VRAM allocator & llama.cpp hot-swap engine.
Loads, evicts, and generates from GGUF models with VRAM-aware scheduling.

MockLLM fallback: If model files are missing or llama-cpp-python fails to load,
a deterministic MockLLM is used so the agent graph can still be tested end-to-end.
"""

import json
import os
import subprocess
import time
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Union

from backend.config import get_model_path, get_model_roster, get_max_vram_gb, get_tier
from backend.core.audit_log import AuditLogger

logger = logging.getLogger(__name__)

# Try importing llama_cpp
try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama_cpp not installed. Model loading will use stub implementation.")

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


class MockLLM:
    """
    Deterministic mock LLM for testing without model weights.
    Returns hardcoded responses based on the prompt content.
    """

    def create_chat_completion(self, messages: Union[str, List[dict]], **kwargs) -> dict:
        """
        Mock chat completion. Inspects the messages to return appropriate responses.

        Args:
            messages: Either a string prompt or a list of message dicts.

        Returns:
            A dict with 'choices' key mimicking llama-cpp-python output.
        """
        # Extract the text content from messages
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

        # If asked for a JSON plan, return a hardcoded plan
        if "plan" in lower or "steps" in lower or "decompose" in lower:
            plan = json.dumps([
                {"tool": "file_io", "action": "read", "args": ["test.txt"]},
                {"tool": "llm", "action": "summarize", "args": []}
            ])
            return {"choices": [{"text": plan}]}

        # If asked to summarize, return a mock summary
        if "summar" in lower:
            return {"choices": [{"text": "This is a mock summary of the provided content."}]}

        # Default response
        return {"choices": [{"text": f"[MockLLM] Processed: {text[:120]}"}]}

    def create_completion(self, prompt: str, **kwargs) -> dict:
        """Alias for create_chat_completion with a plain string."""
        return self.create_chat_completion(prompt, **kwargs)

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
            try:
                model_handle = llama_cpp.Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_gpu_layers=-1,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load model {model_name} from {model_path}: {e}. "
                    f"Using MockLLM fallback."
                )
                model_handle = self._mock_llm
        else:
            logger.info(
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
