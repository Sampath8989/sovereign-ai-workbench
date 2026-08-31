"""
Citation Verifier: CoT verification of generated text against sources.
Checks if all claims in the generated text are grounded in the provided sources.
"""

import logging
from typing import List, Dict

from backend.core.model_manager import ModelManager

logger = logging.getLogger(__name__)

VERIFY_SYSTEM_PROMPT = """You are a citation verifier. Given a generated text and a set of source documents,
determine if all factual claims in the text are grounded in the sources.

Rules:
- A claim is "grounded" if the information is present in or directly inferable from the sources.
- A claim is "not grounded" if it introduces information not found in any source.
- Be strict: if a specific number, name, or detail in the text doesn't appear in sources, it's not grounded.

Reply with EXACTLY this format:
VERDICT: YES or NO
REASON: <one sentence explanation>

Example:
VERDICT: YES
REASON: All claims about the corrosion limit match the source document."""


class CitationVerifier:
    """
    Verifies that generated text is grounded in retrieved sources.
    Falls back to MockLLM if no real model is available.
    """

    def __init__(self, model_manager: ModelManager = None):
        self.model_manager = model_manager or ModelManager()

    def verify(self, generated_text: str, sources: List[Dict]) -> Dict:
        """
        Verify that generated text is grounded in the provided sources.

        Args:
            generated_text: The text to verify.
            sources: List of {"text": "...", "metadata": {...}} dicts.

        Returns:
            {"grounded": bool, "reason": str}
        """
        if not generated_text or not sources:
            return {"grounded": False, "reason": "No text or no sources provided."}

        # Build source context
        source_text = "\n\n".join(
            f"[Source {i+1}: {s.get('metadata', {}).get('source', 'unknown')}]\n{s['text']}"
            for i, s in enumerate(sources[:5])  # limit to top 5 sources
        )

        messages = [
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Generated text:\n{generated_text}\n\nSources:\n{source_text}"},
        ]

        try:
            from backend.config import get_coder_model
            model_name = get_coder_model()
            response = self.model_manager.generate_from_messages(model_name, messages)
            return self._parse_verification(response)
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {"grounded": False, "reason": f"Verification error: {e}"}

    def _parse_verification(self, response: str) -> Dict:
        """Parse the verification response into a structured result."""
        response_upper = response.upper()

        if "VERDICT: YES" in response_upper:
            reason = self._extract_reason(response)
            return {"grounded": True, "reason": reason}
        elif "VERDICT: NO" in response_upper:
            reason = self._extract_reason(response)
            return {"grounded": False, "reason": reason}
        else:
            # Fallback: check for YES/NO keywords
            if "YES" in response_upper and "NO" not in response_upper:
                return {"grounded": True, "reason": response[:200]}
            elif "NO" in response_upper and "YES" not in response_upper:
                return {"grounded": False, "reason": response[:200]}
            else:
                return {"grounded": False, "reason": f"Ambiguous verification: {response[:200]}"}

    def _extract_reason(self, response: str) -> str:
        """Extract the REASON line from the verification response."""
        for line in response.split("\n"):
            if line.strip().upper().startswith("REASON:"):
                return line.strip()[7:].strip()
        return response[:200]
