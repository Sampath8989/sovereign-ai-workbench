"""
Shared Confidence Warning Helpers
=================================
Reusable logic for applying low-confidence warnings across all vision tools.
Prevents drift between handwriting_triage, pid_extractor, and photo_analyzer.

Limitation: This is MOCK/DEMO behavior — the confidence values come from
MockVisionModel and are not calibrated real model confidence scores.
"""

import logging

logger = logging.getLogger(__name__)

# Threshold below which a "LOW CONFIDENCE" warning is prepended.
CONFIDENCE_THRESHOLD = 0.6


def safe_confidence(confidence, default: float = 0.0) -> float:
    """
    Ensure confidence is a valid float.  Fail-safe: None, missing, or
    non-numeric values default to ``default`` (0.0), which will trigger
    the low-confidence warning.
    """
    if confidence is None or not isinstance(confidence, (int, float)):
        return default
    return float(confidence)


def apply_confidence_warning(text: str, confidence: float, tool_name: str = "tool") -> str:
    """
    Prepend the low-confidence warning prefix if confidence is below threshold.

    Args:
        text: The raw output text from the vision model.
        confidence: The confidence score (0.0–1.0).
        tool_name: Name of the calling tool (for logging).

    Returns:
        The text with warning prefix if below threshold, or original text.
    """
    if confidence < CONFIDENCE_THRESHOLD and text:
        logger.warning(
            f"{tool_name} confidence {confidence:.3f} below {CONFIDENCE_THRESHOLD} threshold"
        )
        return f"\u26a0\ufe0f LOW CONFIDENCE - HUMAN REVIEW REQUIRED: {text}"
    return text
