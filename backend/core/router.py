"""
Semantic Router: Classifies user prompts into task categories.
Deterministic keyword-based routing — no ML required.
"""

import logging

logger = logging.getLogger(__name__)


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

    # CODE routing
    if any(kw in lower for kw in ("code", "script", "execute")):
        logger.info(f"Routing prompt to CODE: {prompt[:60]}")
        return "CODE"

    # FILE routing
    if any(kw in lower for kw in ("read", "file", "write")):
        logger.info(f"Routing prompt to FILE: {prompt[:60]}")
        return "FILE"

    # VISION routing
    if any(kw in lower for kw in ("image", "scan", "drawing")):
        logger.info(f"Routing prompt to VISION: {prompt[:60]}")
        return "VISION"

    # Default: TEXT
    logger.info(f"Routing prompt to TEXT: {prompt[:60]}")
    return "TEXT"
