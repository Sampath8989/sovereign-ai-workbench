"""
Planner: Task decomposer using the ReWOO (Reasoning WithOut Observation) pattern.
Generates a JSON plan of steps from a user prompt.
"""

import json
import logging
from typing import List

from backend.core.model_manager import ModelManager

logger = logging.getLogger(__name__)

# The system prompt that instructs the LLM to produce a JSON plan
PLAN_SYSTEM_PROMPT = """You are a task planner. Given a user request, produce a JSON array of steps.
Each step is an object with keys: "tool", "action", "args".

Available tools:
- "file_io": actions "read" (args: [filename]) or "write" (args: [filename, content])
- "llm": action "summarize" (args: [text_or_placeholder])
- "code": action "execute" (args: [code_string])

Output ONLY the JSON array. No explanation. Example:
[{"tool": "file_io", "action": "read", "args": ["test.txt"]}, {"tool": "llm", "action": "summarize", "args": []}]
"""

# Fallback plan when LLM output cannot be parsed
FALLBACK_PLAN: List[dict] = [
    {"tool": "llm", "action": "summarize", "args": []}
]


def generate_plan(prompt: str, model_manager: ModelManager = None) -> List[dict]:
    """
    Generate a plan (list of step dicts) from a user prompt.

    Args:
        prompt: The user's request.
        model_manager: ModelManager instance to use for generation.

    Returns:
        A list of dicts, each with keys: tool, action, args.
    """
    if model_manager is None:
        model_manager = ModelManager()

    messages = [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        # Use the router model for planning
        from backend.config import get_router_model
        model_name = get_router_model()

        response = model_manager.generate_from_messages(model_name, messages)
        logger.info(f"Planner raw response: {response[:200]}")

        # Parse JSON from the response
        # Strip markdown code fences if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            # Remove ```json ... ``` wrapper
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        plan = json.loads(cleaned)

        # Handle MockLLM wrapped format: {"mock": true, "plan": [...]}
        if isinstance(plan, dict) and "plan" in plan:
            is_mock = plan.get("mock", False)
            plan = plan["plan"]
            if is_mock:
                logger.info("Planner received MockLLM response (mock=True)")

        if not isinstance(plan, list):
            logger.warning(f"Planner returned non-list: {type(plan)}. Using fallback.")
            return _make_fallback(prompt)

        logger.info(f"Planner generated {len(plan)} steps")
        return plan

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse planner output: {e}. Using fallback plan.")
        return _make_fallback(prompt)
    except Exception as e:
        logger.error(f"Planner error: {e}. Using fallback plan.")
        return _make_fallback(prompt)


def _make_fallback(prompt: str) -> List[dict]:
    """Create a fallback plan that sends the prompt directly to the LLM."""
    return [{"tool": "llm", "action": "summarize", "args": [prompt]}]
