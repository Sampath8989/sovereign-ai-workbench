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

    Uses MockLLM for plan generation because small local models (0.5B) cannot
    reliably produce structured JSON. The real model is used for synthesis
    in synthesize_node, where natural language quality matters.

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
        # Always use MockLLM for planning — small models can't produce
        # reliable JSON. The real model is used for synthesis instead.
        from backend.core.model_manager import MockLLM
        mock_llm = MockLLM()
        output = mock_llm.create_chat_completion(messages)
        response = output["choices"][0]["text"]
        logger.info(f"Planner raw response: {response[:200]}")

        # Check if MockLLM returned a direct response (greeting/simple chat)
        # instead of a JSON plan. Direct responses start with [MockLLM] prefix
        # and don't contain JSON structure.
        if response.startswith("[MockLLM] "):
            direct_text = response[len("[MockLLM] "):]
            # If it doesn't look like JSON, it's a direct conversational response.
            # Wrap it as a plan with a special marker so the graph can detect it.
            if not direct_text.strip().startswith('{') and not direct_text.strip().startswith('['):
                if mock_llm._is_greeting(prompt):
                    logger.info("Planner: MockLLM returned direct response for greeting")
                    return [{"tool": "llm", "action": "summarize", "args": [direct_text], "direct_response": True}]
                else:
                    logger.info("Planner: MockLLM returned direct string for non-greeting; using fallback plan")
                    return _make_fallback(prompt)

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
                logger.info("Planner using MockLLM for structured plan generation")

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


def is_direct_response(plan: List[dict]) -> bool:
    """Check if a plan is actually a direct response (not a task plan).

    When MockLLM detects greetings or simple chat, it returns a plan containing
    the response text in the first step's args rather than a real task plan.
    This flag signals the orchestrator to skip the execute→retrieve→verify
    pipeline and use the response as-is.
    """
    if not plan or not isinstance(plan, list):
        return False
    step = plan[0]
    if not isinstance(step, dict):
        return False
    # Explicit direct response marker
    if step.get("direct_response") is True:
        return True
    if step.get("action") == "direct_response" or step.get("tool") == "direct_response":
        return True
    # Direct response has tool=llm, action=summarize, and args containing a greeting response
    if step.get("tool") == "llm" and step.get("action") == "summarize":
        args = step.get("args", [])
        if args and isinstance(args[0], str):
            text = args[0]
            if "Sovereign AI Workbench" in text and ("Hello!" in text or "locally-hosted" in text):
                return True
    return False
