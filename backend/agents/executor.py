"""
Executor: Tool dispatcher that executes individual steps from the planner's plan.
Routes each step to the appropriate tool (file_io, llm, code).
"""

import logging
from typing import Dict

from backend.core.model_manager import ModelManager
from backend.tools.file_io import read_file, write_file

logger = logging.getLogger(__name__)


def execute_step(step: dict, context: dict, model_manager: ModelManager = None) -> str:
    """
    Execute a single step from the plan.

    Args:
        step: A dict with keys "tool", "action", "args".
        context: Accumulated context from previous steps. Updated in-place.
        model_manager: ModelManager instance for LLM calls.

    Returns:
        The result of executing the step.
    """
    tool = step.get("tool", "")
    action = step.get("action", "")
    args = step.get("args", [])

    logger.info(f"Executing step: tool={tool}, action={action}, args={args}")

    result = ""

    if tool == "file_io":
        result = _execute_file_io(action, args)
    elif tool == "llm":
        result = _execute_llm(action, args, context, model_manager)
    elif tool == "code":
        result = _execute_code(action, args)
    else:
        result = f"Error: Unknown tool '{tool}'"

    # Store result in context
    step_index = len([k for k in context if k.startswith("step_")])
    context[f"step_{step_index}_result"] = result
    context[f"step_{step_index}_tool"] = tool
    context[f"step_{step_index}_action"] = action

    logger.info(f"Step result: {result[:200]}")
    return result


def _execute_file_io(action: str, args: list) -> str:
    """Execute a file_io tool step."""
    if action == "read":
        filename = args[0] if args else ""
        return read_file(filename)
    elif action == "write":
        filename = args[0] if len(args) > 0 else ""
        content = args[1] if len(args) > 1 else ""
        return write_file(filename, content)
    else:
        return f"Error: Unknown file_io action '{action}'"


def _execute_llm(
    action: str, args: list, context: dict, model_manager: ModelManager = None
) -> str:
    """Execute an LLM tool step."""
    if action == "summarize":
        # If args contain text, use it. Otherwise, compile from context.
        if args and args[0]:
            text_to_summarize = args[0]
        else:
            # Gather results from previous steps
            text_to_summarize = " ".join(
                v for k, v in sorted(context.items())
                if k.endswith("_result")
            )
            if not text_to_summarize:
                text_to_summarize = "No context available."

        if model_manager is None:
            model_manager = ModelManager()

        from backend.config import get_coder_model
        model_name = get_coder_model()

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Summarize the following content concisely.",
            },
            {"role": "user", "content": text_to_summarize},
        ]

        return model_manager.generate_from_messages(model_name, messages)
    else:
        return f"Error: Unknown LLM action '{action}'"


def _execute_code(action: str, args: list) -> str:
    """Execute a code tool step."""
    if action == "execute":
        code = args[0] if args else ""
        # Use the sandbox manager for safe execution
        try:
            from backend.core.sandbox_manager import SandboxManager
            sm = SandboxManager()
            result = sm.execute_code(code)
            if result.get("exit_code", -1) == 0:
                return result.get("stdout", "")
            else:
                return f"Error (exit {result.get('exit_code')}): {result.get('stderr', '')}"
        except Exception as e:
            return f"Error executing code: {e}"
    else:
        return f"Error: Unknown code action '{action}'"
