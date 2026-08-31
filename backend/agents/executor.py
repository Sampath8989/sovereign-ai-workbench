"""
Executor: Tool dispatcher that executes individual steps from the planner's plan.
Routes each step to the appropriate tool (file_io, llm, code).
"""

import json
import logging
from typing import Dict

from backend.core.model_manager import ModelManager
from backend.tools.file_io import read_file, write_file
from backend.tools.calculator import solve_expression
from backend.tools.doc_generator import generate_doc
from backend.tools.ppt_generator import generate_ppt
from backend.tools.spreadsheet_analyzer import read_sheet
from backend.tools.spreadsheet_generator import generate_sheet
from backend.tools.pid_extractor import extract_topology
from backend.tools.handwriting_triage import read_note
from backend.tools.photo_analyzer import analyze_nameplate

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
    elif tool == "calculator":
        result = _execute_calculator(args)
    elif tool == "doc_generator":
        result = _execute_doc_generator(args)
    elif tool == "ppt_generator":
        result = _execute_ppt_generator(args)
    elif tool == "spreadsheet_generator":
        result = _execute_spreadsheet_generator(args)
    elif tool == "spreadsheet_analyzer":
        result = _execute_spreadsheet_analyzer(args)
    elif tool == "pid_extractor":
        result = _execute_pid_extractor(args)
    elif tool == "handwriting_triage":
        result = _execute_handwriting_triage(args)
    elif tool == "photo_analyzer":
        result = _execute_photo_analyzer(args)
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


def _execute_calculator(args: list) -> str:
    """Execute the symbolic calculator tool."""
    expression = args[0] if args else ""
    return solve_expression(expression)


def _execute_doc_generator(args: list) -> str:
    """Execute the Word document generator tool."""
    filename = args[0] if len(args) > 0 else "output.docx"
    title = args[1] if len(args) > 1 else "Untitled"
    content = args[2] if len(args) > 2 else ""
    return generate_doc(filename, title, content)


def _execute_ppt_generator(args: list) -> str:
    """Execute the PowerPoint generator tool."""
    filename = args[0] if len(args) > 0 else "output.pptx"
    title = args[1] if len(args) > 1 else "Untitled"
    bullet_points = args[2] if len(args) > 2 else []
    if isinstance(bullet_points, str):
        bullet_points = [bullet_points]
    return generate_ppt(filename, title, bullet_points)


def _execute_spreadsheet_generator(args: list) -> str:
    """Execute the spreadsheet generator tool."""
    filename = args[0] if len(args) > 0 else "output.xlsx"
    data = args[1] if len(args) > 1 else [["", ""]]
    return generate_sheet(filename, data)


def _execute_spreadsheet_analyzer(args: list) -> str:
    """Execute the spreadsheet analyzer tool."""
    filename = args[0] if len(args) > 0 else ""
    cell_range = args[1] if len(args) > 1 else "A1:D50"
    data = read_sheet(filename, cell_range)
    # Return data as a string representation
    return str(data)


def _execute_pid_extractor(args: list) -> str:
    """Execute the P&ID topology extractor tool."""
    image_path = args[0] if args else "workspace/sandbox_files/test_pid.png"
    result = extract_topology(image_path)
    return json.dumps(result)


def _execute_handwriting_triage(args: list) -> str:
    """Execute the handwriting triage reader tool."""
    image_path = args[0] if args else "workspace/sandbox_files/test_note.jpg"
    result = read_note(image_path)
    return json.dumps(result)


def _execute_photo_analyzer(args: list) -> str:
    """Execute the field photo analyzer tool."""
    image_path = args[0] if args else "workspace/sandbox_files/test_photo.jpg"
    result = analyze_nameplate(image_path)
    return json.dumps(result)
