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

# Cap the text handed to an LLM summarize step. The models run with a 2048-token
# context window, so dumping a long file (or worse, binary decoded as text) into
# the prompt makes llama.cpp raise "Requested tokens (...) exceed context window".
# Roughly 3000 chars fit comfortably alongside the system prompt and the reply.
_MAX_SUMMARIZE_CONTEXT_CHARS = 3000


def _cap_context(text: str, limit: int = _MAX_SUMMARIZE_CONTEXT_CHARS) -> str:
    """Truncate context text to ``limit`` chars with an explicit marker."""
    if not text or len(text) <= limit:
        return text
    return text[:limit] + "\n\n... [content truncated to fit the model context window]"


# Request-scoped set of already-emitted artifact keys (tool, casefolded filename).
# Cleared per request so one generation request produces exactly one artifact
# per output filename. This is the isolation boundary for deliverable writes.
_emitted_artifacts: set = set()


def reset_artifact_tracking() -> None:
    """Clear the request-scoped artifact tracker (call at request start)."""
    _emitted_artifacts.clear()


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

    # --- Single-artifact guarantee (Bug 9) ---
    # File-generating tools (doc/ppt/sheet) must never write a second artifact
    # for the same filename within one request. Case-variant duplicates
    # ("Report.docx" vs "report.docx") collapse to the first writer.
    if tool in {"doc_generator", "ppt_generator", "spreadsheet_generator"}:
        fname = args[0] if args else ""
        if fname:
            key = (tool, fname.lower())
            if key in _emitted_artifacts:
                logger.info(
                    f"execute_step: duplicate artifact {tool}/{fname} already "
                    f"emitted this request; skipping duplicate write"
                )
                return f"Deliverable already generated: {fname}"
            _emitted_artifacts.add(key)

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
            # If there are no prior step results in context and user did not ask to summarize,
            # this was an unneeded fallback step; return directly to avoid wasting CPU inference
            has_prior_results = any(k.endswith("_result") for k in context)
            if not has_prior_results and "summar" not in text_to_summarize.lower():
                return text_to_summarize
        else:
            # Gather results from previous steps
            text_to_summarize = " ".join(
                v for k, v in sorted(context.items())
                if k.endswith("_result")
            )
            if not text_to_summarize:
                return "Ready for synthesis."

        # Never send the raw (possibly multi-hundred-KB / binary-decoded) text
        # to the model — truncate so the prompt fits the 2048-token window.
        text_to_summarize = _cap_context(text_to_summarize)

        if model_manager is None:
            from backend.core.model_manager import get_model_manager
            model_manager = get_model_manager()

        from backend.config import get_coder_model
        # Reuse resident model if available to prevent model eviction thrashing on CPU
        model_name = None
        if hasattr(model_manager, "resident_models") and model_manager.resident_models:
            model_name = list(model_manager.resident_models.keys())[0]
        if not model_name:
            model_name = get_coder_model()

        # MockLLM summarize is a pure-text path: it must never return a nested
        # tool plan (which would leak into the pipeline as fabricated JSON).
        from backend.core.model_manager import MockLLM
        model = model_manager.load_model(model_name, reject_oversized=False)
        if isinstance(model, MockLLM):
            return model.summarize(text_to_summarize)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Summarize the following content concisely.",
            },
            {"role": "user", "content": text_to_summarize},
        ]

        return model_manager.generate_from_messages(model_name, messages, max_tokens=256)
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
    """Execute the document generator tool (PDF or Word)."""
    filename = args[0] if len(args) > 0 else "output.docx"
    title = args[1] if len(args) > 1 else "Untitled"
    content = args[2] if len(args) > 2 else ""
    output_format = args[3] if len(args) > 3 else None
    return generate_doc(filename, title, content, output_format=output_format)


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
