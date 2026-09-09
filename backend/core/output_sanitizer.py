"""
Output Sanitizer: strips internal orchestrator state from user-facing text.

This runs at the response-formatting boundary (backend/main.py) and is the
single, authoritative place where internal execution-trace tokens are removed
before a response reaches the client. It must never rely on the model to "not
mention" internal state — if a token appears, this module removes it.

Also strips DeepSeek R1-style thinking blocks so chain-of-thought
never renders raw in the chat UI.
"""

import re

# Internal scratchpad keys produced by the executor: step_0_result, step_1_tool,
# step_2_action, etc. Both the bare token and the "Step step_0_result: ..."
# label form are stripped.
_STEP_KEY_RE = re.compile(r'\bstep_\d+_(?:result|tool|action)\b', re.IGNORECASE)
_STEP_LABEL_RE = re.compile(r'\bStep\s+step_\d+_(?:result|tool|action)\s*:?', re.IGNORECASE)

# Section markers the orchestrator injects into LLM prompts. If a weak model
# echoes them back, they must not reach the client.
_SECTION_MARKER_RE = re.compile(
    r'\b(?:User request|Execution results|Retrieved sources|Sources|Generated text)\s*:',
    re.IGNORECASE,
)

# Raw MockLLM/plan JSON leaked into output: {"mock": true, "plan": [...]}
_PLAN_JSON_RE = re.compile(r'\{\s*"mock"\s*:\s*true\s*,\s*"plan"\s*:\s*\[.*?\]\s*\}', re.DOTALL)
_TOOL_PLAN_JSON_RE = re.compile(r'\{\s*"tool"\s*:\s*"[a-z_0-9]+"\s*,\s*"action"\s*:\s*"[a-z_0-9]+"\s*,[^}]*\}', re.DOTALL)
_OPEN_TOOL_PLAN_RE = re.compile(r'\{\s*"tool"\s*:\s*"[a-z_0-9]+"\s*,')

# DeepSeek R1 reasoning blocks: <think> ... </think> and <thinking> ... </thinking>
_REASONING_BLOCK_RE = re.compile(
    r'\s*<(?:think|thinking)>.*?</(?:think|thinking)>\s*',
    re.DOTALL | re.IGNORECASE,
)

# Verbose "Task completed. Result: ..." wrapper around raw tool output
_TASK_COMPLETED_RE = re.compile(r'^Task completed\.\s*Result:\s*', re.IGNORECASE)

# Residual closing brackets from partially-collapsed plan JSON (e.g. "]}",
# "]}]" after the inner objects were removed) and the "[Warning: " wrapper
# the verifier may wrap around leaked plan JSON.
_RESIDUAL_CLOSE_RE = re.compile(r'\s*\]\s*\}\s*\]?\s*$')
_RESIDUAL_WARNING_OPEN_RE = re.compile(r'\[\s*Warning\s*:\s*$')


def strip_internal_trace_tokens(text: str, strip_reasoning: bool = False) -> str:
    """
    Remove all internal execution-trace tokens and raw plan JSON from ``text``.

    Args:
        text: Input string to sanitize.
        strip_reasoning: If True, remove <think>...</think> blocks completely
            (useful for docx/pptx export). If False, preserve them for UI-level
            folding/collapsing.

    Idempotent. Safe to call at any boundary; intended for the final
    response-formatting step before the client receives output.
    """
    if not text:
        return text

    cleaned = text

    # 1. Reasoning blocks (optional: when exporting docs, strip entirely)
    if strip_reasoning:
        cleaned = _REASONING_BLOCK_RE.sub(' ', cleaned)

    # 2. "Step step_0_result: ..." scratchpad labels
    cleaned = _STEP_LABEL_RE.sub('', cleaned)

    # 3. Bare step_N_result / step_N_tool / step_N_action tokens
    cleaned = _STEP_KEY_RE.sub('', cleaned)

    # 4. Orchestrator prompt section markers
    cleaned = _SECTION_MARKER_RE.sub('', cleaned)

    # 5. Raw plan JSON (wrapped mock format and inline tool steps).
    #    Iterate because nested structures need several passes to fully
    #    collapse before their closing braces remain.
    for _ in range(3):
        cleaned = _PLAN_JSON_RE.sub(' ', cleaned)
        cleaned = _TOOL_PLAN_JSON_RE.sub(' ', cleaned)
        cleaned = _OPEN_TOOL_PLAN_RE.sub(' ', cleaned)

    # 5b. Residual bracket fragments from partially-collapsed plan JSON
    cleaned = _RESIDUAL_CLOSE_RE.sub('', cleaned)
    cleaned = _RESIDUAL_WARNING_OPEN_RE.sub('', cleaned)
    # Stray leading comma from a removed array element
    cleaned = re.sub(r'^\s*,\s*', '', cleaned)

    # 6. "Task completed. Result: " wrapper around raw tool output
    cleaned = _TASK_COMPLETED_RE.sub('', cleaned)

    # Collapse runaway whitespace left behind by removals
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()