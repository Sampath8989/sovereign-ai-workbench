"""
LangGraph State Machine: Orchestrates the ReWOO agent pipeline.
Pipeline: START → plan_node → execute_node → retrieve_node → synthesize_node → END
"""

import concurrent.futures
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Annotated, List, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.core.model_manager import ModelManager
from backend.agents.planner import generate_plan, is_direct_response
from backend.agents.executor import execute_step
from backend.tools.rag_search import get_rag
from backend.tools.citation_tagger import tag_citations
from backend.agents.verifier import CitationVerifier
from backend.core.router import classify_intent, get_direct_response, is_trivial_query

logger = logging.getLogger(__name__)

# Keep prompts inside the 2048-token llama.cpp context window. Roughly 3.5-4
# chars per token for typical text; these caps leave room for the instruction
# prompt and the requested output tokens.
_MAX_CONTEXT_CHARS = 3500
_MAX_SOURCE_CHARS = 2500


def _cap_context(text: str, limit: int) -> str:
    """Truncate a context block to ``limit`` chars with an explicit marker."""
    if not text or len(text) <= limit:
        return text
    return text[:limit] + "\n\n...[additional context omitted to fit the model context window]"


def _is_multipart_query(prompt: str) -> bool:
    """Detect if a user prompt contains a multi-part question set."""
    # Document generation specs (e.g. "Create a PDF... 1. Section A, 2. Section B")
    # are unified deliverable specifications, not separate multi-part reasoning questions.
    from backend.core.model_manager import MockLLM
    if MockLLM._is_doc_generation_intent(prompt):
        return False

    matches = re.findall(
        r"(?:^|\n)\s*(?:(?:\d+[\.\)]|Part\s+\d+:?|Question\s+\d+:?|Q\d+:?))\s+",
        prompt,
        re.IGNORECASE,
    )
    return len(matches) >= 2


def _split_multipart_query(prompt: str) -> List[dict]:
    """Split a multi-part prompt into individual sub-questions with headers."""
    pattern = re.compile(
        r"(?:^|\n)\s*((?:(?:\d+[\.\)]|Part\s+\d+:?|Question\s+\d+:?|Q\d+:?))\s*)",
        re.IGNORECASE,
    )
    splits = pattern.split(prompt)
    parts = []
    if len(splits) >= 3:
        for i in range(1, len(splits), 2):
            marker = splits[i].strip()
            content = splits[i + 1].strip() if i + 1 < len(splits) else ""
            parts.append({"header": marker, "prompt": content})
    return parts


# Module-level model manager singleton for the graph
_model_manager: ModelManager = None


def _get_model_manager() -> ModelManager:
    global _model_manager
    from backend.core.model_manager import get_model_manager
    _model_manager = get_model_manager()
    return _model_manager


class AgentState(TypedDict, total=False):
    """State type for the agent graph."""
    input: str
    plan: list
    context: dict
    output: str
    retrieved_sources: list
    verification: dict
    role: str
    model_used: str
    deliverables: list
    selected_model: str
    retrieval_invoked: bool
    image_path: str


def plan_node(state: AgentState) -> dict:
    """
    Generate a plan from the user input.
    Runs intent classification BEFORE model selection and BEFORE retrieval.
    """
    user_input = state.get("input", "")
    if is_trivial_query(user_input):
        direct_text = get_direct_response(user_input)
        logger.info(
            f"plan_node: Deterministic intent check ({classify_intent(user_input)}) "
            "— bypassing planning, model selection, and retrieval"
        )
        return {
            "plan": [
                {
                    "tool": "direct_response",
                    "action": "direct_response",
                    "args": [direct_text],
                    "direct_response": True,
                }
            ]
        }

    logger.info(f"plan_node: Generating plan for input: {user_input[:100]}")
    plan = generate_plan(user_input, _get_model_manager())
    logger.info(f"plan_node: Plan has {len(plan)} steps")
    return {"plan": plan}


def should_skip_to_synthesize(state: AgentState) -> bool:
    """
    Check whether the execute→retrieve→verify pipeline should be bypassed.

    Bypasses deterministically when:
    - the plan carries an explicit direct-response marker, OR
    - the intent classifier (rule-based, deterministic) labels the input as
      GREETING/SMALLTALK. This runs BEFORE retrieval so trivial queries can
      never be routed through RAG, regardless of what the planner LLM produced.
    """
    plan = state.get("plan", [])
    if is_direct_response(plan):
        return True
    # Deterministic intent gate: greetings and small talk bypass retrieval.
    return is_trivial_query(state.get("input", ""))


def execute_node(state: AgentState) -> dict:
    """
    Execute each step in the plan sequentially, accumulating context.
    """
    logger.info(f"execute_node: Executing {len(state['plan'])} steps")
    # Fresh artifact tracker per request: exactly one artifact per filename.
    from backend.agents.executor import reset_artifact_tracking
    reset_artifact_tracking()
    context = dict(state.get("context", {}))

    for i, step in enumerate(state["plan"]):
        logger.info(f"execute_node: Step {i + 1}/{len(state['plan'])}")
        try:
            execute_step(step, context, _get_model_manager())
        except Exception as e:
            logger.error(f"execute_node: Step {i + 1} failed: {e}")
            context[f"step_{i}_result"] = f"Error: {e}"

    return {"context": context}


def retrieve_node(state: AgentState) -> dict:
    """
    Retrieve relevant documents from the knowledge base.
    Respects RBAC: role is passed through to HybridRAG.search().
    """
    user_input = state.get("input", "")
    from backend.core.router import is_trivial_query, should_invoke_retrieval
    if is_trivial_query(user_input) or not should_invoke_retrieval(user_input):
        logger.info(f"retrieve_node: Non-retrieval query detected ('{user_input[:60]}') — bypassing retrieval completely")
        return {
            "context": state.get("context", {}),
            "retrieved_sources": [],
            "retrieval_invoked": False,
        }

    role = state.get("role", "engineer")
    logger.info(f"retrieve_node: Searching KB for: {user_input[:100]} (role={role})")
    
    try:
        rag = get_rag()
        sources = rag.search(user_input, top_k=3, role=role)
        logger.info(f"retrieve_node: Found {len(sources)} sources")
        
        # Update context with retrieved sources
        context = dict(state.get("context", {}))
        context["retrieved_sources"] = sources
        
        return {
            "context": context,
            "retrieved_sources": sources,
            "retrieval_invoked": True,
        }
    except Exception as e:
        logger.error(f"retrieve_node: RAG search failed: {e}")
        return {
            "context": state.get("context", {}),
            "retrieved_sources": [],
            "retrieval_invoked": False,
        }


def compute_synthesis_token_budget(user_prompt: str, is_doc_gen: bool = False) -> int:
    """
    Scale synthesis token budget dynamically based on request complexity and structure.
    Allows responsive quick answers for simple queries while preventing mid-content
    truncation on detailed specs, multi-page documents, and structured deliverables.
    Upper bound is 2048 tokens.
    """
    env_override = os.getenv("MAX_SYNTHESIS_TOKENS")
    if env_override and env_override.strip() and env_override.isdigit() and int(env_override) > 128:
        return int(env_override)

    prompt_lower = user_prompt.lower()

    # Multi-page requests (e.g. 2-page, 3-page): give ample headroom up to 2048 tokens
    multi_page_match = re.search(r'\b(\d+)\s*[- ]?page\b', prompt_lower)
    if multi_page_match:
        pages = int(multi_page_match.group(1))
        # Two or more pages gets the full budget: content generation must be
        # able to finish before the PDF-rendering tool call consumes it, or the
        # response truncates mid-list and no usable document is produced.
        return min(2048, max(1536, pages * 1024))

    if any(k in prompt_lower for k in [
        "comprehensive", "detailed", "curriculum", "syllabus", "exhaustive",
        "full guide", "handbook", "syntax included", "complete reference",
        "all the syntax", "in-depth", "deep dive"
    ]):
        return 2048

    if is_doc_gen or any(k in prompt_lower for k in ["table of contents", "outline", "section 1", "chapter"]):
        return 1536

    if len(user_prompt) > 300:
        return 1024
    elif len(user_prompt) > 150:
        return 512

    return int(env_override or "256")


def _generate_doc_content(
    state: dict,
    context_text: str,
    source_context: str,
    model_manager,
    model_name: str,
    budget: int,
    is_mock_model: bool,
) -> str:
    """
    Generate the BODY CONTENT of a requested deliverable in a dedicated step,
    separate from the tool call that renders the file (PDF/DOCX/PPTX/XLSX).

    This keeps document rendering independent from chat synthesis: even if a
    small local model truncates its conversational reply mid-list, the file is
    rendered from complete, separately-generated content and the turn is only
    marked done once that file exists on disk.
    """
    from backend.core.model_manager import MockLLM

    user_input = state.get("input", "")

    if is_mock_model:
        # MockLLM deterministically composes a complete body for the request
        # (including the full tutorial text for "python basics" style specs).
        return MockLLM._compose_doc_body(user_input)

    system = (
        "You are an expert AI assistant on an air-gapped sovereign workbench. "
        "Your job is to write the full BODY CONTENT of the deliverable document the user requested. "
        "Output the actual finished content now: a title heading, sections, paragraphs, bullet lists, "
        "and code blocks where the request asks for them. "
        "Do not describe what you will do, do not present an outline or table of contents instead of "
        "the content itself, and never start a reply with phrases like 'I will provide...'. "
        "Write the complete document body — it will be rendered into the deliverable file verbatim."
    )
    user = f"Document request: {user_input}\n\nExecution results:\n{context_text}{source_context}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        raw = model_manager.generate_from_messages(model_name, messages, max_tokens=budget)
    except Exception as e:
        logger.error(f"synthesize_node: Deliverable content generation failed: {e}")
        raw = ""

    content = (raw or "").strip()

    # Placeholder guard: an empty reply, or a truncated "I will provide a
    # concise outline..." promise, is NOT document content. Fall back to the
    # deterministic composer so the render tool always has real content.
    placeholder = re.match(
        r"^\s*(?:i will|here(?:'s| is)|okay|sure|of course|let me)[\s,:]",
        content,
        re.IGNORECASE,
    )
    if not content or (placeholder and len(content) < 300):
        logger.warning(
            "synthesize_node: Model returned placeholder/empty doc content; "
            "falling back to deterministic composer"
        )
        content = MockLLM._compose_doc_body(user_input)

    return content


def synthesize_node(state: AgentState) -> dict:
    """
    Take the accumulated context and retrieved sources, generate answer,
    tag citations, and verify grounding.

    When the plan is a direct response (greeting/simple chat), the plan's
    first step contains the response text in args. We use it directly
    without calling the LLM again.
    """
    logger.info("synthesize_node: Generating final answer")

    context = state.get("context", {})
    retrieved_sources = state.get("retrieved_sources", [])
    retrieval_invoked = state.get("retrieval_invoked", False)
    plan = state.get("plan", [])
    user_input = state.get("input", "")
    from backend.core.model_manager import MockLLM
    is_doc_gen = MockLLM._is_doc_generation_intent(user_input)

    # Deterministic intent gate: greetings/small talk always produce a direct
    # template response — no model call, no retrieval, no citations.
    if is_trivial_query(user_input):
        direct_text = get_direct_response(user_input)
        logger.info("synthesize_node: Deterministic direct response (intent classifier)")
        return {
            "output": direct_text,
            "model_used": "Direct Response (Greeting)",
            "verification": {"grounded": True, "reason": "Direct conversational response"},
            "retrieval_invoked": False,
        }

    # Check if this is a direct response from MockLLM (greeting/simple chat)
    if plan and is_direct_response(plan):
        # The response text is in plan[0].args[0]
        direct_text = plan[0].get("args", [""])[0] if plan[0].get("args") else ""
        if direct_text:
            logger.info("synthesize_node: Using direct response (skipped pipeline)")
            return {
                "output": direct_text,
                "model_used": "Direct Response (Greeting)",
                "verification": {"grounded": True, "reason": "Direct conversational response"},
                "retrieval_invoked": False,
            }

    # Check context for security injection blocks from tools (Issue 3)
    # Calculator code execution attempts must never receive workaround suggestions (e.g. os.popen),
    # must log an explicit sandbox audit event to AuditLogger, and must surface a hard refusal message.
    tool_results = [v for k, v in sorted(context.items()) if k.endswith("_result") and isinstance(v, str)]
    for tr in tool_results:
        if "[SECURITY_BLOCK]" in tr:
            logger.warning("synthesize_node: Hard security block triggered in tool result")
            return {
                "output": "Execution blocked: System access and code execution attempts via the calculator tool are strictly prohibited by sandbox policy.",
                "model_used": "Security Policy (Sandbox)",
                "verification": {"grounded": True, "reason": "Security policy enforced"},
                "deliverables": [],
                "retrieval_invoked": False,
            }

    # Check context for explicit no-match tool results (Bug 5)
    # If a file search/io tool failed to find the requested file, surface explicit
    # no-match result and never fabricate a source name or confidence score.
    # Exception: if the missing file is an image file, fail hard with image_attach_failed.
    for tr in tool_results:
        if '"no_match_found"' in tr or '"status": "no_match_found"' in tr or 'Error: File not found' in tr:
            img_m = re.search(r'Error:\s*File\s+not\s+found:\s*([a-zA-Z0-9_\-\./]+\.(?:png|jpg|jpeg|webp|bmp|gif|tiff|tif))\b', tr, re.IGNORECASE)
            if img_m:
                from backend.core.model_manager import ModelInferenceError
                raise ModelInferenceError("image_attach_failed", f"Image file not found: {img_m.group(1)}")
            import json as _json
            msg = "The requested file could not be located in the sandbox."
            try:
                parsed = _json.loads(tr)
                msg = parsed.get("message", msg)
            except Exception:
                if 'Error: File not found:' in tr:
                    msg = tr
            logger.info(f"synthesize_node: Explicit no-match found in tool results: {msg}")
            return {
                "output": f"No match found. {msg}",
                "model_used": "Tool Response",
                "verification": {"grounded": True, "reason": "No match verified"},
                "deliverables": [],
                "retrieval_invoked": False,
            }

    # Explicit refusal if retrieval was invoked for a domain query and no confident sources exist (Issue 1)
    from backend.core.router import should_invoke_retrieval
    is_domain_query = should_invoke_retrieval(user_input)
    has_meaningful_results = any(
        k.endswith("_result") and len(str(v).strip()) > 0 and not str(v).startswith("Error")
        for k, v in context.items()
    )
    if retrieval_invoked and not retrieved_sources and is_domain_query and not has_meaningful_results:
        logger.info("synthesize_node: Explicit refusal — information not found in available sources")
        return {
            "output": "The requested information was not found in available sources.",
            "model_used": "Knowledge Base (Threshold Gate)",
            "verification": {"grounded": True, "reason": "Information not found in available sources"},
            "deliverables": [],
            "retrieval_invoked": True,
        }

    # Compile context into clean summary WITHOUT internal scratchpad tokens (Bug 3)
    clean_results = []
    for idx, (k, v) in enumerate(sorted((k, v) for k, v in context.items() if k.endswith("_result"))):
        clean_results.append(f"Result {idx + 1}: {v}")
    context_text = "\n".join(clean_results) if clean_results else "No results from previous steps."

    # Build source context for grounding
    source_context = ""
    if retrieved_sources:
        source_context = "\n\nRetrieved sources:\n" + "\n".join(
            f"[{i+1}] {s.get('text', '')}" for i, s in enumerate(retrieved_sources)
        )

    # Cap what goes into the model prompt. Real GGUF models run at n_ctx=2048:
    # a prompt longer than that makes llama.cpp raise "Requested tokens (...) exceed
    # context window of 2048" (e.g. a 40 KB screenshot decoded as text ≈ 31k tokens).
    # Keep the prompt comfortably inside the window so generation can use the
    # remaining budget instead of erroring.
    context_text = _cap_context(context_text, _MAX_CONTEXT_CHARS)
    source_context = _cap_context(source_context, _MAX_SOURCE_CHARS)

    model_manager = _get_model_manager()
    from backend.core.router import auto_select_model
    from backend.config import get_coder_model, _model_file_valid

    selected = state.get("selected_model")
    if selected == "mock":
        model_name = "mock"
    elif not selected or selected == "auto":
        model_name = auto_select_model(state["input"])
    elif _model_file_valid(selected):
        model_name = selected
    else:
        model_name = get_coder_model()

    # Identify whether a real GGUF model or the deterministic MockLLM is used.
    handle = model_manager.load_model(model_name, reject_oversized=False)
    is_mock_model = isinstance(handle, MockLLM)

    # ------------------------------------------------------------------
    # File-output intent: generate document content in a SEPARATE step from
    # the tool call that renders the file. Prompts that imply a deliverable
    # ("PDF", "document", output-format instructions) must never end as a
    # plain free-text chat completion — the content step below feeds a
    # forced file-rendering tool call further down.
    # ------------------------------------------------------------------
    target_fmt = None
    doc_content = None
    if is_doc_gen:
        target_fmt = MockLLM.detect_format(user_input)
        if target_fmt is not None:
            logger.info(f"synthesize_node: File-output intent detected (format={target_fmt}); "
                        "generating deliverable content in a dedicated step")
            max_content_tokens = compute_synthesis_token_budget(user_input, is_doc_gen=True)
            logger.info(f"synthesize_node: Deliverable content token budget: {max_content_tokens} tokens")
            doc_content = _generate_doc_content(
                state,
                context_text,
                source_context,
                model_manager,
                model_name,
                max_content_tokens,
                is_mock_model,
            )

    # Check for multi-part reasoning query (Issue 2)
    # Decompose into sequential sub-calls with independent timeouts (60s) and token limits (1024)
    # Preserve completed partial answers even if a later sub-question fails or times out.
    multipart_parts = _split_multipart_query(user_input) if _is_multipart_query(user_input) else []

    if multipart_parts and len(multipart_parts) >= 2:
        logger.info(f"synthesize_node: Handling multi-part query with {len(multipart_parts)} sub-calls")
        sub_timeout = int(os.getenv("MULTIPART_SUB_TIMEOUT", "60"))
        completed_parts = []

        for idx, part in enumerate(multipart_parts):
            part_header = part["header"]
            part_prompt = part["prompt"]
            logger.info(f"synthesize_node: Processing sub-part {idx + 1}/{len(multipart_parts)}: {part_prompt[:80]}")

            part_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert AI assistant on an air-gapped sovereign workbench. "
                        "Answer this specific sub-question clearly, concisely, and accurately based on the execution results and retrieved sources below. "
                        "Always cite exact numbers, facts, metrics, and thresholds from retrieved sources when answering technical questions."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Sub-question: {part_prompt}\n\nExecution results:\n{context_text}{source_context}",
                },
            ]

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        model_manager.generate_from_messages,
                        model_name,
                        part_messages,
                        max_tokens=1024,
                    )
                    part_raw = future.result(timeout=sub_timeout)
            except concurrent.futures.TimeoutError:
                logger.warning(f"synthesize_node: Sub-part {idx + 1} timed out after {sub_timeout}s")
                part_raw = f"*[Response for sub-question {idx + 1} timed out after {sub_timeout} seconds]*"
            except Exception as part_err:
                logger.error(f"synthesize_node: Sub-part {idx + 1} failed: {part_err}")
                part_raw = f"*[Response for sub-question {idx + 1} encountered an error: {part_err}]*"

            part_human = _humanize_output(part_raw, context, part_prompt)
            completed_parts.append(f"### {part_header}\n{part_human.strip()}")

        raw_output = "\n\n".join(completed_parts)
    else:
        if doc_content is not None:
            # File-output request: the deliverable body generated above IS the
            # answer content. Do NOT accept a free-text chat completion here —
            # a model talking about a PDF instead of producing one is exactly
            # the failure mode this branch prevents.
            raw_output = doc_content
            logger.info(f"synthesize_node: Using dedicated deliverable content ({len(raw_output)} chars)")
        else:
            # Multimodal image attachment
            image_ref = state.get("image_path")
            if not image_ref:
                # Search user prompt for referenced image file
                match = re.search(
                    r'\b([a-zA-Z0-9_\-\./]+\.(?:png|jpg|jpeg|webp|bmp|gif|tiff|tif))\b',
                    user_input,
                    re.IGNORECASE,
                )
                if match:
                    image_ref = match.group(1)

            if not image_ref:
                # Search context for referenced image file
                for k, v in context.items():
                    if isinstance(v, str):
                        match = re.search(
                            r'\b([a-zA-Z0-9_\-\./]+\.(?:png|jpg|jpeg|webp|bmp|gif|tiff|tif))\b',
                            v,
                            re.IGNORECASE,
                        )
                        if match:
                            image_ref = match.group(1)
                            break

            attached_image_uri = None
            if image_ref:
                import base64
                from backend.tools.file_io import _safe_resolve
                from backend.core.model_manager import ModelInferenceError
                try:
                    img_path = _safe_resolve(image_ref)
                except Exception as e:
                    raise ModelInferenceError("image_attach_failed", f"Invalid image path '{image_ref}': {e}")

                if not img_path.exists():
                    raise ModelInferenceError("image_attach_failed", f"Image file not found: {image_ref}")
                if not img_path.is_file():
                    raise ModelInferenceError("image_attach_failed", f"Image path is not a file: {image_ref}")
                if img_path.stat().st_size == 0:
                    raise ModelInferenceError("image_attach_failed", f"Image file is empty (0 bytes): {image_ref}")

                img_ext = img_path.suffix.lower()
                valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif"}
                if img_ext not in valid_exts:
                    raise ModelInferenceError("image_attach_failed", f"Unsupported image format '{img_ext}': {image_ref}")

                try:
                    with open(img_path, "rb") as f:
                        img_bytes = f.read()
                    if not img_bytes:
                        raise ModelInferenceError("image_attach_failed", f"Image file is empty: {image_ref}")

                    mime_map = {
                        ".png": "image/png",
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".webp": "image/webp",
                        ".gif": "image/gif",
                        ".bmp": "image/bmp",
                        ".tiff": "image/tiff",
                        ".tif": "image/tiff",
                    }
                    mime = mime_map.get(img_ext, "image/png")
                    b64_str = base64.b64encode(img_bytes).decode("utf-8")
                    attached_image_uri = f"data:{mime};base64,{b64_str}"

                    logger.info(
                        f"[MULTIMODAL PAYLOAD OUTGOING] Target model: {model_name}, "
                        f"Image file: {img_path.name}, MIME: {mime}, "
                        f"Data URI prefix: {attached_image_uri[:40]}..., "
                        f"Base64 length: {len(b64_str)} chars, File size: {len(img_bytes)} bytes"
                    )
                except ModelInferenceError:
                    raise
                except Exception as e:
                    raise ModelInferenceError("image_attach_failed", f"Failed to read/encode image '{image_ref}': {e}")

            if attached_image_uri:
                # Filter out binary image placeholder warnings from context so the vision
                # model is not confused by "not text — it cannot be read as text"
                clean_context_items = [
                    v for k, v in sorted(context.items())
                    if k.endswith("_result") and isinstance(v, str)
                    and "[Image file:" not in v
                    and "This is a binary image" not in v
                    and "binary image file" not in v
                    and "Ready for synthesis" not in v
                    and "No context available" not in v
                ]
                clean_context = "\n".join(f"Result {i+1}: {item}" for i, item in enumerate(clean_context_items)) if clean_context_items else ""

                query = state['input'].strip()
                if re.fullmatch(r'(?:analyze|inspect|view|read|summarize|explain|open)?\s*(?:the\s+)?(?:uploaded\s+)?(?:file\s*:?\s*)?[a-zA-Z0-9_\-\./]+\.(?:png|jpg|jpeg|webp|bmp|gif|tiff|tif)', query, re.IGNORECASE):
                    query = "Analyze this image and describe what you see in detail."
                elif image_ref:
                    query = re.sub(rf'\b(?:the\s+)?(?:uploaded\s+)?(?:file\s*:?\s*)?{re.escape(image_ref)}\b', 'this image', query, flags=re.IGNORECASE)
                    query = re.sub(r'[:,\-]?\s*this image\s*$', '', query, flags=re.IGNORECASE).strip()
                    if not query or query.lower() in {'this image', 'the image'}:
                        query = "Analyze this image and describe what you see in detail."

                prompt_parts = [query]
                if clean_context:
                    prompt_parts.append(f"Context:\n{clean_context}")
                if source_context:
                    prompt_parts.append(f"Sources:\n{source_context.strip()}")
                prompt_text = "\n\n".join(prompt_parts)

                user_content = [
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": attached_image_uri,
                        },
                    },
                ]
                # For vision multimodal inference, omit the custom text system prompt so
                # the chat handler (e.g. Llava15ChatHandler) uses its native Vicuna template
                # without conflicting instructions
                messages = [
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ]
            else:
                user_content = f"User request: {state['input']}\n\nExecution results:\n{context_text}{source_context}"
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert AI assistant on an air-gapped sovereign workbench. "
                            "Answer the user's request clearly, concisely, and accurately based on the execution results and retrieved sources below. "
                            "Always cite exact numbers, facts, metrics, and thresholds from retrieved sources when answering technical questions."
                        ),
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ]

            # Dynamically scale token budget based on request structure and complexity
            max_synthesis_tokens = compute_synthesis_token_budget(user_input, is_doc_gen=is_doc_gen)
            logger.info(f"synthesize_node: Dynamic synthesis token budget: {max_synthesis_tokens} tokens")
            raw_output = model_manager.generate_from_messages(model_name, messages, max_tokens=max_synthesis_tokens)
            logger.info(f"synthesize_node: Raw output: {raw_output[:200]}")

    # Determine which model actually generated the response
    actual_handle = model_manager.resident_models.get(model_name)
    from backend.core.model_manager import MockLLM
    if isinstance(actual_handle, MockLLM):
        model_used = "MockLLM"
    else:
        model_used = model_name

    logger.info(
        f"[MODEL_SERVING] Model serving synthesis: {model_used} "
        f"(Handle: {type(actual_handle).__name__}, Resident: {list(model_manager.resident_models.keys())})"
    )

    # If the output is raw MockLLM plan JSON, construct a user-friendly message
    output = _humanize_output(raw_output, context, user_input)

    # Tag citations ONLY when retrieval was actually invoked AND the retrieved
    # chunk demonstrably supports the claim. If no retrieval occurred,
    # no citation markup may appear under any circumstance (Bug 2).
    if retrieval_invoked and retrieved_sources:
        output = tag_citations(output, retrieved_sources)
        logger.info(f"synthesize_node: Tagged citations in output")
    else:
        # Hard boundary: strip any stray citation markup produced by the model
        output = _strip_stray_citations(output)

    # Extract deliverables from context (case-insensitive dedup)
    deliverables = []
    deliverables_lower = set()
    for k, v in sorted(context.items()):
        if k.endswith("_result") and isinstance(v, str):
            from pathlib import Path as _P
            try:
                p = _P(v)
                if p.suffix.lower() in ('.docx', '.pptx', '.xlsx', '.pdf', '.txt', '.csv') and p.is_file():
                    if p.name.lower() not in deliverables_lower:
                        deliverables_lower.add(p.name.lower())
                        deliverables.append(p.name)
            except Exception:
                pass

    # ----------------------------------------------------------------------
    # File-output completion gate. When the request implies a deliverable file
    # (PDF/DOCX/PPTX/XLSX), the orchestrator must force the render tool call
    # and only mark the task done once the tool succeeded and the file exists
    # in the output directory. Render content comes from the dedicated content
    # step (doc_content), never from truncated chat prose.
    # ----------------------------------------------------------------------
    if is_doc_gen and target_fmt is not None and doc_content is not None:
        from backend.tools.doc_generator import generate_doc, OUTPUT_DIR
        from backend.core.output_sanitizer import strip_internal_trace_tokens

        clean_body = re.sub(r'\n*Deliverable generated:.*', '', doc_content, flags=re.IGNORECASE).strip()
        clean_body = re.sub(r'\n*\[Warning:.*\]', '', clean_body).strip()
        clean_body = strip_internal_trace_tokens(clean_body, strip_reasoning=True).strip()

        if deliverables:
            # A file-producing tool already ran during execution; re-render each
            # artifact with the complete, separately-generated content so a
            # truncated model reply can never become the file body.
            for d in list(deliverables):
                ext = os.path.splitext(d)[1].lower().lstrip(".")
                doc_title = d.rsplit(".", 1)[0].replace("_", " ").title()
                try:
                    generate_doc(
                        d,
                        doc_title,
                        clean_body,
                        output_format=ext if ext in ("pdf", "docx", "pptx", "xlsx") else target_fmt,
                    )
                    logger.info(f"synthesize_node: Rendered {ext} deliverable {d} from generated content")
                except Exception as e:
                    logger.error(f"Failed to render deliverable {d}: {e}")
        else:
            # No file-producing tool step ran — force the render tool call now
            # instead of letting the turn end on free text (Bug: PDF requests).
            logger.warning(
                f"synthesize_node: File-output intent ({target_fmt}) produced no deliverable; "
                "forcing deliverable render tool call"
            )
            target_title = MockLLM._extract_doc_title(user_input)
            target_filename = MockLLM._extract_doc_filename(user_input)
            if not target_filename.lower().endswith(f".{target_fmt}"):
                if target_filename:
                    target_filename = f"{target_filename.rsplit('.', 1)[0]}.{target_fmt}"
                else:
                    slug = re.sub(r'[^a-zA-Z0-9_-]+', '_', target_title.lower()).strip('_')[:30]
                    target_filename = f"{slug or 'document'}_{uuid.uuid4().hex[:8]}.{target_fmt}"
            try:
                gen_res = generate_doc(target_filename, target_title, clean_body, output_format=target_fmt)
                from pathlib import Path as _P
                p = _P(gen_res)
                if p.is_file():
                    deliverables.append(p.name)
                    logger.info(f"synthesize_node: Forced render produced deliverable {p.name}")
            except Exception as e:
                logger.error(f"synthesize_node: Forced deliverable render failed: {e}")

        # Completion gate: only deliverables whose file actually exists on disk
        # count as done. Drop anything the render step failed to produce.
        deliverables = [d for d in deliverables if (Path(OUTPUT_DIR) / d).is_file()]

        # If the file still does not exist after forcing the tool call, the task
        # is NOT complete — surface that honestly instead of claiming a PDF.
        if not deliverables:
            err_note = (
                f"\n\n[Error] The {target_fmt.upper()} file could not be generated: "
                "the document-rendering tool did not produce an output file."
            )
            if err_note not in output:
                output += err_note
            logger.error(
                f"synthesize_node: File-output request failed to produce a {target_fmt.upper()} deliverable"
            )

    if deliverables:
        for d in deliverables:
            if d.lower() not in output.lower():
                output += f"\n\nDeliverable generated: {d}"

    # Verify grounding only when retrieval was active and returned sources
    retrieval_active = retrieval_invoked or bool(retrieved_sources)
    if retrieval_active and retrieved_sources and len(retrieved_sources) > 0:
        verifier = CitationVerifier(model_manager)
        verification = verifier.verify(output, retrieved_sources)
        logger.info(f"synthesize_node: Verification: {verification}")

        if not verification.get("grounded", False):
            output += f"\n\n[Warning: {verification.get('reason', 'Claims may not be fully grounded in sources.')}]"
    else:
        verification = {"grounded": True, "reason": "No retrieval sources to verify against."}

    # Response-formatting boundary sanitization (Bug 3)
    from backend.core.output_sanitizer import strip_internal_trace_tokens
    output = strip_internal_trace_tokens(output)

    return {
        "output": output,
        "model_used": model_used,
        "verification": verification,
        "deliverables": deliverables,
        "retrieval_invoked": retrieval_invoked,
    }


def _strip_stray_citations(text: str) -> str:
    """
    Remove citation markup like ``[Source: foo.txt]`` / ``[Source: foo.txt, Page 1]``
    from text when retrieval was NOT invoked — the model must never self-attach
    citations without retrieval-backed evidence.
    """
    if not text:
        return text
    return re.sub(
        r'\s*\[Source:\s*[^\]]+\]',
        '',
        text,
    )


def _humanize_output(raw_output: str, context: dict, user_input: str) -> str:
    """
    Detect raw plan JSON in the output and replace with a human-readable message
    referencing actual file paths produced by the executor.
    """
    # Check if output looks like MockLLM plan JSON
    if raw_output.startswith('{"mock":') or raw_output.startswith('[{"tool":'):
        # Collect file paths from executor results
        file_paths = []
        for k, v in sorted(context.items()):
            if k.endswith("_result") and isinstance(v, str) and ("/" in v or "\\" in v):
                # Looks like a file path
                from pathlib import Path as _P
                try:
                    p = _P(v)
                    if p.suffix in ('.docx', '.pptx', '.xlsx', '.pdf', '.txt', '.csv'):
                        file_paths.append(v)
                except Exception:
                    pass

        if file_paths:
            paths_str = ", ".join(file_paths)
            return f"Deliverables generated successfully: {paths_str}"

        # Check context for tool results that contain file paths
        tool_results = [v for k, v in sorted(context.items()) if k.endswith("_result") and isinstance(v, str)]
        if tool_results:
            # Use the last meaningful result
            last_result = tool_results[-1]
            # Explicit no-match results are surfaced as a clean message, never
            # as raw JSON or a fabricated success.
            if '"no_match_found"' in last_result or '"status": "no_match_found"' in last_result:
                import json as _json
                try:
                    parsed = _json.loads(last_result)
                    msg = parsed.get("message", "No matching file found.")
                    return f"No match found. {msg}"
                except Exception:
                    return "No match found. The requested file could not be located in the sandbox."
            if len(last_result) > 5 and not last_result.startswith("Error"):
                return f"Task completed. Result: {last_result}"
            # Surface genuine tool failures honestly instead of claiming success
            if last_result.startswith("Error"):
                return last_result

        return "Task completed successfully."

    return raw_output


def build_graph():
    """
    Build and compile the LangGraph state machine.

    Returns:
        A compiled graph app that can be invoked with AgentState.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("plan_node", plan_node)
    graph.add_node("execute_node", execute_node)
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("synthesize_node", synthesize_node)

    # Add edges
    graph.add_edge(START, "plan_node")

    # Conditional edge from plan_node: skip pipeline for greetings/simple chat
    graph.add_conditional_edges(
        "plan_node",
        should_skip_to_synthesize,
        {True: "synthesize_node", False: "execute_node"},
    )
    graph.add_edge("execute_node", "retrieve_node")
    graph.add_edge("retrieve_node", "synthesize_node")
    graph.add_edge("synthesize_node", END)

    # Compile
    app = graph.compile()
    logger.info("LangGraph compiled successfully")
    return app


# Compiled graph instance (created at import time)
app = build_graph()
