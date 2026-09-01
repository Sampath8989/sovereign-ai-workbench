"""
LangGraph State Machine: Orchestrates the ReWOO agent pipeline.
Pipeline: START → plan_node → execute_node → retrieve_node → synthesize_node → END
"""

import logging
from typing import Annotated, List, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.core.model_manager import ModelManager
from backend.agents.planner import generate_plan, is_direct_response
from backend.agents.executor import execute_step
from backend.tools.rag_search import get_rag
from backend.tools.citation_tagger import tag_citations
from backend.agents.verifier import CitationVerifier

logger = logging.getLogger(__name__)

# Module-level model manager singleton for the graph
_model_manager: ModelManager = None


def _get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


class AgentState(TypedDict):
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


def plan_node(state: AgentState) -> dict:
    """
    Generate a plan from the user input.
    """
    logger.info(f"plan_node: Generating plan for input: {state['input'][:100]}")
    plan = generate_plan(state["input"], _get_model_manager())
    logger.info(f"plan_node: Plan has {len(plan)} steps")
    return {"plan": plan}


def should_skip_to_synthesize(state: AgentState) -> bool:
    """Check if the plan is a direct response (greeting/simple chat) that
    should skip the execute→retrieve→verify pipeline."""
    plan = state.get("plan", [])
    return is_direct_response(plan)


def execute_node(state: AgentState) -> dict:
    """
    Execute each step in the plan sequentially, accumulating context.
    """
    logger.info(f"execute_node: Executing {len(state['plan'])} steps")
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
    role = state.get("role", "engineer")
    logger.info(f"retrieve_node: Searching KB for: {state['input'][:100]} (role={role})")
    
    try:
        rag = get_rag()
        sources = rag.search(state["input"], top_k=3, role=role)
        logger.info(f"retrieve_node: Found {len(sources)} sources")
        
        # Update context with retrieved sources
        context = dict(state.get("context", {}))
        context["retrieved_sources"] = sources
        
        return {"context": context, "retrieved_sources": sources}
    except Exception as e:
        logger.error(f"retrieve_node: RAG search failed: {e}")
        return {"context": state.get("context", {}), "retrieved_sources": []}


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
    plan = state.get("plan", [])

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
            }

    # Compile context into a readable summary
    context_text = "\n".join(
        f"Step {k}: {v}" for k, v in sorted(context.items()) if k.endswith("_result")
    )

    if not context_text.strip():
        context_text = "No results from previous steps."

    # Build source context for grounding
    source_context = ""
    if retrieved_sources:
        source_context = "\n\nRetrieved sources:\n" + "\n".join(
            f"[{i+1}] {s.get('text', '')}" for i, s in enumerate(retrieved_sources)
        )

    model_manager = _get_model_manager()
    from backend.config import get_coder_model
    model_name = get_coder_model()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Based on the execution context below, "
                "provide a clear, concise answer to the user's original request. "
                "Use the retrieved sources to ground your answer when available."
            ),
        },
        {
            "role": "user",
            "content": f"User request: {state['input']}\n\nExecution results:\n{context_text}{source_context}",
        },
    ]

    raw_output = model_manager.generate_from_messages(model_name, messages)
    logger.info(f"synthesize_node: Raw output: {raw_output[:200]}")

    # Determine which model actually generated the response
    actual_handle = model_manager.resident_models.get(model_name)
    from backend.core.model_manager import MockLLM
    if isinstance(actual_handle, MockLLM):
        model_used = "MockLLM"
    else:
        model_used = model_name

    # If the output is raw MockLLM plan JSON, construct a user-friendly message
    output = _humanize_output(raw_output, context, state.get("input", ""))

    # Tag citations
    if retrieved_sources:
        output = tag_citations(output, retrieved_sources)
        logger.info(f"synthesize_node: Tagged citations in output")

    # Extract deliverables from context
    deliverables = []
    for k, v in sorted(context.items()):
        if k.endswith("_result") and isinstance(v, str):
            from pathlib import Path as _P
            try:
                p = _P(v)
                if p.suffix.lower() in ('.docx', '.pptx', '.xlsx', '.pdf', '.txt', '.csv'):
                    if p.name not in deliverables:
                        deliverables.append(p.name)
            except Exception:
                pass

    if deliverables:
        for d in deliverables:
            if d not in output:
                output += f"\n\nDeliverable generated: {d}"

    # Verify grounding
    verifier = CitationVerifier(model_manager)
    verification = verifier.verify(output, retrieved_sources)
    logger.info(f"synthesize_node: Verification: {verification}")

    if not verification.get("grounded", False):
        output += f"\n\n[Warning: {verification.get('reason', 'Claims may not be fully grounded in sources.')}]"

    return {"output": output, "model_used": model_used, "verification": verification, "deliverables": deliverables}


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
            if len(last_result) > 5 and not last_result.startswith("Error"):
                return f"Task completed. Result: {last_result}"

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
