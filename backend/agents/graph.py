"""
LangGraph State Machine: Orchestrates the ReWOO agent pipeline.
Pipeline: START → plan_node → execute_node → synthesize_node → END
"""

import logging
from typing import Annotated, List, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.core.model_manager import ModelManager
from backend.agents.planner import generate_plan
from backend.agents.executor import execute_step

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


def plan_node(state: AgentState) -> dict:
    """
    Generate a plan from the user input.
    """
    logger.info(f"plan_node: Generating plan for input: {state['input'][:100]}")
    plan = generate_plan(state["input"], _get_model_manager())
    logger.info(f"plan_node: Plan has {len(plan)} steps")
    return {"plan": plan}


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


def synthesize_node(state: AgentState) -> dict:
    """
    Take the accumulated context and produce a natural language answer.
    """
    logger.info("synthesize_node: Generating final answer")

    context = state.get("context", {})

    # Compile context into a readable summary
    context_text = "\n".join(
        f"Step {k}: {v}" for k, v in sorted(context.items()) if k.endswith("_result")
    )

    if not context_text.strip():
        context_text = "No results from previous steps."

    model_manager = _get_model_manager()
    from backend.config import get_coder_model
    model_name = get_coder_model()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Based on the execution context below, "
                "provide a clear, concise answer to the user's original request."
            ),
        },
        {
            "role": "user",
            "content": f"User request: {state['input']}\n\nExecution results:\n{context_text}",
        },
    ]

    output = model_manager.generate_from_messages(model_name, messages)
    logger.info(f"synthesize_node: Output: {output[:200]}")
    return {"output": output}


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
    graph.add_node("synthesize_node", synthesize_node)

    # Add edges
    graph.add_edge(START, "plan_node")
    graph.add_edge("plan_node", "execute_node")
    graph.add_edge("execute_node", "synthesize_node")
    graph.add_edge("synthesize_node", END)

    # Compile
    app = graph.compile()
    logger.info("LangGraph compiled successfully")
    return app


# Compiled graph instance (created at import time)
app = build_graph()
