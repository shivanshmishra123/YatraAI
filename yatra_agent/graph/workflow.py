"""LangGraph workflow definition for Yatra AI."""

import logging
from typing import Literal
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from yatra_agent.graph import nodes
from yatra_agent.graph.state import TravelPlanState, is_planning_ready

logger = logging.getLogger("yatra_ai.workflow")


def route_user_query(state: TravelPlanState) -> Literal["simple_query", "missing_info", "planning_ready"]:
    """Conditional router that determines the next graph node based on extracted state."""
    query_type = state.get("query_type", "simple")

    if query_type == "simple":
        return "simple_query"

    # For full planning or parameter accumulation
    ready, missing = is_planning_ready(state)
    if not ready:
        return "missing_info"
    else:
        return "planning_ready"


def missing_info_adapter_node(state: TravelPlanState) -> TravelPlanState:
    """Node adapter that determines missing fields and generates follow-up prompt."""
    _, missing = is_planning_ready(state)
    return nodes.missing_info_node(state, missing)


def build_travel_workflow():
    """Builds and compiles the Yatra AI StateGraph workflow with memory checkpointing."""
    workflow = StateGraph(TravelPlanState)

    # 1. Register Nodes
    workflow.add_node("extract_info", nodes.extract_info_node)
    workflow.add_node("pre_analyze", nodes.pre_analyze_node)
    workflow.add_node("simple_query", nodes.simple_query_node)
    workflow.add_node("missing_info", missing_info_adapter_node)
    workflow.add_node("planning_ready", nodes.planning_ready_node)

    # 2. Sequential Edges
    workflow.set_entry_point("extract_info")
    workflow.add_edge("extract_info", "pre_analyze")

    # 3. Conditional Routing
    workflow.add_conditional_edges(
        "pre_analyze",
        route_user_query,
        {
            "simple_query": "simple_query",
            "missing_info": "missing_info",
            "planning_ready": "planning_ready",
        },
    )

    # 4. Terminal Edges for Phase 1
    workflow.add_edge("simple_query", END)
    workflow.add_edge("missing_info", END)
    workflow.add_edge("planning_ready", END)

    # 5. Compile with in-memory checkpointer for multi-turn state persistence
    checkpointer = MemorySaver()
    compiled_app = workflow.compile(checkpointer=checkpointer)
    return compiled_app
