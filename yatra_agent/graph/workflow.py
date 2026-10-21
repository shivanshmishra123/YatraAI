"""LangGraph workflow definition for Yatra AI."""

import logging
from typing import Literal
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from yatra_agent.graph import nodes
from yatra_agent.graph.state import TravelPlanState, is_planning_ready

logger = logging.getLogger("yatra_ai.workflow")


def route_user_query(
    state: TravelPlanState,
) -> Literal["simple_query", "missing_info", "gemini_strategy", "planning_ready"]:
    """Conditional router that determines the next graph node based on extracted state."""
    query_type = state.get("query_type", "simple")

    if query_type == "simple":
        return "simple_query"

    # For full planning or parameter accumulation
    ready, missing = is_planning_ready(state)
    if not ready:
        return "missing_info"
    elif state.get("needs_deep_analysis"):
        return "gemini_strategy"
    else:
        return "planning_ready"


def route_validation_cycle(
    state: TravelPlanState,
) -> Literal["synthesizer", "react_agent_execution"]:
    """Determines whether to proceed to synthesis or re-plan based on validator feedback."""
    from yatra_agent.config.settings import MAX_VALIDATION_CYCLES

    if state.get("validation_passed", True) or state.get("iteration_count", 0) >= MAX_VALIDATION_CYCLES:
        return "synthesizer"
    return "react_agent_execution"


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
    workflow.add_node("gemini_strategy", nodes.gemini_strategy_node)
    workflow.add_node("react_agent_execution", nodes.react_agent_execution_node)
    workflow.add_node("critique_validator", nodes.critique_validator_node)
    workflow.add_node("synthesizer", nodes.synthesizer_node)

    # 2. Entry Edges
    workflow.set_entry_point("extract_info")
    workflow.add_edge("extract_info", "pre_analyze")

    # 3. Conditional Routing from pre_analyze
    workflow.add_conditional_edges(
        "pre_analyze",
        route_user_query,
        {
            "simple_query": "simple_query",
            "missing_info": "missing_info",
            "gemini_strategy": "gemini_strategy",
            "planning_ready": "planning_ready",
        },
    )

    # 4. Phase 4 Dual-Model Pipeline Edges
    workflow.add_edge("planning_ready", "react_agent_execution")
    workflow.add_edge("gemini_strategy", "react_agent_execution")
    workflow.add_edge("react_agent_execution", "critique_validator")

    # 5. Validation Loop conditional edge
    workflow.add_conditional_edges(
        "critique_validator",
        route_validation_cycle,
        {
            "synthesizer": "synthesizer",
            "react_agent_execution": "react_agent_execution",
        },
    )

    workflow.add_edge("synthesizer", END)
    workflow.add_edge("simple_query", END)
    workflow.add_edge("missing_info", END)

    # 6. Compile with in-memory checkpointer for multi-turn state persistence
    checkpointer = MemorySaver()
    compiled_app = workflow.compile(checkpointer=checkpointer)
    return compiled_app
