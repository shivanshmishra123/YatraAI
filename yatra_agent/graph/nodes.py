"""Processing nodes for Yatra AI LangGraph workflow."""

from yatra_agent.graph.state import TravelPlanState


def extract_info_node(state: TravelPlanState) -> TravelPlanState:
    """Extract structured travel parameters from user input (Phase 1)."""
    return state


def pre_analyze_node(state: TravelPlanState) -> TravelPlanState:
    """Classify complexity and route query (Phase 1/4)."""
    return state


def simple_query_node(state: TravelPlanState) -> TravelPlanState:
    """Handle general travel destination questions (Phase 1)."""
    return state


def gemini_strategy_node(state: TravelPlanState) -> TravelPlanState:
    """Perform complex route and budget decomposition (Phase 4)."""
    return state


def react_agent_execution_node(state: TravelPlanState) -> TravelPlanState:
    """Execute LangChain ReAct agent calling MCP tools (Phase 3)."""
    return state


def critique_validator_node(state: TravelPlanState) -> TravelPlanState:
    """Validate constraints and budget feasibility (Phase 4)."""
    return state


def synthesizer_node(state: TravelPlanState) -> TravelPlanState:
    """Synthesize complete markdown itinerary (Phase 1/4)."""
    return state
