"""TravelPlanState TypedDict definition for LangGraph."""

from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict


def update_messages(existing: List[Any], new_messages: List[Any]) -> List[Any]:
    """Helper reducer to append messages."""
    return existing + new_messages


class TravelPlanState(TypedDict, total=False):
    """Complete state representation for a travel planning conversation in Yatra AI."""

    # Conversation history
    messages: List[Dict[str, Any]]

    # Extracted travel parameters
    origin: str
    destinations: List[str]
    departure_date: str
    return_date: str
    num_days: int
    budget: float
    budget_currency: str
    num_travelers: int
    traveler_details: str
    preferences: List[str]
    optimization_goal: str
    travel_mode_pref: str

    # Workflow routing & analysis
    query_type: str                 # "simple" | "full_planning"
    needs_deep_analysis: bool
    complexity_reason: str

    # Real-time tool results (via MCP servers)
    train_data: Dict[str, Any]
    driving_data: Dict[str, Any]
    hotel_data: Dict[str, Any]
    weather_data: Dict[str, Any]
    flight_data: Dict[str, Any]
    bus_data: Dict[str, Any]
    rag_results: List[Any]

    # Gemini deep planning & optimization
    query_plan: List[Dict[str, Any]]
    gemini_analysis: str

    # Constraint validation loop
    validation_passed: bool
    validation_feedback: str
    iteration_count: int

    # Final output
    final_plan: str
