"""TravelPlanState TypedDict definition and state management helpers for Yatra AI."""

from typing import Annotated, Any, Dict, List, Optional, Tuple
from typing_extensions import TypedDict


def create_initial_state() -> "TravelPlanState":
    """Create an empty, initialized TravelPlanState dictionary."""
    return {
        "messages": [],
        "origin": "",
        "destinations": [],
        "departure_date": "",
        "return_date": "",
        "num_days": 0,
        "budget": 0.0,
        "budget_currency": "INR",
        "num_travelers": 1,
        "traveler_details": "",
        "preferences": [],
        "optimization_goal": "balanced",
        "travel_mode_pref": "",
        "query_type": "simple",
        "needs_deep_analysis": False,
        "complexity_reason": "",
        "train_data": {},
        "driving_data": {},
        "hotel_data": {},
        "weather_data": {},
        "flight_data": {},
        "bus_data": {},
        "rag_results": [],
        "query_plan": [],
        "gemini_analysis": "",
        "validation_passed": True,
        "validation_feedback": "",
        "iteration_count": 0,
        "final_plan": "",
    }


def is_planning_ready(state: "TravelPlanState") -> Tuple[bool, List[str]]:
    """Check if all essential fields are collected to trigger the full itinerary planning pipeline.
    
    Returns:
        (ready: bool, missing_fields: list[str])
    """
    missing = []
    if not state.get("origin"):
        missing.append("origin")
    if not state.get("destinations") or len(state["destinations"]) == 0:
        missing.append("destinations")
    if not state.get("departure_date") and (not state.get("num_days") or state.get("num_days") <= 0):
        missing.append("departure_date")
    
    return len(missing) == 0, missing


def merge_state_updates(state: "TravelPlanState", updates: Dict[str, Any]) -> "TravelPlanState":
    """Merge extracted updates into an existing travel state without overwriting non-empty fields with empties."""
    updated: TravelPlanState = dict(state)  # type: ignore
    for key, val in updates.items():
        if val is None or val == "" or val == [] or val == {} or val == 0:
            continue
        # For destinations, append or merge uniquely if list
        if key == "destinations" and isinstance(val, list):
            existing_dests = [d.lower() for d in updated.get("destinations", [])]
            for d in val:
                if d and d.lower() not in existing_dests:
                    updated.setdefault("destinations", []).append(d)
        elif key == "preferences" and isinstance(val, list):
            existing_prefs = [p.lower() for p in updated.get("preferences", [])]
            for p in val:
                if p and p.lower() not in existing_prefs:
                    updated.setdefault("preferences", []).append(p)
        else:
            updated[key] = val  # type: ignore
            
    return updated


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
