"""Unit tests for TravelPlanState helpers and accumulation."""

import pytest
from yatra_agent.graph.state import (
    create_initial_state,
    is_planning_ready,
    merge_state_updates,
)


def test_initial_state_defaults():
    """Verify initial state structure."""
    state = create_initial_state()
    assert state["origin"] == ""
    assert state["destinations"] == []
    assert state["budget_currency"] == "INR"
    assert state["num_travelers"] == 1
    assert state["validation_passed"] is True


def test_is_planning_ready_evaluation():
    """Verify ready check flags missing fields."""
    state = create_initial_state()
    ready, missing = is_planning_ready(state)
    assert ready is False
    assert "origin" in missing
    assert "destinations" in missing
    assert "departure_date" in missing

    # Add origin and destination
    state["origin"] = "Delhi"
    state["destinations"] = ["Jaipur"]
    ready, missing = is_planning_ready(state)
    assert ready is False
    assert "departure_date" in missing

    # Add departure date or num_days
    state["num_days"] = 3
    ready, missing = is_planning_ready(state)
    assert ready is True
    assert len(missing) == 0


def test_merge_state_updates_non_destructive():
    """Verify merging state updates preserves previous fields and prevents overwriting with blanks."""
    state = create_initial_state()
    state["origin"] = "Delhi"
    state["num_travelers"] = 2

    updates = {
        "destinations": ["Jaipur"],
        "budget": 8000.0,
        "origin": "",  # Empty shouldn't overwrite existing
    }

    updated = merge_state_updates(state, updates)
    assert updated["origin"] == "Delhi"
    assert updated["destinations"] == ["Jaipur"]
    assert updated["budget"] == 8000.0
    assert updated["num_travelers"] == 2
