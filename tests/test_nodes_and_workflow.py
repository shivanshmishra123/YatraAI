"""Integration tests for Phase 1 nodes and compiled LangGraph workflow."""

import pytest
from yatra_agent.graph.state import create_initial_state
from yatra_agent.graph.nodes import extract_info_node, pre_analyze_node
from yatra_agent.graph.workflow import build_travel_workflow


def test_extraction_node_single_turn():
    """Verify extract_info_node parses origin, destination, days, and budget from text."""
    state = create_initial_state()
    state["messages"] = [
        {"role": "user", "content": "Plan a trip from Delhi to Jaipur for 3 days with budget 8000"}
    ]

    updated = extract_info_node(state)
    assert updated["origin"] == "Delhi"
    assert "Jaipur" in updated["destinations"]
    assert updated["num_days"] == 3
    assert updated["budget"] == 8000.0


def test_extraction_node_hindi_input():
    """Verify extract_info_node parses Hindi input."""
    state = create_initial_state()
    state["messages"] = [
        {"role": "user", "content": "kal Dilli se Jaipur jaana hai"}
    ]

    updated = extract_info_node(state)
    assert updated["origin"] == "Delhi"
    assert "Jaipur" in updated["destinations"]
    assert updated["departure_date"] != ""


def test_pre_analyze_complexity():
    """Verify pre-analysis classifies multi-city or tight budget."""
    state = create_initial_state()
    state["destinations"] = ["Delhi", "Agra", "Jaipur"]
    analyzed = pre_analyze_node(state)
    assert analyzed["needs_deep_analysis"] is True
    assert "multi_destination_route" in analyzed["complexity_reason"]


def test_langgraph_workflow_multi_turn():
    """Verify compiled LangGraph StateGraph executes across multiple turns."""
    workflow = build_travel_workflow()
    config = {"configurable": {"thread_id": "test_session_1"}}

    state = create_initial_state()

    # Turn 1: User provides destination only
    state["messages"] = [{"role": "user", "content": "I want to visit Jaipur"}]
    res1 = workflow.invoke(state, config=config)

    assert "Jaipur" in res1.get("destinations", [])
    # Should ask for missing fields (origin, dates)
    assistant_msgs1 = [m for m in res1["messages"] if m["role"] == "assistant"]
    assert len(assistant_msgs1) >= 1

    # Turn 2: User adds origin and days
    res1["messages"].append({"role": "user", "content": "Starting from Delhi for 3 days"})
    res2 = workflow.invoke(res1, config=config)

    assert res2.get("origin") == "Delhi"
    assert "Jaipur" in res2.get("destinations", [])
    assert res2.get("num_days") == 3
    # Now all required info is collected, assistant should confirm readiness
    assistant_msgs2 = [m for m in res2["messages"] if m["role"] == "assistant"]
    assert len(assistant_msgs2) >= 2
    assert "Collected" in assistant_msgs2[-1]["content"] or "Jaipur" in assistant_msgs2[-1]["content"]
