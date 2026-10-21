"""Integration tests for Phase 2 RAG and Phase 3 full planning workflow."""

import pytest
from yatra_agent.graph.state import create_initial_state
from yatra_agent.graph.workflow import build_travel_workflow


def test_full_planning_workflow_pipeline():
    """Verify end-to-end trip planning: NLU extraction -> routing -> tool execution -> synthesis."""
    workflow = build_travel_workflow()
    config = {"configurable": {"thread_id": "phase3_test_session"}}

    state = create_initial_state()
    state["messages"] = [
        {"role": "user", "content": "Plan Delhi to Jaipur, 3 days, ₹8000, starting Dec 15"}
    ]

    result = workflow.invoke(state, config=config)

    # 1. State parameters verified
    assert result["origin"] == "Delhi"
    assert "Jaipur" in result["destinations"]
    assert result["num_days"] == 3
    assert result["budget"] == 8000.0

    # 2. Phase 3 MCP tool data verified in state
    assert "trains" in result.get("train_data", {})
    assert len(result["train_data"]["trains"]) >= 1

    assert result.get("driving_data", {}).get("distance_km") == 280
    assert result.get("driving_data", {}).get("estimated_fuel_cost_inr", {}).get("petrol", 0) > 0

    assert "hotels" in result.get("hotel_data", {})
    assert len(result["hotel_data"]["hotels"]) >= 1

    assert "temperature_range" in result.get("weather_data", {})

    # 3. Final synthesized plan generated with INR pricing
    assert result.get("final_plan") != ""
    assert "Complete Itinerary" in result["final_plan"]
    assert "IRCTC" in result["final_plan"]
    assert "₹" in result["final_plan"]
    assert "Day 1" in result["final_plan"]
    assert "Day 2" in result["final_plan"]
    assert "Day 3" in result["final_plan"]


def test_simple_query_with_rag_enrichment():
    """Verify simple destination query searches ChromaDB RAG and cites knowledge base."""
    workflow = build_travel_workflow()
    config = {"configurable": {"thread_id": "phase2_rag_session"}}

    state = create_initial_state()
    state["messages"] = [
        {"role": "user", "content": "What are the best places in Jaipur?"}
    ]

    result = workflow.invoke(state, config=config)

    # Assistant response should be generated
    assistant_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
    assert len(assistant_msgs) >= 1
    reply = assistant_msgs[-1]["content"]

    # Should contain rich knowledge points and attribution
    assert "Jaipur" in reply or "Amber" in reply or "Hawa Mahal" in reply
    assert "Based on our travel guide" in reply or "Source:" in reply or "Knowledge Base" in reply
