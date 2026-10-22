"""Phase 4 unit and integration tests: Dual-model routing, Gemini reasoning, and multi-city circuits."""

import pytest
from yatra_agent.graph.state import create_initial_state
from yatra_agent.graph.nodes import (
    extract_info_node,
    pre_analyze_node,
    gemini_strategy_node,
    critique_validator_node,
    synthesizer_node,
)
from yatra_agent.graph.workflow import build_travel_workflow, route_user_query, route_validation_cycle
from yatra_agent.tools.gemini_tool import (
    GeminiReasoningTool,
    call_gemini_for_strategy,
    call_gemini_for_optimization,
)


# ---------------------------------------------------------------------------
# 1. Pre-Analysis Classification Tests (10+ Scenarios)
# ---------------------------------------------------------------------------
def test_pre_analysis_multi_destination():
    """Multi-destination routes must trigger Gemini deep analysis."""
    state = create_initial_state()
    state["destinations"] = ["Agra", "Jaipur"]
    analyzed = pre_analyze_node(state)
    assert analyzed["needs_deep_analysis"] is True
    assert "multi_destination_route" in analyzed["complexity_reason"]


def test_pre_analysis_tight_budget():
    """Budget below ₹500/person/day must trigger tight_budget deep analysis."""
    state = create_initial_state()
    state["destinations"] = ["Goa"]
    state["num_days"] = 5
    state["num_travelers"] = 1
    state["budget"] = 2000.0  # ₹400/day < ₹500
    analyzed = pre_analyze_node(state)
    assert analyzed["needs_deep_analysis"] is True
    assert "tight_budget" in analyzed["complexity_reason"]


def test_pre_analysis_standard_budget():
    """Generous budget for single city does not trigger deep analysis."""
    state = create_initial_state()
    state["destinations"] = ["Jaipur"]
    state["num_days"] = 3
    state["num_travelers"] = 1
    state["budget"] = 9000.0  # ₹3000/day > ₹500
    analyzed = pre_analyze_node(state)
    assert analyzed["needs_deep_analysis"] is False
    assert analyzed["complexity_reason"] == "standard_route"


def test_pre_analysis_elderly_constraints():
    """Elderly or senior traveler details must trigger special constraints."""
    state = create_initial_state()
    state["destinations"] = ["Haridwar"]
    state["traveler_details"] = "traveling with elderly parents"
    analyzed = pre_analyze_node(state)
    assert analyzed["needs_deep_analysis"] is True
    assert "special_constraints" in analyzed["complexity_reason"]


def test_pre_analysis_wheelchair_accessibility():
    """Wheelchair / disability constraints trigger deep analysis."""
    state = create_initial_state()
    state["destinations"] = ["Udaipur"]
    state["preferences"] = ["wheelchair accessible"]
    analyzed = pre_analyze_node(state)
    assert analyzed["needs_deep_analysis"] is True
    assert "special_constraints" in analyzed["complexity_reason"]


def test_pre_analysis_infant_toddler():
    """Traveling with toddler or infant triggers deep analysis."""
    state = create_initial_state()
    state["destinations"] = ["Goa"]
    state["messages"] = [{"role": "user", "content": "Trip to Goa with a toddler"}]
    analyzed = pre_analyze_node(state)
    assert analyzed["needs_deep_analysis"] is True
    assert "special_constraints" in analyzed["complexity_reason"]


def test_pre_analysis_optimization_cheapest():
    """Cheapest goal triggers optimization requirement."""
    state = create_initial_state()
    state["destinations"] = ["Varanasi"]
    state["optimization_goal"] = "cheapest"
    analyzed = pre_analyze_node(state)
    assert analyzed["needs_deep_analysis"] is True
    assert "optimization_required" in analyzed["complexity_reason"]


def test_pre_analysis_optimization_fastest():
    """Fastest goal triggers optimization requirement."""
    state = create_initial_state()
    state["destinations"] = ["Chennai"]
    state["optimization_goal"] = "fastest"
    analyzed = pre_analyze_node(state)
    assert analyzed["needs_deep_analysis"] is True
    assert "optimization_required" in analyzed["complexity_reason"]


def test_pre_analysis_simple_question_no_deep():
    """Simple destination question should not trigger deep analysis."""
    state = create_initial_state()
    state["query_type"] = "simple"
    state["messages"] = [{"role": "user", "content": "What are the best places in Jaipur?"}]
    analyzed = pre_analyze_node(state)
    assert analyzed["needs_deep_analysis"] is False


def test_pre_analysis_standard_planning_no_deep():
    """Standard single-city trip with normal budget does not trigger deep analysis."""
    state = create_initial_state()
    state["origin"] = "Delhi"
    state["destinations"] = ["Jaipur"]
    state["num_days"] = 3
    state["budget"] = 8000.0
    analyzed = pre_analyze_node(state)
    assert analyzed["needs_deep_analysis"] is False


# ---------------------------------------------------------------------------
# 2. Gemini Reasoning Tool Unit Tests
# ---------------------------------------------------------------------------
def test_gemini_strategy_generation():
    """Verify GeminiReasoningTool generates structured query plan and execution plan."""
    tool = GeminiReasoningTool()
    state = create_initial_state()
    state["origin"] = "Delhi"
    state["destinations"] = ["Agra", "Jaipur"]
    state["num_days"] = 5
    state["budget"] = 15000.0
    state["num_travelers"] = 2
    state["traveler_details"] = "elderly parents"

    strategy = tool.generate_strategy(state)

    assert bool(strategy)
    assert any(k in strategy for k in ("optimization_strategy", "execution_plan", "itinerary_name", "query_plan"))


def test_gemini_optimization_commentary():
    """Verify Gemini commentary addresses route minimization and accessibility constraints."""
    state = create_initial_state()
    state["origin"] = "Delhi"
    state["destinations"] = ["Agra", "Jaipur"]
    state["budget"] = 15000.0
    state["num_days"] = 5
    state["traveler_details"] = "senior citizens"
    state["complexity_reason"] = "multi_destination_route, special_constraints"

    commentary = call_gemini_for_optimization(state)
    assert "Gemini" in commentary
    assert "Delhi" in commentary
    assert "Elderly / Accessibility Care" in commentary


def test_gemini_optimization_commentary():
    """Verify Gemini commentary addresses route minimization and accessibility constraints."""
    state = create_initial_state()
    state["origin"] = "Delhi"
    state["destinations"] = ["Agra", "Jaipur"]
    state["budget"] = 15000.0
    state["num_days"] = 5
    state["traveler_details"] = "senior citizens"
    state["complexity_reason"] = "multi_destination_route, special_constraints"

    commentary = call_gemini_for_optimization(state)
    assert "Gemini 1.5 Pro Strategic Optimization Commentary" in commentary
    assert "Delhi" in commentary
    assert "Elderly / Accessibility Care" in commentary


# ---------------------------------------------------------------------------
# 3. Critique Validator Node Guardrail Tests
# ---------------------------------------------------------------------------
def test_critique_validator_cycle_guardrail():
    """Validator allows maximum MAX_VALIDATION_CYCLES before forcing pass."""
    state = create_initial_state()
    state["train_data"] = {}
    state["hotel_data"] = {}
    state["iteration_count"] = 0

    # Cycle 1: missing data causes failure
    v1 = critique_validator_node(state)
    assert v1["iteration_count"] == 1
    assert v1["validation_passed"] is False

    # Cycle 2: reaching max cycles triggers completion guardrail
    v2 = critique_validator_node(v1)
    assert v2["iteration_count"] == 2
    assert v2["validation_passed"] is True
    assert "Validation passed on cycle 2" in v2["validation_feedback"]


def test_route_validation_cycle_router():
    """Router directs to synthesizer if valid, or react_agent_execution if replan needed."""
    state = create_initial_state()
    state["validation_passed"] = False
    state["iteration_count"] = 1
    assert route_validation_cycle(state) == "react_agent_execution"

    state["validation_passed"] = True
    assert route_validation_cycle(state) == "synthesizer"


# ---------------------------------------------------------------------------
# 4. Multi-Destination End-to-End Workflow Integration Test
# ---------------------------------------------------------------------------
def test_multi_city_complex_workflow_execution():
    """Verify full dual-model LangGraph pipeline:
    NLU -> pre-analysis -> Gemini strategy -> multi-segment execution -> validation -> synthesis.
    """
    workflow = build_travel_workflow()
    config = {"configurable": {"thread_id": "phase4_multi_city_test"}}

    state = create_initial_state()
    state["messages"] = [
        {"role": "user", "content": "Delhi to Agra and Jaipur, 5 days, ₹15000 for elderly parents"}
    ]

    result = workflow.invoke(state, config=config)

    # 1. Pre-analysis & Complexity Check
    assert result["needs_deep_analysis"] is True
    assert "multi_destination_route" in result["complexity_reason"]
    assert "special_constraints" in result["complexity_reason"]

    # 2. Re-ordered & Decomposed Multi-City Circuit
    assert result["origin"] == "Delhi"
    assert "Agra" in result["destinations"]
    assert "Jaipur" in result["destinations"]
    assert len(result.get("query_plan", [])) >= 4

    # 3. Tool results collected for all legs
    assert "trains" in result.get("train_data", {})
    assert len(result["train_data"]["trains"]) >= 1

    # Total distance combines legs: Delhi->Agra (240km) + Agra->Jaipur (240km) + Jaipur->Delhi (280km) = 760km
    assert result.get("driving_data", {}).get("distance_km", 0) >= 700

    assert "hotels" in result.get("hotel_data", {})
    assert len(result["hotel_data"]["hotels"]) >= 1

    # 4. Critique validator passed
    assert result.get("validation_passed") is True

    # 5. Rich Synthesized Output
    final_plan = result.get("final_plan", "")
    assert "Complete Itinerary" in final_plan
    assert "Gemini 1.5 Pro" in final_plan
    assert "Transport Modes Comparison" in final_plan
    assert "Itemized Budget Breakdown" in final_plan
    assert "Day 1" in final_plan
    assert "Day 5" in final_plan
    assert "₹" in final_plan
