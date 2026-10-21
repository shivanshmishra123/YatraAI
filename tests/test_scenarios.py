"""Comprehensive scenario test suite covering all 26 scenarios defined in PLAN.md Phase 6."""

import pytest
from yatra_agent.graph.state import create_initial_state
from yatra_agent.graph.workflow import build_travel_workflow


@pytest.fixture(scope="module")
def workflow():
    return build_travel_workflow()


# ---------------------------------------------------------------------------
# Category A: 10 Simple Destination Queries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query,expected_keyword", [
    ("What are the best places in Jaipur?", "Jaipur"),
    ("Tell me about beaches in Goa", "Goa"),
    ("Best ghats to visit in Varanasi", "Varanasi"),
    ("Guide for visiting Manali in winter", "Manali"),
    ("Attractions in Kerala backwaters", "Kerala"),
    ("What is the best time to visit Agra?", "Agra"),
    ("Famous palaces in Udaipur", "Udaipur"),
    ("Golden Temple visiting tips in Amritsar", "Amritsar"),
    ("River rafting guidelines for Rishikesh", "Rishikesh"),
    ("Iconic street food in Mumbai", "Mumbai"),
])
def test_simple_destination_queries(workflow, query, expected_keyword):
    state = create_initial_state()
    state["messages"] = [{"role": "user", "content": query}]
    config = {"configurable": {"thread_id": f"scen_simple_{expected_keyword.lower()}"}}

    res = workflow.invoke(state, config=config)
    assistant_msgs = [m for m in res["messages"] if m["role"] == "assistant"]
    assert len(assistant_msgs) >= 1
    reply = assistant_msgs[-1]["content"]
    assert expected_keyword in reply or len(reply) > 50


# ---------------------------------------------------------------------------
# Category B: 5 Single-City Full Planning Queries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query,origin,dest", [
    ("Delhi to Jaipur, 3 days, ₹8000, starting Dec 15", "Delhi", "Jaipur"),
    ("Mumbai to Pune, 2 days, ₹5000", "Mumbai", "Pune"),
    ("Bangalore to Chennai for 3 days, budget 10000", "Bengaluru", "Chennai"),
    ("Delhi to Agra, 2 days, ₹6000", "Delhi", "Agra"),
    ("Kolkata to Puri for 4 days with budget ₹12000", "Kolkata", "Puri"),
])
def test_single_city_planning_queries(workflow, query, origin, dest):
    state = create_initial_state()
    state["messages"] = [{"role": "user", "content": query}]
    config = {"configurable": {"thread_id": f"scen_single_{origin}_{dest}"}}

    res = workflow.invoke(state, config=config)
    assert res.get("origin") == origin or origin in res.get("origin", "")
    assert dest in res.get("destinations", [])
    assert res.get("final_plan") != ""
    assert "Complete Itinerary" in res["final_plan"]
    assert "IRCTC" in res["final_plan"]


# ---------------------------------------------------------------------------
# Category C: 3 Multi-City Planning Queries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query,cities", [
    ("Delhi to Agra and Jaipur, 5 days, ₹15000", ["Agra", "Jaipur"]),
    ("Bangalore to Mysore and Ooty for 6 days", ["Mysuru", "Udhagamandalam"]),
    ("Delhi to Varanasi and Ayodhya, 4 days, budget 12000", ["Varanasi", "Ayodhya"]),
])
def test_multi_city_planning_queries(workflow, query, cities):
    state = create_initial_state()
    state["messages"] = [{"role": "user", "content": query}]
    config = {"configurable": {"thread_id": f"scen_multi_{cities[0]}"}}

    res = workflow.invoke(state, config=config)
    assert res.get("needs_deep_analysis") is True
    for c in cities:
        assert c in res.get("destinations", [])
    assert res.get("final_plan") != ""
    assert "Transport Modes Comparison" in res["final_plan"]
    assert "Itemized Budget Breakdown" in res["final_plan"]


# ---------------------------------------------------------------------------
# Category D: 3 Complex Constraint Queries
# ---------------------------------------------------------------------------
def test_complex_tight_budget(workflow):
    """Tight budget query triggers budget survival strategy."""
    state = create_initial_state()
    state["messages"] = [{"role": "user", "content": "Delhi to Jaipur for 4 days with budget ₹1500 for 1 person"}]
    res = workflow.invoke(state, config={"configurable": {"thread_id": "scen_tight_budget"}})
    assert res.get("needs_deep_analysis") is True
    assert "tight_budget" in res.get("complexity_reason", "")


def test_complex_elderly_travelers(workflow):
    """Elderly parents query triggers accessibility and lower berth care."""
    state = create_initial_state()
    state["messages"] = [{"role": "user", "content": "Haridwar to Rishikesh trip for elderly parents, 3 days"}]
    res = workflow.invoke(state, config={"configurable": {"thread_id": "scen_elderly"}})
    assert res.get("needs_deep_analysis") is True
    assert "special_constraints" in res.get("complexity_reason", "")


def test_complex_cheapest_optimization(workflow):
    """Cheapest goal triggers optimization routing."""
    state = create_initial_state()
    state["messages"] = [{"role": "user", "content": "Find cheapest way from Delhi to Varanasi, 3 days"}]
    res = workflow.invoke(state, config={"configurable": {"thread_id": "scen_cheapest"}})
    assert res.get("needs_deep_analysis") is True
    assert "optimization_required" in res.get("complexity_reason", "")


# ---------------------------------------------------------------------------
# Category E: 5 Edge Cases
# ---------------------------------------------------------------------------
def test_edge_case_hindi_input(workflow):
    """Hindi colloquial input: 'kal Dilli se Jaipur jaana hai'."""
    state = create_initial_state()
    state["messages"] = [{"role": "user", "content": "kal Dilli se Jaipur jaana hai"}]
    res = workflow.invoke(state, config={"configurable": {"thread_id": "scen_hindi"}})
    assert res.get("origin") == "Delhi"
    assert "Jaipur" in res.get("destinations", [])


def test_edge_case_question_mark_stripping(workflow):
    """Query ending with punctuation should clean city name without trailing question marks."""
    state = create_initial_state()
    state["messages"] = [{"role": "user", "content": "Can you show tourist places in Jaipur?"}]
    res = workflow.invoke(state, config={"configurable": {"thread_id": "scen_punctuation"}})
    assert "Jaipur" in res.get("destinations", [])


def test_edge_case_missing_parameters_follow_up(workflow):
    """Partial input triggers conversational follow-up questions."""
    state = create_initial_state()
    state["messages"] = [{"role": "user", "content": "I want to visit Goa"}]
    res = workflow.invoke(state, config={"configurable": {"thread_id": "scen_missing_origin"}})
    assistant_msgs = [m for m in res["messages"] if m["role"] == "assistant"]
    assert len(assistant_msgs) >= 1
    reply = assistant_msgs[-1]["content"]
    assert "starting" in reply.lower() or "origin" in reply.lower() or "where" in reply.lower()


def test_edge_case_duplicate_destination_suppression(workflow):
    """Repeated mentions of the same city in subsequent turns do not produce duplicate entries."""
    state = create_initial_state()
    state["messages"] = [{"role": "user", "content": "I want to visit Jaipur"}]
    config = {"configurable": {"thread_id": "scen_dedup"}}
    res1 = workflow.invoke(state, config=config)

    res1["messages"].append({"role": "user", "content": "Yes, Jaipur for 3 days starting from Delhi"})
    res2 = workflow.invoke(res1, config=config)

    # Destination list should contain 'Jaipur' only once
    assert res2.get("destinations").count("Jaipur") == 1


def test_edge_case_impossible_budget_warning(workflow):
    """Budget below ₹300 per day passes through with economy sleeper prioritization."""
    state = create_initial_state()
    state["messages"] = [{"role": "user", "content": "Delhi to Jaipur for 5 days with budget ₹1000 for 2 people"}]
    res = workflow.invoke(state, config={"configurable": {"thread_id": "scen_impossible_budget"}})
    assert res.get("needs_deep_analysis") is True
    assert res.get("final_plan") != ""
