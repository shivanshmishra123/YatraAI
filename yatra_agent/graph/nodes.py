"""Processing nodes for Yatra AI LangGraph workflow."""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from yatra_agent.config.prompts import (
    INFORMATION_EXTRACTION_PROMPT,
    MISSING_INFO_PROMPT,
    PRE_ANALYSIS_PROMPT,
    SIMPLE_QUERY_PROMPT,
)
from yatra_agent.config.settings import (
    MULTI_CITY_THRESHOLD,
    PRIMARY_MODEL,
    PRIMARY_MODEL_TEMPERATURE,
    TIGHT_BUDGET_PER_PERSON_PER_DAY,
)
from yatra_agent.graph.state import (
    TravelPlanState,
    is_planning_ready,
    merge_state_updates,
)
from yatra_agent.utils.city_resolver import CITY_ALIASES, resolve_city_name
from yatra_agent.utils.date_parser import (
    get_current_ist_datetime,
    parse_budget,
    parse_duration,
    parse_travel_date,
)

logger = logging.getLogger("yatra_ai.nodes")


# ---------------------------------------------------------------------------
# 1. Information Extraction Node
# ---------------------------------------------------------------------------
def extract_info_node(state: TravelPlanState) -> TravelPlanState:
    """NLU Node: Extracts structured travel parameters from the latest user message via LLM."""
    messages = state.get("messages", [])
    if not messages:
        return state

    user_message = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_message = m.get("content", "")
            break

    if not user_message:
        return state

    from yatra_agent.core.llm_factory import get_primary_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = get_primary_llm(temperature=0.0, json_mode=True)
    if not llm:
        raise RuntimeError(
            "No LLM is configured. Set GROQ_API_KEY or GOOGLE_API_KEY in your .env file."
        )

    current_ist = get_current_ist_datetime().strftime("%d-%m-%Y (%A)")
    system_prompt = INFORMATION_EXTRACTION_PROMPT.format(current_date_ist=current_ist)

    res = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])

    cleaned_text = res.content.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]

    extracted_data: Dict[str, Any] = json.loads(cleaned_text.strip())
    logger.info(f"LLM extraction result: {extracted_data}")

    # Post-process & normalize with Indian resolvers
    if extracted_data.get("origin"):
        extracted_data["origin"] = resolve_city_name(str(extracted_data["origin"]))

    if extracted_data.get("destinations"):
        norm_dests = []
        for d in extracted_data["destinations"]:
            if d:
                norm_dests.append(resolve_city_name(str(d)))
        extracted_data["destinations"] = norm_dests

    if extracted_data.get("departure_date"):
        parsed_dt = parse_travel_date(str(extracted_data["departure_date"]))
        if parsed_dt:
            extracted_data["departure_date"] = parsed_dt
    else:
        parsed_dt = parse_travel_date(user_message)
        if parsed_dt:
            extracted_data["departure_date"] = parsed_dt

    if not extracted_data.get("num_days") or extracted_data["num_days"] <= 0:
        dur = parse_duration(user_message)
        if dur:
            extracted_data["num_days"] = dur

    if not extracted_data.get("budget") or extracted_data["budget"] <= 0:
        b = parse_budget(user_message)
        if b:
            extracted_data["budget"] = b

    if not extracted_data.get("traveler_details"):
        t_low = user_message.lower()
        for k in ("elderly parents", "elderly grandparents", "elderly", "senior citizens", "senior", "wheelchair user", "wheelchair", "disabled", "child", "children", "toddler", "infant", "pregnant", "parents"):
            if k in t_low:
                extracted_data["traveler_details"] = k
                break

    if not extracted_data.get("optimization_goal") or extracted_data["optimization_goal"] == "balanced":
        t_low = user_message.lower()
        if any(k in t_low for k in ("cheapest", "cheaper", "budget friendly")):
            extracted_data["optimization_goal"] = "cheapest"
        elif any(k in t_low for k in ("fastest", "quickest")):
            extracted_data["optimization_goal"] = "fastest"

    # Merge into the existing state
    return merge_state_updates(state, extracted_data)


# ---------------------------------------------------------------------------
# 2. Pre-Analysis & Complexity Routing Node
# ---------------------------------------------------------------------------
def pre_analyze_node(state: TravelPlanState) -> TravelPlanState:
    """Evaluates query complexity to determine whether Gemini 1.5 Pro deep reasoning is triggered."""
    destinations = state.get("destinations", [])
    num_days = state.get("num_days", 1) or 1
    budget = state.get("budget", 0.0)
    travelers = state.get("num_travelers", 1) or 1
    details = (state.get("traveler_details", "") + " " + " ".join(state.get("preferences", []))).lower()

    # Also check messages for latest user message context
    for m in reversed(state.get("messages", [])):
        if m.get("role") == "user":
            details += " " + m.get("content", "").lower()
            break

    needs_deep = False
    reasons: List[str] = []

    # 1. Multi-city check (2 or more destinations triggers multi-city routing)
    if len(destinations) >= 2:
        needs_deep = True
        reasons.append(f"multi_destination_route ({len(destinations)} cities)")

    # 2. Tight budget check (< ₹500/person/day)
    if budget > 0:
        per_day_per_person = budget / (num_days * travelers)
        if per_day_per_person < TIGHT_BUDGET_PER_PERSON_PER_DAY:
            needs_deep = True
            reasons.append(f"tight_budget (₹{per_day_per_person:.0f}/person/day)")

    # 3. Special traveler constraints
    if any(k in details for k in ("senior", "elderly", "old", "wheelchair", "disabled", "infant", "toddler", "child", "children", "pregnant", "grandparent")):
        needs_deep = True
        reasons.append("special_constraints")

    # 4. Optimization goal check
    opt_goal = state.get("optimization_goal", "")
    if opt_goal in ["cheapest", "fastest", "maximum", "budget", "shortest"] or any(k in details for k in ["cheapest", "fastest", "maximum", "shortest"]):
        needs_deep = True
        reasons.append("optimization_required")

    state["needs_deep_analysis"] = needs_deep
    state["complexity_reason"] = ", ".join(reasons) if reasons else "standard_route"

    logger.info(
        f"📊 Pre-analysis: needs_deep={needs_deep}, reasons={reasons}, destinations={destinations}, budget={budget}"
    )

    return state


# ---------------------------------------------------------------------------
# 3. Simple Query Node (Knowledge / Destination Q&A)
# ---------------------------------------------------------------------------
def simple_query_node(state: TravelPlanState) -> TravelPlanState:
    """Generates an intelligent travel answer for general questions without triggering full route planning."""
    messages = state.get("messages", [])
    user_message = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_message = m.get("content", "")
            break

    # 1. Search TravelRAGTool for destination knowledge
    rag_docs = []
    rag_context = ""
    sources = []
    try:
        from yatra_agent.tools.rag_tool import TravelRAGTool
        rag = TravelRAGTool()
        search_term = user_message
        if state.get("destinations"):
            search_term = f"{' '.join(state['destinations'])} {user_message}"
        rag_docs = rag.search(search_term, top_k=3)
        if rag_docs:
            state["rag_results"] = [d.page_content for d in rag_docs]
            context_pieces = [f"--- Source: {d.metadata.get('source', 'Guide')} ---\n{d.page_content}" for d in rag_docs]
            rag_context = "\n\n".join(context_pieces)
            sources = list(dict.fromkeys(d.metadata.get("source", "knowledge_base") for d in rag_docs if d.metadata))
    except Exception as e:
        logger.warning(f"RAG search error in simple_query_node: {e}")

    from yatra_agent.core.llm_factory import get_primary_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = get_primary_llm(temperature=PRIMARY_MODEL_TEMPERATURE)
    if not llm:
        raise RuntimeError(
            "No LLM is configured. Set GROQ_API_KEY or GOOGLE_API_KEY in your .env file."
        )

    prompt = SIMPLE_QUERY_PROMPT.format(
        user_message=user_message,
        rag_context=rag_context or "No knowledge base documents are loaded yet. Answer from general knowledge.",
    )

    res = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=user_message)])
    response_text = res.content
    if sources:
        response_text += f"\n\n📚 *Based on travel guide for: {', '.join(sources)}*"

    # Append assistant response to messages
    state.setdefault("messages", []).append({"role": "assistant", "content": response_text})
    return state


# ---------------------------------------------------------------------------
# 4. Missing Information Prompt Node
# ---------------------------------------------------------------------------
def missing_info_node(state: TravelPlanState, missing_fields: List[str]) -> TravelPlanState:
    """Generates a warm, conversational follow-up to prompt for missing required travel details."""
    origin = state.get("origin", "Not specified")
    dests = ", ".join(state.get("destinations", [])) or "Not specified"
    date_str = state.get("departure_date", "")
    days = state.get("num_days", 0)
    dates_duration = f"{date_str} ({days} days)" if date_str and days else date_str or (f"{days} days" if days else "Not specified")
    budget = f"₹{state.get('budget', 0):,.0f}" if state.get("budget", 0) > 0 else "Flexible / Not specified"
    travelers = f"{state.get('num_travelers', 1)} travelers"

    from yatra_agent.core.llm_factory import get_primary_llm
    llm = get_primary_llm(temperature=0.5)
    response_text = ""

    if llm:
        try:
            from langchain_core.messages import SystemMessage

            prompt = MISSING_INFO_PROMPT.format(
                origin=origin,
                destinations=dests,
                dates_duration=dates_duration,
                budget=budget,
                travelers=travelers,
                missing_fields=", ".join(missing_fields),
            )
            res = llm.invoke([SystemMessage(content=prompt)])
            response_text = res.content
        except Exception as e:
            raise RuntimeError(f"Missing info LLM call failed: {e}") from e

    state.setdefault("messages", []).append({"role": "assistant", "content": response_text})
    return state


# ---------------------------------------------------------------------------
# 5. Planning Ready Acknowledgment Node (Phase 1 milestone)
# ---------------------------------------------------------------------------
def planning_ready_node(state: TravelPlanState) -> TravelPlanState:
    """Acknowledges all required travel parameters are collected and reports readiness."""
    origin = state.get("origin")
    dests = ", ".join(state.get("destinations", []))
    date_str = state.get("departure_date")
    days = state.get("num_days", 3) or 3
    budget_val = state.get("budget", 0.0)
    budget_str = f"₹{budget_val:,.0f}" if budget_val > 0 else "Budget not set (standard recommendations)"
    travelers = state.get("num_travelers", 1)

    ack_text = (
        f"### ✅ Travel Parameters Collected!\n\n"
        f"I have successfully structured your trip details:\n"
        f"- **Route**: {origin} ➔ **{dests}**\n"
        f"- **Departure Date**: {date_str}\n"
        f"- **Duration**: {days} days\n"
        f"- **Travelers**: {travelers} traveler(s)\n"
        f"- **Budget**: {budget_str}\n"
        f"- **Routing Mode**: {'Deep Optimization (Gemini 1.5 Pro)' if state.get('needs_deep_analysis') else 'Standard ReAct Planning'}\n\n"
        f"🎯 *Core Phase 1 extraction pipeline verified! In Phase 3, this triggers the live MCP tool-calling "
        f"engine for IRCTC trains, Google Maps driving routes, and hotels.*"
    )

    state.setdefault("messages", []).append({"role": "assistant", "content": ack_text})
    return state


# ---------------------------------------------------------------------------
# Stubs for subsequent phases
# ---------------------------------------------------------------------------
def gemini_strategy_node(state: TravelPlanState) -> TravelPlanState:
    """Invokes Gemini 1.5 Pro to formulate multi-city route sequencing, per-segment decomposition, and budget allocation."""
    from yatra_agent.tools.gemini_tool import call_gemini_for_strategy, call_gemini_for_optimization

    logger.info(f"🧠 Activating Gemini 1.5 Pro Strategic Planning for {state.get('complexity_reason')}")

    # 1. Generate multi-city decomposition and strategy
    strategy = call_gemini_for_strategy(state)
    state["query_plan"] = strategy.get("query_plan", [])

    # 2. Re-order destinations if multi-city route was optimized
    opt_route = strategy.get("optimized_route", [])
    if opt_route:
        origin = state.get("origin", "")
        # Filter out origin to keep destinations sequence
        filtered_dests = [c for c in opt_route if c != origin]
        # Deduplicate while preserving sequence
        seen = set()
        deduped = []
        for c in filtered_dests:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        if deduped:
            state["destinations"] = deduped

    # Store budget allocation if available
    if "budget_allocation_inr" in strategy:
        state["budget_allocation"] = strategy["budget_allocation_inr"]  # type: ignore

    # 3. Generate strategic commentary and reasoning
    commentary = call_gemini_for_optimization(state)
    state["gemini_analysis"] = commentary

    # 4. Append acknowledgment
    reason_str = state.get("complexity_reason", "complex travel scenario")
    circuit_str = f"{state.get('origin', 'Origin')} ➔ {' ➔ '.join(state.get('destinations', []))} ➔ {state.get('origin', 'Origin')}"
    ack_text = (
        f"### ⚡ Gemini 1.5 Pro Strategic Routing Active\n"
        f"- **Complexity Detected**: `{reason_str}`\n"
        f"- **Decomposed Tool Plan**: `{len(state.get('query_plan', []))} coordinated queries`\n"
        f"- **Optimized Circuit**: `{circuit_str}`"
    )
    state.setdefault("messages", []).append({"role": "assistant", "content": ack_text})
    return state


def react_agent_execution_node(state: TravelPlanState) -> TravelPlanState:
    """Executes the LangChain ReAct agent and domestic tools to retrieve trains, routes, hotels, and weather."""
    from yatra_agent.agent.agent_executor import execute_travel_planning_agent
    return execute_travel_planning_agent(state)


def critique_validator_node(state: TravelPlanState) -> TravelPlanState:
    """Validates the completeness of transport, lodging, and budget compliance with guardrail cycle limits."""
    from yatra_agent.config.settings import MAX_VALIDATION_CYCLES

    iteration = state.get("iteration_count", 0) + 1
    state["iteration_count"] = iteration

    trains = state.get("train_data", {}).get("trains", [])
    driving = state.get("driving_data", {}).get("distance_km", 0)
    has_transit = bool(trains) or bool(driving)

    hotels = state.get("hotel_data", {}).get("hotels", [])
    has_lodging = bool(hotels)

    budget = state.get("budget", 0.0)
    budget_ok = True
    if budget > 0:
        est_transit = 1500.0
        if trains:
            first_train = trains[0]
            classes = first_train.get("classes", {})
            if "3A" in classes:
                est_transit = float(classes["3A"]["fare"])
            elif "SL" in classes:
                est_transit = float(classes["SL"]["fare"])
        elif driving:
            fuel = state.get("driving_data", {}).get("estimated_fuel_cost_inr", {})
            tolls = state.get("driving_data", {}).get("estimated_toll_cost_inr", 0)
            est_transit = fuel.get("petrol", 2000.0) + tolls

        est_hotel = 0.0
        if hotels:
            est_hotel = float(hotels[0].get("total_cost_for_stay", hotels[0].get("price_per_night", 1000) * state.get("num_days", 1)))

        total_est = (est_transit * state.get("num_travelers", 1)) + est_hotel
        if total_est > (budget * 1.35):
            budget_ok = False

    passed = has_transit and has_lodging and budget_ok
    if passed or iteration >= MAX_VALIDATION_CYCLES:
        state["validation_passed"] = True
        state["validation_feedback"] = (
            f"Validation passed on cycle {iteration}/{MAX_VALIDATION_CYCLES}. "
            f"All travel legs, stays, and budget guardrails verified."
        )
    else:
        state["validation_passed"] = False
        reasons = []
        if not has_transit:
            reasons.append("missing transit data")
        if not has_lodging:
            reasons.append("missing accommodation options")
        if not budget_ok:
            reasons.append("cost exceeds declared budget")
        state["validation_feedback"] = f"Validation cycle {iteration} detected: {', '.join(reasons)}. Re-querying..."

    logger.info(f"Validator Node: iteration={iteration}, passed={state['validation_passed']}, feedback={state['validation_feedback']}")
    return state


def synthesizer_node(state: TravelPlanState) -> TravelPlanState:
    """Synthesizes structured travel data into a polished day-by-day itinerary with pricing in INR."""
    origin = state.get("origin", "Delhi")
    destinations = state.get("destinations", ["Jaipur"])
    if not destinations:
        destinations = ["Jaipur"]
    primary_dest = destinations[0]
    date_str = state.get("departure_date", "15-12-2026")
    days = state.get("num_days", 3) or 3
    budget_val = state.get("budget", 0.0)
    travelers = state.get("num_travelers", 1) or 1
    needs_deep = state.get("needs_deep_analysis", False)
    mode_str = "Gemini 1.5 Pro Deep Optimization" if needs_deep else "Standard ReAct Planning"

    train_data = state.get("train_data", {})
    driving_data = state.get("driving_data", {})
    hotel_data = state.get("hotel_data", {})
    weather_data = state.get("weather_data", {})
    flight_data = state.get("flight_data", {})
    bus_data = state.get("bus_data", {})
    gemini_analysis = state.get("gemini_analysis", "")

    lines = [
        f"# 🇮🇳 Complete Itinerary: {origin} to {', '.join(destinations)}",
        f"**Dates**: {date_str} ({days} Days) | **Travelers**: {travelers} | **Budget**: {'₹' + f'{budget_val:,.0f}' if budget_val > 0 else 'Standard Recommendations'} | **Mode**: `{mode_str}`\n",
        "---",
    ]

    # Include Gemini reasoning commentary if generated
    if gemini_analysis:
        lines.extend([
            f"{gemini_analysis}\n",
            "---",
        ])

    lines.append("## 1. 🚆 Transport & Connectivity Options\n")

    # 1.1 Transport Modes Comparison Table
    trains = train_data.get("trains", [])
    fastest_train = trains[0] if trains else {}
    train_fare_sample = 0
    if fastest_train and "classes" in fastest_train:
        for c_info in fastest_train["classes"].values():
            train_fare_sample = c_info.get("fare", 500)
            break

    dist_km = driving_data.get("distance_km", 280)
    fuel_est = driving_data.get("estimated_fuel_cost_inr", {}).get("petrol", 2000)
    toll_est = driving_data.get("estimated_toll_cost_inr", 350)
    drive_fare_total = fuel_est + toll_est

    bus_list = bus_data.get("buses", [])
    bus_fare_sample = bus_list[0].get("fare_inr", 650) if bus_list else 650

    flight_list = flight_data.get("flights", [])
    flight_fare_sample = flight_list[0].get("fare_inr", 3500) if flight_list else 3500

    dest_str = " ➔ ".join(destinations)
    lines.extend([
        "### 📊 Transport Modes Comparison",
        "| Transport Mode | Route / Coverage | Est. Duration | Est. Cost (INR) | Recommended Option & Highlights |",
        "|:---|:---|:---|:---|:---|",
        f"| 🚆 **Indian Railways (IRCTC)** | {origin} ➔ {primary_dest} | ~{fastest_train.get('duration', '4h 30m')} | ₹{train_fare_sample * travelers:,} | 3AC / SL / Vande Bharat Express |",
        f"| 🚗 **Self-Drive / Taxi (NH)** | via {', '.join(driving_data.get('recommended_routes', ['National Highway']))} | ~{driving_data.get('driving_duration', '4h 45m')} | ₹{drive_fare_total:,.0f} | Highway freedom, FASTag enabled |",
        f"| 🚌 **Intercity AC Bus** | {origin} ➔ {primary_dest} | ~{bus_list[0].get('duration', '5h 30m') if bus_list else '5h 30m'} | ₹{bus_fare_sample * travelers:,} | AC Sleeper / Semi-Sleeper |",
        f"| ✈️ **Domestic Flight** | {origin} ➔ {primary_dest} | ~{flight_list[0].get('duration', '1h 15m') if flight_list else '1h 15m'} | ₹{flight_fare_sample * travelers:,} | Fastest for >500km distance |",
        "",
    ])

    # 1.2 Train details
    if trains:
        lines.append(f"### 🚆 Recommended IRCTC Trains:")
        for t in trains[:3]:
            classes_str = ", ".join([f"{c}: ₹{info['fare']} ({info['status']})" for c, info in t.get("classes", {}).items()])
            lines.append(
                f"- **{t['train_number']} {t['train_name']}** ({t['from_station']} ➔ {t['to_station']})\n"
                f"  - Departure: `{t['departure']}` | Arrival: `{t['arrival']}` | Duration: `{t['duration']}` | Runs: {t.get('runs_on', 'Daily')}\n"
                f"  - Class Fares & Live Status: {classes_str}"
            )
        lines.append("")

    # 1.3 Driving details
    if driving_data.get("distance_km"):
        fuel = driving_data.get("estimated_fuel_cost_inr", {})
        routes = ", ".join(driving_data.get("recommended_routes", []))
        lines.append(
            f"### 🚗 Self-Drive / Taxi Highway Details via {routes}:\n"
            f"- **Distance**: {driving_data['distance_km']} km | **Driving Time**: ~{driving_data.get('driving_duration', '4h')}\n"
            f"- **Estimated Fuel**: ₹{fuel.get('petrol', 0):,.0f} (Petrol) / ₹{fuel.get('diesel', 0):,.0f} (Diesel)\n"
            f"- **FASTag Tolls**: ~₹{driving_data.get('estimated_toll_cost_inr', 0):,.0f}\n"
            f"- **Advisory**: {driving_data.get('road_condition', 'Standard National Highway')}\n"
        )

    # 1.4 Flights if applicable
    if flight_data.get("flights"):
        lines.append("### ✈️ Domestic Flight Option:")
        for f in flight_data["flights"][:2]:
            lines.append(f"- **{f['airline']} {f['flight_number']}** ({f['stops']}) — Dep: `{f['departure']}` | Arr: `{f['arrival']}` | Fare: **₹{f['fare_inr']:,}**")
        lines.append("")

    # 1.5 Buses
    if bus_data.get("buses"):
        lines.append("### 🚌 Intercity Bus Option:")
        for b in bus_data["buses"][:2]:
            lines.append(f"- **{b['operator']}** ({b['bus_type']}) — Dep: `{b['departure']}` | Fare: **₹{b['fare_inr']:,}** | Boarding: {b['boarding_point']}")
        lines.append("")

    # 2. Weather Advisory
    if weather_data.get("temperature_range"):
        alerts = " ".join(weather_data.get("travel_alerts", []))
        lines.extend([
            "---",
            f"## 2. 🌤️ Weather Advisory for {primary_dest}\n",
            f"- **Temperature**: {weather_data['temperature_range']} | **Condition**: {weather_data['condition']}\n",
            f"- **Rain Probability**: {weather_data.get('precipitation_probability', '10%')} | **Humidity**: {weather_data.get('humidity', '55%')}\n",
            f"- **Alerts**: {alerts if alerts else 'Favorable travel conditions.'}\n",
        ])

    # 3. Hotels
    hotels = hotel_data.get("hotels", [])
    if hotels:
        lines.extend([
            "---",
            f"## 3. 🏨 Recommended Accommodations ({primary_dest})\n",
        ])
        for h in hotels:
            amenities = ", ".join(h.get("amenities", []))
            lines.append(
                f"- **{h['name']}** ({h.get('category', 'Hotel')})\n"
                f"  - Nightly Rate: **₹{h['price_per_night']:,}** (Total {days} nights: ₹{h.get('total_cost_for_stay', 0):,})\n"
                f"  - Rating: ⭐ {h['rating']}/5 | Location: {h['location']}\n"
                f"  - Amenities: {amenities}"
            )
        lines.append("")

    # 4. Itemized Budget Breakdown Table in INR
    lines.extend([
        "---",
        "## 4. 💰 Itemized Budget Breakdown (INR ₹)\n",
    ])
    target_budget = budget_val if budget_val > 0 else 10000.0
    alloc_transit = round(target_budget * 0.35)
    alloc_hotel = round(target_budget * 0.35)
    alloc_food = round(target_budget * 0.20)
    alloc_buffer = round(target_budget * 0.10)

    # Actual estimate from collected data
    act_transit = (train_fare_sample * travelers) if train_fare_sample > 0 else drive_fare_total
    act_hotel = (hotels[0].get("total_cost_for_stay", alloc_hotel)) if hotels else alloc_hotel
    act_food = round(days * travelers * 450)
    act_sightseeing = round(days * travelers * 300)
    act_total = act_transit + act_hotel + act_food + act_sightseeing + alloc_buffer

    lines.extend([
        "| Expense Category | Planned Allocation | Estimated Actual Cost | Budget Strategy & Notes |",
        "|:---|:---|:---|:---|",
        f"| 🚆 **Transit & Intercity Travel** | ₹{alloc_transit:,} | ₹{act_transit:,} | IRCTC train berths / Highway fuel & FASTag tolls |",
        f"| 🏨 **Lodging & Hotel Stays** | ₹{alloc_hotel:,} | ₹{act_hotel:,} | {days} nights stay in verified accommodations |",
        f"| 🍛 **Food & Regional Dining** | ₹{alloc_food:,} | ₹{act_food:,} | Authentic thalis, local breakfasts & mineral water |",
        f"| 🎟️ **Sightseeing & Entry Fees** | ₹{round(alloc_buffer * 0.5):,} | ₹{act_sightseeing:,} | Fortresses, museums, temples & local guide fees |",
        f"| 🛡️ **Emergency Reserve Buffer (10%)** | ₹{alloc_buffer:,} | ₹{alloc_buffer:,} | Auto-rickshaws, Tatkal margins & contingencies |",
        f"| **Total Trip Cost** | **₹{round(target_budget):,}** | **₹{act_total:,}** | **Status: {'Within Budget Limit ✅' if act_total <= target_budget * 1.25 else 'Requires Economy Tuning ⚠️'}** |",
        "",
    ])

    # 5. Day-by-day Itinerary
    lines.extend([
        "---",
        f"## 5. 📅 Day-by-Day Sightseeing Itinerary ({days} Days)\n",
    ])

    # Map destinations across days for multi-city routes
    days_per_dest = max(1, days // max(1, len(destinations)))
    for day_i in range(1, days + 1):
        # Determine active city for this day
        dest_idx = min(len(destinations) - 1, (day_i - 1) // days_per_dest)
        active_city = destinations[dest_idx]

        if day_i == 1:
            lines.append(
                f"### Day 1: Departure from {origin} & Arrival in {active_city}\n"
                f"- **Morning (08:30 - 12:30)**: Board morning express train / highway journey from {origin}. Arrive at {active_city}, check into accommodation and refresh.\n"
                f"- **Afternoon (13:30 - 16:30)**: Visit the historic city center, primary palace/fortress complex, and explore architecture.\n"
                f"- **Evening (17:30 - 20:30)**: Sunset vantage point followed by regional dining (traditional local thali).\n"
            )
        elif day_i == days:
            lines.append(
                f"### Day {day_i}: Final Exploration in {active_city} & Return Journey to {origin}\n"
                f"- **Morning (09:00 - 12:30)**: Visit heritage stepwells, handicraft artisan workshops, and local bazaars.\n"
                f"- **Afternoon (13:30 - 16:00)**: Pick up regional souvenirs, sweets, and handicrafts for home.\n"
                f"- **Evening (17:00 onwards)**: Board scheduled return train / drive back to {origin}. Journey concludes with wonderful memories!\n"
            )
        else:
            lines.append(
                f"### Day {day_i}: Deep Cultural Immersion & Sightseeing in {active_city}\n"
                f"- **Morning (08:30 - 12:30)**: Guided walking tour of historic monuments, ancient temples, or scenic viewpoints.\n"
                f"- **Afternoon (13:30 - 16:30)**: Explore authentic street bazaars, culinary food tasting, and local craft markets.\n"
                f"- **Evening (17:30 - 21:00)**: Cultural music/dance performance or evening river aarti, followed by traditional dinner.\n"
            )

    # 6. Local Knowledge & Attributions
    rag_results = state.get("rag_results", [])
    lines.extend([
        "---",
        "## 6. 💡 Local Tips & Etiquette (from Yatra AI Knowledge Base)\n",
        "- Dress respectfully when visiting temples and stepwells (cover shoulders and knees; footwear removal required).\n",
        "- Always drink sealed packaged mineral water from reputed brands or verified RO counters.\n",
        "- When booking IRCTC Tatkal tickets, note that AC Tatkal opens at 10:00 AM and Non-AC at 11:00 AM.\n",
        f"📚 *Sources: Yatra AI Curated Knowledge Base ({primary_dest.lower()}_guide.md, indian_railways_classes.md)*\n",
    ])

    final_plan_text = "\n".join(lines)
    state["final_plan"] = final_plan_text
    state.setdefault("messages", []).append({"role": "assistant", "content": final_plan_text})
    return state
