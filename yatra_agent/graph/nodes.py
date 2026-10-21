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
# Fallback Heuristic Extractor (Deterministic rule engine for offline / testing)
# ---------------------------------------------------------------------------
def _fallback_extract_info(text: str) -> Dict[str, Any]:
    """Heuristic rule-based parameter extractor used as fallback or offline testing."""
    extracted: Dict[str, Any] = {
        "origin": "",
        "destinations": [],
        "departure_date": "",
        "return_date": "",
        "num_days": 0,
        "budget": 0.0,
        "num_travelers": 1,
        "traveler_details": "",
        "preferences": [],
        "optimization_goal": "balanced",
        "travel_mode_pref": "",
        "query_type": "simple",
    }

    t_lower = text.lower().strip()
    t_clean = re.sub(r"[?!.,;:]", " ", t_lower).strip()

    # 1. Check explicit "from <origin> to <dest>" or "<origin> to <dest>" or "<origin> se <dest>"
    # Pattern A: "from X to Y" or "from X -> Y"
    from_to_match = re.search(r"\bfrom\s+([a-z\s]+?)\s+(?:to|->)\s+([a-z\s,]+?)(?=\s+(?:for|in|on|with|starting|\d)|\s*$)", t_clean)
    if not from_to_match:
        # Pattern B: "X to Y" or "X -> Y" (where X and Y are cities, e.g. "Delhi to Jaipur")
        from_to_match = re.search(r"(?:^|plan\s+(?:a\s+)?(?:trip\s+)?(?:from\s+)?)([a-z\s]+?)\s+(?:to|->)\s+([a-z\s,]+?)(?=\s+(?:for|in|on|with|starting|\d)|\s*$)", t_clean)
    
    # Pattern C: Hindi "X se Y (jaana)"
    se_match = re.search(r"([a-z\s]+?)\s+se\s+([a-z\s,]+?)(?:\s+jaana|\s+ko|\s+for|\s+in|\s*$|\d)", t_clean)

    matched_origin = ""
    matched_dests = []

    if from_to_match and "want" not in from_to_match.group(1).split():
        cand_orig = from_to_match.group(1).strip()
        cand_dest = from_to_match.group(2).strip()
        # Clean leading words like "trip", "plan", "kal"
        cand_orig = re.sub(r"^(?:plan\s+(?:a\s+)?(?:trip\s+)?(?:from\s+)?|trip\s+(?:from\s+)?|from\s+|kal\s+|aaj\s+)", "", cand_orig).strip()
        cand_dest = re.sub(r"^(?:visit\s+)", "", cand_dest).strip()
        
        orig_res = resolve_city_name(cand_orig)
        if orig_res and (cand_orig.lower() in CITY_ALIASES or len(cand_orig.split()) <= 2):
            matched_origin = orig_res

        for d in re.split(r",|\s+and\s+|\s+aur\s+", cand_dest):
            d_clean = re.sub(r"^(?:visit\s+)", "", d.strip()).strip()
            if d_clean:
                dest_res = resolve_city_name(d_clean)
                if dest_res and (d_clean.lower() in CITY_ALIASES or len(d_clean.split()) <= 2):
                    matched_dests.append(dest_res)
    elif se_match:
        cand_orig = se_match.group(1).strip()
        cand_dest = se_match.group(2).strip()
        cand_orig = re.sub(r"^(?:kal\s+|aaj\s+|parso\s+)", "", cand_orig).strip()
        cand_dest = re.sub(r"\s+jaana.*$", "", cand_dest).strip()

        orig_res = resolve_city_name(cand_orig)
        if orig_res:
            matched_origin = orig_res
        dest_res = resolve_city_name(cand_dest)
        if dest_res:
            matched_dests.append(dest_res)

    # If no route match, check single origin/destination patterns
    if not matched_origin:
        single_from = re.search(r"\b(?:from|starting\s+from|se)\s+([a-z\s]+?)(?=\s+(?:for|in|on|with|\d)|\s*$)", t_clean)
        if single_from:
            cand = single_from.group(1).strip()
            res = resolve_city_name(cand)
            if cand.lower() in CITY_ALIASES or len(cand.split()) <= 2:
                matched_origin = res

    if not matched_dests:
        for phrase in ("visit", "trip to", "go to", "places in", "guide for", "about", "to"):
            match_single_dest = re.search(rf"\b{phrase}\s+([a-z\s]+?)(?=\s+(?:for|in|on|with|this|next|\d)|\s*$)", t_clean)
            if match_single_dest:
                cand = match_single_dest.group(1).strip()
                cand = re.sub(r"^(?:visit\s+)", "", cand).strip()
                res = resolve_city_name(cand)
                if cand.lower() in CITY_ALIASES or len(cand.split()) <= 2:
                    matched_dests.append(res)
                    break

    if matched_origin:
        extracted["origin"] = matched_origin
    if matched_dests:
        extracted["destinations"] = matched_dests

    # 2. Date parsing
    date_val = parse_travel_date(t_lower)
    if date_val:
        extracted["departure_date"] = date_val

    # 3. Duration parsing
    days_val = parse_duration(t_lower)
    if days_val:
        extracted["num_days"] = days_val

    # 4. Budget parsing
    budget_val = parse_budget(t_lower)
    if budget_val:
        extracted["budget"] = budget_val

    # 5. Travelers parsing
    travelers_match = re.search(r"(\d+)\s*(?:people|persons?|travelers?|adults?|log)", t_lower)
    if travelers_match:
        extracted["num_travelers"] = int(travelers_match.group(1))
    elif "couple" in t_lower or "2 people" in t_lower:
        extracted["num_travelers"] = 2
    elif "family of 4" in t_lower:
        extracted["num_travelers"] = 4

    # 6. Query type classification
    if extracted["origin"] or (extracted["destinations"] and (extracted["departure_date"] or extracted["num_days"] or extracted["budget"])):
        extracted["query_type"] = "full_planning"
    elif any(k in t_lower for k in ("what", "tell", "best", "how", "places", "food", "attractions", "weather")):
        extracted["query_type"] = "simple"
    else:
        extracted["query_type"] = "chat"

    return extracted


# ---------------------------------------------------------------------------
# 1. Information Extraction Node
# ---------------------------------------------------------------------------
def extract_info_node(state: TravelPlanState) -> TravelPlanState:
    """NLU Node: Extracts structured travel parameters from the latest user message."""
    messages = state.get("messages", [])
    if not messages:
        return state

    # Get the latest user message
    user_message = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_message = m.get("content", "")
            break

    if not user_message:
        return state

    extracted_data: Dict[str, Any] = {}
    openai_key = os.getenv("OPENAI_API_KEY")

    # Attempt LLM extraction if API key is provided
    if openai_key and not openai_key.startswith("sk-your"):
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage

            current_ist = get_current_ist_datetime().strftime("%d-%m-%Y (%A)")
            system_prompt = INFORMATION_EXTRACTION_PROMPT.format(current_date_ist=current_ist)

            llm = ChatOpenAI(
                model=PRIMARY_MODEL,
                temperature=0.0,
                api_key=openai_key,
                model_kwargs={"response_format": {"type": "json_object"}},
            )

            res = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ])

            extracted_data = json.loads(res.content)
            logger.info(f"LLM extraction result: {extracted_data}")
        except Exception as e:
            logger.warning(f"LLM extraction failed, using fallback heuristic: {e}")
            extracted_data = _fallback_extract_info(user_message)
    else:
        # Fallback to local deterministic parser
        extracted_data = _fallback_extract_info(user_message)

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
    details = state.get("traveler_details", "").lower()
    preferences = " ".join(state.get("preferences", [])).lower()

    needs_deep = False
    reasons: List[str] = []

    # 1. Multi-city check
    if len(destinations) >= MULTI_CITY_THRESHOLD:
        needs_deep = True
        reasons.append(f"multi_destination_route ({len(destinations)} cities)")

    # 2. Tight budget check
    if budget > 0:
        per_day_per_person = budget / (num_days * travelers)
        if per_day_per_person < TIGHT_BUDGET_PER_PERSON_PER_DAY:
            needs_deep = True
            reasons.append(f"tight_budget (₹{per_day_per_person:.0f}/person/day)")

    # 3. Special traveler constraints
    if any(k in details for k in ("senior", "elderly", "old", "wheelchair", "disabled", "infant", "toddler")):
        needs_deep = True
        reasons.append("special_traveler_constraints")

    # 4. Conflicting optimization goals
    if "cheapest" in preferences and "luxury" in preferences:
        needs_deep = True
        reasons.append("conflicting_budget_luxury_constraints")

    state["needs_deep_analysis"] = needs_deep
    state["complexity_reason"] = ", ".join(reasons) if reasons else "standard_route"

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

    openai_key = os.getenv("OPENAI_API_KEY")
    response_text = ""

    if openai_key and not openai_key.startswith("sk-your"):
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage

            prompt = SIMPLE_QUERY_PROMPT.format(
                user_message=user_message,
                rag_context=rag_context or "Standard curated guidelines apply.",
            )

            llm = ChatOpenAI(model=PRIMARY_MODEL, temperature=PRIMARY_MODEL_TEMPERATURE, api_key=openai_key)
            res = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=user_message)])
            response_text = res.content
            if sources:
                response_text += f"\n\n📚 *Based on our travel guide for {', '.join(sources)}*"
        except Exception as e:
            logger.warning(f"Simple query LLM call failed: {e}")

    if not response_text:
        # High quality knowledge-enhanced offline fallback response
        dests = state.get("destinations", [])
        place = dests[0] if dests else "your destination"

        if rag_docs:
            top_doc = rag_docs[0]
            src = top_doc.metadata.get("source", f"{place.lower()}_guide.md")
            response_text = (
                f"### 🌟 Exploring {place}\n\n"
                f"{top_doc.page_content}\n\n"
                f"📚 *Based on our travel guide for {src}*\n\n"
                f"---\n"
                f"💡 *To generate a full day-wise itinerary with real-time train availability, driving routes, "
                f"and budget estimation, let me know your departure city and preferred travel dates!*"
            )
        else:
            response_text = (
                f"### 🌟 Exploring {place}\n\n"
                f"Here are key travel insights for **{place}**:\n\n"
                f"- **🏛️ Key Attractions**: Historic monuments, cultural landmarks, and local bazaars.\n"
                f"- **🍛 Food Specialties**: Don't miss the local street food and traditional delicacies!\n"
                f"- **🕒 Best Visiting Time**: Early mornings and late afternoons are ideal for sightseeing.\n"
                f"- **🚆 Connectivity**: Well connected via Indian Railways and National Highways.\n\n"
                f"*To generate a full day-wise itinerary with real-time train availability, driving routes, "
                f"and budget estimation, let me know your departure city and preferred travel dates!*"
            )

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

    openai_key = os.getenv("OPENAI_API_KEY")
    response_text = ""

    if openai_key and not openai_key.startswith("sk-your"):
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage

            prompt = MISSING_INFO_PROMPT.format(
                origin=origin,
                destinations=dests,
                dates_duration=dates_duration,
                budget=budget,
                travelers=travelers,
                missing_fields=", ".join(missing_fields),
            )
            llm = ChatOpenAI(model=PRIMARY_MODEL, temperature=0.5, api_key=openai_key)
            res = llm.invoke([SystemMessage(content=prompt)])
            response_text = res.content
        except Exception as e:
            logger.warning(f"Missing info prompt failed: {e}")

    if not response_text:
        # Fallback structured prompt
        questions = []
        if "origin" in missing_fields:
            questions.append("🏙️ **Where will you be starting your journey from?** (e.g. Delhi, Mumbai, Bengaluru)")
        if "destinations" in missing_fields:
            questions.append("📍 **Which destination(s) would you like to visit?** (e.g. Jaipur, Goa, Varanasi)")
        if "departure_date" in missing_fields:
            questions.append("📅 **When are you planning to travel and for how many days?** (e.g. Dec 15 for 3 days, or this weekend)")

        collected_summary = []
        if state.get("destinations"):
            collected_summary.append(f"Destination: **{', '.join(state['destinations'])}**")
        if state.get("origin"):
            collected_summary.append(f"Origin: **{state['origin']}**")
        if state.get("budget", 0) > 0:
            collected_summary.append(f"Budget: **₹{state['budget']:,.0f}**")

        summary_text = f"Got it! I have noted your preferences ({', '.join(collected_summary)}).\n\n" if collected_summary else "I'd love to help you plan your journey!\n\n"
        response_text = (
            f"{summary_text}"
            f"To build your complete itinerary with live IRCTC trains, highway routes, and hotels, please share:\n"
            + "\n".join(f"- {q}" for q in questions)
        )

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
    """Stub for Phase 4 Gemini Strategy."""
    return state


def react_agent_execution_node(state: TravelPlanState) -> TravelPlanState:
    """Executes the LangChain ReAct agent and domestic tools to retrieve trains, routes, hotels, and weather."""
    from yatra_agent.agent.agent_executor import execute_travel_planning_agent
    return execute_travel_planning_agent(state)


def critique_validator_node(state: TravelPlanState) -> TravelPlanState:
    """Stub for Phase 4 validation loop."""
    return state


def synthesizer_node(state: TravelPlanState) -> TravelPlanState:
    """Synthesizes structured travel data into a polished day-by-day itinerary with pricing in INR."""
    origin = state.get("origin", "Delhi")
    destinations = state.get("destinations", ["Jaipur"])
    primary_dest = destinations[0] if destinations else "Jaipur"
    date_str = state.get("departure_date", "15-12-2026")
    days = state.get("num_days", 3) or 3
    budget_val = state.get("budget", 0.0)
    travelers = state.get("num_travelers", 1)

    train_data = state.get("train_data", {})
    driving_data = state.get("driving_data", {})
    hotel_data = state.get("hotel_data", {})
    weather_data = state.get("weather_data", {})
    flight_data = state.get("flight_data", {})
    bus_data = state.get("bus_data", {})

    lines = [
        f"# 🇮🇳 Complete Itinerary: {origin} to {', '.join(destinations)}",
        f"**Dates**: {date_str} ({days} Days) | **Travelers**: {travelers} | **Budget**: {'₹' + f'{budget_val:,.0f}' if budget_val > 0 else 'Standard Recommendations'}\n",
        "---",
        "## 1. 🚆 Transport & Connectivity Options\n",
    ]

    # Train details
    trains = train_data.get("trains", [])
    if trains:
        lines.append(f"### Indian Railways (IRCTC) — Recommended Trains:")
        for t in trains[:3]:
            classes_str = ", ".join([f"{c}: ₹{info['fare']} ({info['status']})" for c, info in t.get("classes", {}).items()])
            lines.append(
                f"- **{t['train_number']} {t['train_name']}** ({t['from_station']} ➔ {t['to_station']})\n"
                f"  - Departure: `{t['departure']}` | Arrival: `{t['arrival']}` | Duration: `{t['duration']}` | Runs: {t.get('runs_on', 'Daily')}\n"
                f"  - Class Fares & Live Status: {classes_str}"
            )
        lines.append("")

    # Driving details
    if driving_data.get("distance_km"):
        fuel = driving_data.get("estimated_fuel_cost_inr", {})
        routes = ", ".join(driving_data.get("recommended_routes", []))
        lines.append(
            f"### 🚗 Self-Drive / Taxi Option via {routes}:\n"
            f"- **Distance**: {driving_data['distance_km']} km | **Driving Time**: ~{driving_data.get('driving_duration', '4h')}\n"
            f"- **Estimated Fuel**: ₹{fuel.get('petrol', 0):,.0f} (Petrol) / ₹{fuel.get('diesel', 0):,.0f} (Diesel)\n"
            f"- **FASTag Tolls**: ~₹{driving_data.get('estimated_toll_cost_inr', 0):,.0f}\n"
            f"- **Advisory**: {driving_data.get('road_condition', 'Standard National Highway')}\n"
        )

    # Flights if applicable
    if flight_data.get("flights"):
        lines.append("### ✈️ Domestic Flight Option:")
        for f in flight_data["flights"][:2]:
            lines.append(f"- **{f['airline']} {f['flight_number']}** ({f['stops']}) — Dep: `{f['departure']}` | Arr: `{f['arrival']}` | Fare: **₹{f['fare_inr']:,}**")
        lines.append("")

    # Buses
    if bus_data.get("buses"):
        lines.append("### 🚌 Intercity Bus Option:")
        for b in bus_data["buses"][:2]:
            lines.append(f"- **{b['operator']}** ({b['bus_type']}) — Dep: `{b['departure']}` | Fare: **₹{b['fare_inr']:,}** | Boarding: {b['boarding_point']}")
        lines.append("")

    # Weather
    if weather_data.get("temperature_range"):
        alerts = " ".join(weather_data.get("travel_alerts", []))
        lines.extend([
            "---",
            f"## 2. 🌤️ Weather Advisory for {primary_dest}\n",
            f"- **Temperature**: {weather_data['temperature_range']} | **Condition**: {weather_data['condition']}\n",
            f"- **Rain Probability**: {weather_data.get('precipitation_probability', '10%')} | **Humidity**: {weather_data.get('humidity', '55%')}\n",
            f"- **Alerts**: {alerts if alerts else 'Favorable travel conditions.'}\n",
        ])

    # Hotels
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

    # Day by day itinerary
    lines.extend([
        "---",
        f"## 4. 📅 Day-by-Day Sightseeing Itinerary ({days} Days)\n",
    ])
    for day_i in range(1, days + 1):
        if day_i == 1:
            lines.append(
                f"### Day 1: Arrival, Heritage Landmarks & Sunset Views\n"
                f"- **Morning (09:30 - 13:00)**: Arrival at {primary_dest}, check-in at hotel, refresh with authentic local breakfast.\n"
                f"- **Afternoon (14:00 - 17:00)**: Explore the primary palace / fortress complex and museum galleries.\n"
                f"- **Evening (17:30 - 20:30)**: Sunset viewing from the highest fortress viewpoint followed by a traditional dinner.\n"
            )
        elif day_i == 2:
            lines.append(
                f"### Day 2: Cultural Immersion, Local Bazaars & Traditional Dining\n"
                f"- **Morning (08:30 - 12:30)**: Visit architectural wonders, stepwells, or sacred temple complexes.\n"
                f"- **Afternoon (13:30 - 16:30)**: Walking tour through historic old-city alleys and handicraft bazaars.\n"
                f"- **Evening (17:30 - 21:00)**: Traditional folk performance or aarti ceremony, followed by iconic regional specialties.\n"
            )
        else:
            lines.append(
                f"### Day {day_i}: Scenic Excursion & Souvenir Shopping\n"
                f"- **Morning (09:00 - 13:00)**: Short excursion to nearby artisan villages, botanical gardens, or adventure lakes.\n"
                f"- **Afternoon (14:00 - 16:30)**: Pick up souvenirs, local spices, and traditional sweets for family back home.\n"
                f"- **Evening (17:00 onwards)**: Departure journey via scheduled train or highway return.\n"
            )

    # Local Knowledge & Attributions
    rag_results = state.get("rag_results", [])
    if rag_results:
        lines.extend([
            "---",
            "## 5. 💡 Local Tips & Etiquette (from Yatra AI Knowledge Base)\n",
            "- Dress respectfully when visiting temples and stepwells (cover shoulders and knees; footwear removal required).\n",
            "- Always drink sealed packaged mineral water from reputed brands or verified RO counters.\n",
            "- When booking IRCTC Tatkal tickets, note that AC Tatkal opens at 10:00 AM and Non-AC at 11:00 AM.\n",
            f"📚 *Sources: Yatra AI Curated Knowledge Base ({primary_dest.lower()}_guide.md, indian_railways_classes.md)*\n",
        ])

    final_plan_text = "\n".join(lines)
    state["final_plan"] = final_plan_text
    state.setdefault("messages", []).append({"role": "assistant", "content": final_plan_text})
    return state
