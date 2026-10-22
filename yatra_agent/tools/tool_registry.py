"""Tool definitions and LangChain tool adapters for Yatra AI."""

import asyncio
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from yatra_agent.tools.mcp_tools import MCPToolManager, ToolDefinition
from yatra_agent.tools.rag_tool import TravelRAGTool

# Global manager instances
_mcp_manager = MCPToolManager()
_rag_tool_instance = None


def _get_rag_tool() -> TravelRAGTool:
    global _rag_tool_instance
    if _rag_tool_instance is None:
        _rag_tool_instance = TravelRAGTool()
    return _rag_tool_instance


def _run_sync(coro):
    """Safely execute an async coroutine from synchronous code in any event loop context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# LangChain Tool Implementations
# ---------------------------------------------------------------------------
@tool
def irctc_train_search(origin: str, destination: str, travel_date: str = "15-12-2026", quota: str = "GN") -> str:
    """Search Indian Railways (IRCTC) trains, schedules, class fares (1A, 2A, 3A, 3E, SL), and seat availability between two Indian cities."""
    res = _run_sync(
        _mcp_manager.call_tool("IRCTC Server", "train_search", {
            "origin": origin,
            "destination": destination,
            "travel_date": travel_date,
            "quota": quota,
        })
    )

    if res.get("status") == "error":
        return f"⚠️ Train search API notice: {res.get('message')}"

    trains = res.get("trains", [])
    if not trains:
        return f"No direct trains found between {origin} and {destination} for {travel_date}."

    lines = [f"### 🚆 IRCTC Train Options ({res.get('origin_city', origin)} ➔ {res.get('destination_city', destination)} on {travel_date}):"]
    for t in trains:
        classes_str = ", ".join([f"{c}: ₹{info.get('fare', 0)} ({info.get('status', 'Available')})" for c, info in t.get("classes", {}).items()])
        lines.append(
            f"- **{t.get('train_number')} {t.get('train_name')}** ({t.get('from_station')} ➔ {t.get('to_station')})\n"
            f"  Dep: `{t.get('departure')}` | Arr: `{t.get('arrival')}` | Duration: `{t.get('duration', 'N/A')}` | Runs: {t.get('runs_on', 'Daily')}\n"
            f"  Fares & Availability: {classes_str or 'Check Amadeus/IRCTC'}"
        )
    return "\n".join(lines)


@tool
def google_maps_directions(origin: str, destination: str) -> str:
    """Query driving route distance, highway driving duration, petrol/diesel fuel costs in INR, and FASTag toll estimates between Indian cities."""
    res = _run_sync(
        _mcp_manager.call_tool("Google Maps Server", "get_directions", {
            "origin": origin,
            "destination": destination,
        })
    )

    if res.get("status") == "error":
        return f"⚠️ Directions API notice: {res.get('message')}"

    fuel = res.get("estimated_fuel_cost_inr", {})
    routes = ", ".join(res.get("recommended_routes", []))
    return (
        f"### 🚗 Road Driving Route ({res.get('origin', origin)} ➔ {res.get('destination', destination)}):\n"
        f"- **Distance**: {res.get('distance_km')} km via {routes}\n"
        f"- **Driving Duration**: {res.get('driving_duration')} (non-stop)\n"
        f"- **Estimated Petrol Fuel Cost**: ₹{fuel.get('petrol', 0):,.2f} (@ ₹100/L, {fuel.get('mileage_assumed_kmpl', 15)} km/L)\n"
        f"- **Estimated Diesel Fuel Cost**: ₹{fuel.get('diesel', 0):,.2f} (@ ₹95/L)\n"
        f"- **Estimated FASTag Tolls**: ₹{res.get('estimated_toll_cost_inr', 0):,.0f}\n"
        f"- **Highway Advisory**: {res.get('road_condition', 'Standard National Highway conditions')}"
    )


@tool
def hotel_search(location: str, budget_per_night: float = 2500.0, num_nights: int = 2) -> str:
    """Search hotels, heritage havelis, and backpacker hostels by location, nightly budget, and rating in Indian cities."""
    res = _run_sync(
        _mcp_manager.call_tool("Google Maps Server", "hotel_search", {
            "location": location,
            "budget_per_night": budget_per_night,
            "num_nights": num_nights,
        })
    )

    if res.get("status") == "error":
        return f"⚠️ Hotel search API notice: {res.get('message')}"

    hotels = res.get("hotels", [])
    if not hotels:
        return f"No hotels found in {location}."

    lines = [f"### 🏨 Recommended Accommodations in {res.get('location', location)} (Target Budget: ₹{budget_per_night:,.0f}/night):"]
    for h in hotels:
        amenities = ", ".join(h.get("amenities", []))
        budget_tag = "✅ Within Budget" if h.get("within_budget") else "⭐ Premium Upgrade"
        lines.append(
            f"- **{h.get('name')}** ({h.get('category', 'Hotel')}) — {budget_tag}\n"
            f"  Price: **₹{h.get('price_per_night', 0):,}/night** (Total stay for {num_nights} nights: ₹{h.get('total_cost_for_stay', 0):,})\n"
            f"  Rating: ⭐ {h.get('rating', 'N/A')}/5 | Neighborhood: {h.get('location', location)}\n"
            f"  Amenities: {amenities or 'WiFi, AC, Breakfast'}"
        )
    return "\n".join(lines)


@tool
def weather_forecast(city: str, date: str = "current") -> str:
    """Get weather forecast, temperatures, rainfall probability, and seasonal advisories (monsoon, fog, heatwave) for an Indian destination."""
    res = _run_sync(
        _mcp_manager.call_tool("Weather Server", "get_forecast", {
            "city": city,
            "date": date,
        })
    )

    if res.get("status") == "error":
        return f"⚠️ Weather API notice: {res.get('message')}"

    alerts = " ".join(res.get("travel_alerts", []))
    return (
        f"### 🌤️ Weather Forecast for {res.get('city', city)} ({res.get('date', date)}):\n"
        f"- **Temperature Range**: {res.get('temperature_range')}\n"
        f"- **Conditions**: {res.get('condition')}\n"
        f"- **Rain Probability**: {res.get('precipitation_probability')} | Humidity: {res.get('humidity', '50%')}\n"
        f"- **Advisories**: {alerts if alerts else 'No severe weather alerts in effect.'}"
    )


@tool
def flight_search(origin: str, destination: str, date: str = "15-12-2026") -> str:
    """Search domestic Indian flights, timings, IATA codes, and ticket fares (IndiGo, Air India, Akasa Air). Trigger when distance > 500km or requested."""
    res = _run_sync(
        _mcp_manager.call_tool("Flight Server", "search_flights", {
            "origin": origin,
            "destination": destination,
            "date": date,
        })
    )

    if res.get("status") == "error":
        return f"⚠️ Flight search API notice: {res.get('message')}"

    flights = res.get("flights", [])
    if not flights:
        return f"No direct flights found between {origin} and {destination}."

    lines = [f"### ✈️ Domestic Flight Options ({res.get('origin', origin)} ➔ {res.get('destination', destination)} on {date}):"]
    for f in flights:
        lines.append(
            f"- **{f.get('airline')} {f.get('flight_number')}** ({f.get('stops')})\n"
            f"  Dep: `{f.get('departure')}` | Arr: `{f.get('arrival')}` | Duration: `{f.get('duration')}` | Fare: **₹{f.get('fare_inr', 0):,}**"
        )
    return "\n".join(lines)


@tool
def bus_search(origin: str, destination: str, date: str = "15-12-2026") -> str:
    """Search intercity AC Volvo and Sleeper buses (redBus, AbhiBus) between Indian cities."""
    res = _run_sync(
        _mcp_manager.call_tool("Bus Server", "search_buses", {
            "origin": origin,
            "destination": destination,
            "date": date,
        })
    )

    if res.get("status") == "error":
        return f"⚠️ Bus search notice: {res.get('message')}"

    booking_links = res.get("booking_links", [])
    if booking_links:
        lines = [f"### 🚌 Intercity Bus Platforms for {res.get('origin', origin)} ➔ {res.get('destination', destination)}:"]
        lines.append(res.get("message", ""))
        for bl in booking_links:
            lines.append(f"- **[{bl['platform']}]({bl['url']})**: {bl['description']}")
        return "\n".join(lines)

    buses = res.get("buses", [])
    if not buses:
        return f"No intercity buses found between {origin} and {destination}."

    lines = [f"### 🚌 Intercity Bus Options ({res.get('origin', origin)} ➔ {res.get('destination', destination)} on {date}):"]
    for b in buses:
        lines.append(
            f"- **{b.get('operator')}** ({b.get('bus_type')})\n"
            f"  Dep: `{b.get('departure')}` | Arr: `{b.get('arrival')}` | Duration: `{b.get('duration')}` | Fare: **₹{b.get('fare_inr', 0):,}**\n"
            f"  Boarding: {b.get('boarding_point')} | Dropping: {b.get('drop_point')}"
        )
    return "\n".join(lines)


@tool
def rag_knowledge_search(query: str, top_k: int = 3) -> str:
    """Search the Yatra AI vector knowledge base for destination guides, temple etiquette, Indian railway booking tips, and local food specialties."""
    try:
        rag = _get_rag_tool()
        docs = rag.search(query, top_k=top_k)
        if not docs:
            return f"No travel guides found in knowledge base for query: '{query}'."

        lines = [f"### 📚 Curated Knowledge Base Insights for '{query}':"]
        for d in docs:
            src = d.metadata.get("source", "guide.md")
            lines.append(f"#### Source: `{src}`\n{d.page_content}\n")
        return "\n".join(lines)
    except Exception as e:
        return f"Knowledge base search unavailable: {e}"


def get_all_tools():
    """Return all LangChain tool instances for binding to LangChain ReAct agents."""
    return [
        irctc_train_search,
        google_maps_directions,
        hotel_search,
        weather_forecast,
        flight_search,
        bus_search,
        rag_knowledge_search,
    ]
