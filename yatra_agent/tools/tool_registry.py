"""Tool definitions and LangChain tool adapters for Yatra AI."""

import asyncio
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from yatra_agent.tools.mcp_tools import MCPToolManager, ToolDefinition
from yatra_agent.tools.rag_tool import TravelRAGTool

# Global manager instances
_mcp_manager = MCPToolManager()
_rag_tool = TravelRAGTool()


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

    trains = res.get("trains", [])
    if not trains:
        return f"No direct trains found between {origin} and {destination} for {travel_date}."

    lines = [f"### 🚆 IRCTC Train Options ({res['origin_city']} ➔ {res['destination_city']} on {travel_date}):"]
    for t in trains:
        classes_str = ", ".join([f"{c}: ₹{info['fare']} ({info['status']})" for c, info in t.get("classes", {}).items()])
        lines.append(
            f"- **{t['train_number']} {t['train_name']}** ({t['from_station']} ➔ {t['to_station']})\n"
            f"  Dep: `{t['departure']}` | Arr: `{t['arrival']}` | Duration: `{t['duration']}` | Runs: {t.get('runs_on', 'Daily')}\n"
            f"  Fares & Availability: {classes_str}"
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

    fuel = res.get("estimated_fuel_cost_inr", {})
    routes = ", ".join(res.get("recommended_routes", []))
    return (
        f"### 🚗 Road Driving Route ({res['origin']} ➔ {res['destination']}):\n"
        f"- **Distance**: {res['distance_km']} km via {routes}\n"
        f"- **Driving Duration**: {res['driving_duration']} (non-stop)\n"
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

    hotels = res.get("hotels", [])
    if not hotels:
        return f"No hotels found in {location}."

    lines = [f"### 🏨 Recommended Accommodations in {res['location']} (Target Budget: ₹{budget_per_night:,.0f}/night):"]
    for h in hotels:
        amenities = ", ".join(h.get("amenities", []))
        budget_tag = "✅ Within Budget" if h.get("within_budget") else "⭐ Premium Upgrade"
        lines.append(
            f"- **{h['name']}** ({h.get('category', 'Hotel')}) — {budget_tag}\n"
            f"  Price: **₹{h['price_per_night']:,}/night** (Total stay for {num_nights} nights: ₹{h.get('total_cost_for_stay', 0):,})\n"
            f"  Rating: ⭐ {h['rating']}/5 | Neighborhood: {h['location']}\n"
            f"  Amenities: {amenities}"
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

    alerts = " ".join(res.get("travel_alerts", []))
    return (
        f"### 🌤️ Weather Forecast for {res['city']} ({res['date']}):\n"
        f"- **Temperature Range**: {res['temperature_range']}\n"
        f"- **Conditions**: {res['condition']}\n"
        f"- **Rain Probability**: {res['precipitation_probability']} | Humidity: {res.get('humidity', '50%')}\n"
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

    flights = res.get("flights", [])
    if not flights:
        return f"No direct flights found between {origin} and {destination}."

    lines = [f"### ✈️ Domestic Flight Options ({res['origin']} ➔ {res['destination']} on {date}):"]
    for f in flights:
        lines.append(
            f"- **{f['airline']} {f['flight_number']}** ({f['stops']})\n"
            f"  Dep: `{f['departure']}` | Arr: `{f['arrival']}` | Duration: `{f['duration']}` | Fare: **₹{f['fare_inr']:,}**"
        )
    return "\n".join(lines)


@tool
def bus_search(origin: str, destination: str, date: str = "15-12-2026") -> str:
    """Search intercity AC Volvo and Sleeper buses (IntrCity, Zingbus, State RTC) between Indian cities."""
    res = _run_sync(
        _mcp_manager.call_tool("Bus Server", "search_buses", {
            "origin": origin,
            "destination": destination,
            "date": date,
        })
    )

    buses = res.get("buses", [])
    if not buses:
        return f"No intercity buses found between {origin} and {destination}."

    lines = [f"### 🚌 Intercity Bus Options ({res['origin']} ➔ {res['destination']} on {date}):"]
    for b in buses:
        lines.append(
            f"- **{b['operator']}** ({b['bus_type']})\n"
            f"  Dep: `{b['departure']}` | Arr: `{b['arrival']}` | Duration: `{b['duration']}` | Fare: **₹{b['fare_inr']:,}**\n"
            f"  Boarding: {b['boarding_point']} | Dropping: {b['drop_point']}"
        )
    return "\n".join(lines)


@tool
def rag_knowledge_search(query: str, top_k: int = 3) -> str:
    """Search the Yatra AI vector knowledge base for destination guides, temple etiquette, Indian railway booking tips, and local food specialties."""
    docs = _rag_tool.search(query, top_k=top_k)
    if not docs:
        return f"No travel guides found in knowledge base for query: '{query}'."

    lines = [f"### 📚 Curated Knowledge Base Insights for '{query}':"]
    for d in docs:
        src = d.metadata.get("source", "guide.md")
        lines.append(f"#### Source: `{src}`\n{d.page_content}\n")
    return "\n".join(lines)


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
