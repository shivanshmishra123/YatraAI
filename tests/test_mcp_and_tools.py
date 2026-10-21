"""Unit tests for Phase 3: MCP Tool Manager and LangChain Tool Registry."""

import pytest
from yatra_agent.tools.mcp_tools import MCPToolManager
from yatra_agent.tools.tool_registry import (
    bus_search,
    flight_search,
    google_maps_directions,
    hotel_search,
    irctc_train_search,
    weather_forecast,
)


@pytest.mark.asyncio
async def test_mcp_irctc_train_search():
    """Verify IRCTC tool extracts trains with station codes, fares, and classes."""
    mcp = MCPToolManager()
    res = await mcp.call_tool("IRCTC Server", "train_search", {
        "origin": "Delhi",
        "destination": "Jaipur",
        "travel_date": "15-12-2026",
    })

    assert res["status"] == "success"
    assert res["origin_city"] == "Delhi"
    assert res["destination_city"] == "Jaipur"
    assert res["total_trains_found"] >= 1

    first_train = res["trains"][0]
    assert "train_number" in first_train
    assert "classes" in first_train
    # Check CC or 3A class fare
    assert any(c in first_train["classes"] for c in ("CC", "3A", "2A", "1A", "SL"))


@pytest.mark.asyncio
async def test_mcp_google_maps_directions():
    """Verify Google Maps calculates distance, fuel costs in INR, and driving duration."""
    mcp = MCPToolManager()
    res = await mcp.call_tool("Google Maps Server", "get_directions", {
        "origin": "Delhi",
        "destination": "Jaipur",
    })

    assert res["status"] == "success"
    assert res["distance_km"] == 280
    assert "h" in res["driving_duration"]
    assert res["estimated_fuel_cost_inr"]["petrol"] > 1000.0
    assert res["estimated_toll_cost_inr"] > 0


@pytest.mark.asyncio
async def test_mcp_hotel_search():
    """Verify Hotel search returns accommodations categorized with prices and ratings."""
    mcp = MCPToolManager()
    res = await mcp.call_tool("Google Maps Server", "hotel_search", {
        "location": "Jaipur",
        "budget_per_night": 2000.0,
        "num_nights": 3,
    })

    assert res["status"] == "success"
    assert len(res["hotels"]) >= 1
    first_hotel = res["hotels"][0]
    assert first_hotel["price_per_night"] > 0
    assert first_hotel["rating"] >= 4.0
    assert "total_cost_for_stay" in first_hotel


@pytest.mark.asyncio
async def test_mcp_weather_forecast():
    """Verify Weather tool supplies temperatures, conditions, and Indian seasonal advisories."""
    mcp = MCPToolManager()
    # Test Goa in Monsoon
    res_goa = await mcp.call_tool("Weather Server", "get_forecast", {
        "city": "Goa",
        "date": "15-07-2026",
    })
    assert "Monsoon" in res_goa["condition"] or "🌧️" in " ".join(res_goa["travel_alerts"])

    # Test Delhi in Winter
    res_delhi = await mcp.call_tool("Weather Server", "get_forecast", {
        "city": "Delhi",
        "date": "15-12-2026",
    })
    assert "°C" in res_delhi["temperature_range"]


def test_langchain_tools_callable():
    """Verify LangChain tool wrappers return structured markdown strings."""
    train_output = irctc_train_search.invoke({"origin": "Delhi", "destination": "Jaipur"})
    assert "IRCTC Train Options" in train_output
    assert "₹" in train_output

    maps_output = google_maps_directions.invoke({"origin": "Delhi", "destination": "Jaipur"})
    assert "Road Driving Route" in maps_output
    assert "km" in maps_output

    hotel_output = hotel_search.invoke({"location": "Jaipur", "budget_per_night": 2500.0})
    assert "Recommended Accommodations" in hotel_output

    weather_output = weather_forecast.invoke({"city": "Jaipur", "date": "15-12-2026"})
    assert "Weather Forecast" in weather_output
