"""LangChain ReAct Agent for executing domestic Indian travel tools."""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from yatra_agent.config.prompts import PLANNER_SYSTEM_PROMPT
from yatra_agent.config.settings import (
    LONG_DISTANCE_FLIGHT_THRESHOLD,
    PRIMARY_MODEL,
    PRIMARY_MODEL_TEMPERATURE,
)
from yatra_agent.graph.state import TravelPlanState
from yatra_agent.tools.mcp_tools import MCPToolManager
from yatra_agent.tools.tool_registry import (
    bus_search,
    flight_search,
    get_all_tools,
    google_maps_directions,
    hotel_search,
    irctc_train_search,
    rag_knowledge_search,
    weather_forecast,
)

logger = logging.getLogger("yatra_ai.agent")


def build_planning_agent(api_key: Optional[str] = None):
    """Assemble LangChain ReAct agent bound with all 5 MCP travel tools + RAG knowledge retrieval."""
    from yatra_agent.core.llm_factory import get_primary_llm

    llm = get_primary_llm(temperature=PRIMARY_MODEL_TEMPERATURE)
    if not llm:
        return None

    try:
        tools = get_all_tools()
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=PLANNER_SYSTEM_PROMPT,
        )
        return agent
    except Exception as e:
        logger.warning(f"Could not build LangChain ReAct agent: {e}")
        return None


def execute_travel_planning_agent(state: TravelPlanState) -> TravelPlanState:
    """Execute tools for IRCTC trains, road routes, hotels, weather, and RAG knowledge."""
    origin = state.get("origin", "Delhi")
    destinations = state.get("destinations", ["Jaipur"])
    primary_dest = destinations[0] if destinations else "Jaipur"
    travel_date = state.get("departure_date", "15-12-2026") or "15-12-2026"
    num_days = state.get("num_days", 3) or 3
    budget = state.get("budget", 0.0)
    budget_per_night = (budget / num_days) * 0.45 if budget > 0 else 2500.0

    # 1. Direct Live API Tool Execution (Amadeus, Google Maps/Places, WeatherAPI, Buses)
    mcp = MCPToolManager()

    async def _fetch_all_travel_data():
        if len(destinations) <= 1:
            # Single destination flow
            train_task = mcp.call_tool("IRCTC Server", "train_search", {
                "origin": origin,
                "destination": primary_dest,
                "travel_date": travel_date,
            })
            maps_task = mcp.call_tool("Google Maps Server", "get_directions", {
                "origin": origin,
                "destination": primary_dest,
            })
            hotel_task = mcp.call_tool("Google Maps Server", "hotel_search", {
                "location": primary_dest,
                "budget_per_night": budget_per_night,
                "num_nights": num_days,
            })
            weather_task = mcp.call_tool("Weather Server", "get_forecast", {
                "city": primary_dest,
                "date": travel_date,
            })
            bus_task = mcp.call_tool("Bus Server", "search_buses", {
                "origin": origin,
                "destination": primary_dest,
                "date": travel_date,
            })
            flight_task = mcp.call_tool("Flight Server", "search_flights", {
                "origin": origin,
                "destination": primary_dest,
                "date": travel_date,
            })

            res = await asyncio.gather(
                train_task, maps_task, hotel_task, weather_task, bus_task, flight_task,
                return_exceptions=True,
            )
            return [r if isinstance(r, dict) else {} for r in res]
        else:
            # Multi-city circuit planning
            legs = []
            curr = origin
            for d in destinations:
                legs.append((curr, d))
                curr = d
            if curr != origin:
                legs.append((curr, origin))

            train_tasks = [
                mcp.call_tool("IRCTC Server", "train_search", {"origin": f_c, "destination": t_c, "travel_date": travel_date})
                for f_c, t_c in legs
            ]
            maps_tasks = [
                mcp.call_tool("Google Maps Server", "get_directions", {"origin": f_c, "destination": t_c})
                for f_c, t_c in legs
            ]
            hotel_tasks = [
                mcp.call_tool("Google Maps Server", "hotel_search", {"location": d, "budget_per_night": budget_per_night, "num_nights": max(1, num_days // len(destinations))})
                for d in destinations
            ]
            weather_tasks = [
                mcp.call_tool("Weather Server", "get_forecast", {"city": d, "date": travel_date})
                for d in destinations
            ]
            bus_task = mcp.call_tool("Bus Server", "search_buses", {"origin": legs[0][0], "destination": legs[0][1], "date": travel_date})
            flight_task = mcp.call_tool("Flight Server", "search_flights", {"origin": legs[0][0], "destination": legs[0][1], "date": travel_date})

            all_tasks = train_tasks + maps_tasks + hotel_tasks + weather_tasks + [bus_task, flight_task]
            all_res = await asyncio.gather(*all_tasks, return_exceptions=True)

            # Partition results
            n_legs = len(legs)
            n_dests = len(destinations)

            idx = 0
            train_results = [r for r in all_res[idx : idx + n_legs] if isinstance(r, dict)]
            idx += n_legs

            maps_results = [r for r in all_res[idx : idx + n_legs] if isinstance(r, dict)]
            idx += n_legs

            hotel_results = [r for r in all_res[idx : idx + n_dests] if isinstance(r, dict)]
            idx += n_dests

            weather_results = [r for r in all_res[idx : idx + n_dests] if isinstance(r, dict)]
            idx += n_dests

            bus_res = all_res[idx] if isinstance(all_res[idx], dict) else {}
            idx += 1
            flight_res = all_res[idx] if isinstance(all_res[idx], dict) else {}

            # Aggregate train results
            combined_trains = []
            for tr in train_results:
                combined_trains.extend(tr.get("trains", []))
            train_res = {"trains": combined_trains if combined_trains else train_results[0].get("trains", []) if train_results else []}

            # Aggregate maps results
            total_km = sum(m.get("distance_km", 0) for m in maps_results)
            total_petrol = sum(m.get("estimated_fuel_cost_inr", {}).get("petrol", 0) for m in maps_results)
            total_diesel = sum(m.get("estimated_fuel_cost_inr", {}).get("diesel", 0) for m in maps_results)
            total_tolls = sum(m.get("estimated_toll_cost_inr", 0) for m in maps_results)
            all_routes = []
            for m in maps_results:
                all_routes.extend(m.get("recommended_routes", []))

            maps_res = {
                "distance_km": total_km,
                "driving_duration": f"{max(3, total_km // 65)}h total driving",
                "recommended_routes": list(dict.fromkeys(all_routes)),
                "estimated_fuel_cost_inr": {"petrol": total_petrol, "diesel": total_diesel},
                "estimated_toll_cost_inr": total_tolls,
                "road_condition": "Multi-highway circuit connecting national corridors",
            }

            # Aggregate hotel results
            combined_hotels = []
            for hr in hotel_results:
                combined_hotels.extend(hr.get("hotels", []))
            hotel_res = {"hotels": combined_hotels}

            # Aggregate weather results
            combined_alerts = []
            temp_ranges = []
            for wr in weather_results:
                if wr.get("travel_alerts"):
                    combined_alerts.extend(wr.get("travel_alerts"))
                if wr.get("temperature_range"):
                    temp_ranges.append(f"{wr.get('city', '')}: {wr.get('temperature_range')}")

            weather_res = {
                "temperature_range": "; ".join(temp_ranges) if temp_ranges else weather_results[0].get("temperature_range", "22°C - 32°C"),
                "condition": weather_results[0].get("condition", "Partly Cloudy") if weather_results else "Pleasant",
                "humidity": "55%",
                "precipitation_probability": "15%",
                "travel_alerts": list(dict.fromkeys(combined_alerts)),
            }

            return [train_res, maps_res, hotel_res, weather_res, bus_res, flight_res]

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                raw_results = executor.submit(asyncio.run, _fetch_all_travel_data()).result()
        else:
            raw_results = asyncio.run(_fetch_all_travel_data())
    except Exception as e:
        logger.error(f"Error executing async travel data tasks: {e}")
        raw_results = [{}, {}, {}, {}, {}, {}]

    train_res, maps_res, hotel_res, weather_res, bus_res, flight_res = raw_results

    state["train_data"] = train_res
    state["driving_data"] = maps_res
    state["hotel_data"] = hotel_res
    state["weather_data"] = weather_res
    state["bus_data"] = bus_res

    # Check conditional flight threshold
    dist = maps_res.get("distance_km", 0)
    if dist >= LONG_DISTANCE_FLIGHT_THRESHOLD or state.get("travel_mode_pref") == "flight":
        state["flight_data"] = flight_res
    else:
        state["flight_data"] = {}

    # Query RAG for the destinations
    try:
        from yatra_agent.tools.rag_tool import TravelRAGTool
        rag = TravelRAGTool()
        docs = rag.search(f"{primary_dest} travel guide attractions cuisine", top_k=3)
        if docs:
            state["rag_results"] = [d.page_content for d in docs]
    except Exception as e:
        logger.warning(f"RAG search error during agent execution: {e}")

    # 2. Invoke LangChain ReAct Agent for autonomous synthesis if primary LLM is available
    agent = build_planning_agent()
    if agent:
        try:
            instruction = (
                f"Plan a complete trip from {origin} to {', '.join(destinations)} starting on {travel_date} "
                f"for {num_days} days with a total budget of ₹{budget:,.0f}. "
                f"Query trains, driving routes, hotels, weather, and local knowledge base to verify live details."
            )
            agent_res = agent.invoke({"messages": [HumanMessage(content=instruction)]})
            logger.info("LangChain ReAct agent finished execution successfully.")
        except Exception as e:
            logger.warning(f"ReAct agent invocation exception: {e}")

    return state
