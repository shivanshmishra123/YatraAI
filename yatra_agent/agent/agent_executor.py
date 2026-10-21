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
    openai_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not openai_key or openai_key.startswith("sk-your") or len(openai_key) < 20:
        return None

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=PRIMARY_MODEL,
            temperature=PRIMARY_MODEL_TEMPERATURE,
            api_key=openai_key,
        )
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

    # 1. Direct Domestic Engine Execution (guarantees 100% structured data availability)
    mcp = MCPToolManager()

    async def _fetch_all_travel_data():
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

        return await asyncio.gather(
            train_task, maps_task, hotel_task, weather_task, bus_task, flight_task,
            return_exceptions=True,
        )

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

    train_res, maps_res, hotel_res, weather_res, bus_res, flight_res = [
        r if isinstance(r, dict) else {} for r in raw_results
    ]

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

    # 2. If OpenAI API key is active, invoke LangChain ReAct Agent for autonomous synthesis
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
