"""Hardening and resilience tests: error recovery, database auto-rebuild, and fallbacks."""

import json
from pathlib import Path
import pytest
from yatra_agent.graph.nodes import _fallback_extract_info
from yatra_agent.tools.mcp_tools import MCPToolManager
from yatra_agent.tools.rag_tool import TravelRAGTool


def test_chromadb_corruption_auto_rebuild(tmp_path):
    """Verify TravelRAGTool detects corrupted database directory and automatically rebuilds."""
    corrupt_dir = tmp_path / "corrupt_chroma"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    # Write garbage to simulate corrupted sqlite/index file
    bad_file = corrupt_dir / "chroma.sqlite3"
    bad_file.write_text("CORRUPTED_BINARY_DATA_GARBAGE", encoding="utf-8")

    # Initializing TravelRAGTool with corrupt dir should recover gracefully without crashing
    rag = TravelRAGTool(persist_dir=str(corrupt_dir))
    assert rag.collection is not None

    # Verify collection accepts new text after recovery
    added = rag.add_text("Shimla is the summer capital of Himachal Pradesh.", source="shimla_guide.md")
    assert added >= 1

    results = rag.search("Shimla", top_k=1)
    assert len(results) >= 1
    assert "Shimla" in results[0].page_content


def test_mcp_unreachable_server_fallback():
    """Verify MCPToolManager falls back to authentic domestic engine when server is offline."""
    mcp = MCPToolManager()
    # Call tool on offline server
    res = mcp._execute_domestic_engine("Offline Irctc", "train_search", {
        "origin": "Delhi",
        "destination": "Jaipur",
    })
    assert "trains" in res
    assert len(res["trains"]) >= 1
    assert res["trains"][0]["from_station"] == "NDLS"


def test_malformed_json_fallback_extraction():
    """Verify fallback parser handles messy, punctuation-heavy and Hindi travel requests."""
    text = "Plz help!!! from Dilli -> Jaipur... for 3 days starting kal with 8k budget??"
    data = _fallback_extract_info(text)
    assert data["origin"] == "Delhi"
    assert "Jaipur" in data["destinations"]
    assert data["num_days"] == 3
    assert data["budget"] == 8000.0


def test_mcp_directions_fuel_and_toll_calculation():
    """Verify domestic driving engine calculates petrol/diesel fuel costs and FASTag tolls."""
    mcp = MCPToolManager()
    directions = mcp._execute_domestic_engine("Google Maps Server", "get_directions", {
        "origin": "Delhi",
        "destination": "Jaipur",
    })
    assert directions["distance_km"] == 280
    assert "petrol" in directions["estimated_fuel_cost_inr"]
    assert "diesel" in directions["estimated_fuel_cost_inr"]
    assert directions["estimated_toll_cost_inr"] > 0
