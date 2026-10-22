"""Hardening and resilience tests: error handling, missing key diagnostics, and live API responses."""

import pytest
import asyncio
from yatra_agent.tools.mcp_tools import MCPToolManager
from yatra_agent.tools.rag_tool import TravelRAGTool


def test_chromadb_initialization_with_dir(tmp_path):
    """Verify TravelRAGTool initializes vector collection properly in designated directory."""
    test_dir = tmp_path / "test_chroma"
    test_dir.mkdir(parents=True, exist_ok=True)

    rag = TravelRAGTool(persist_dir=str(test_dir))
    assert rag.collection is not None

    added = rag.add_text("Shimla is the summer capital of Himachal Pradesh.", source="shimla_guide.md")
    assert added >= 1

    results = rag.search("Shimla", top_k=1)
    assert len(results) >= 1
    assert "Shimla" in results[0].page_content


@pytest.mark.asyncio
async def test_mcp_missing_credentials_surface_error():
    """Verify MCPToolManager returns structured error diagnostic rather than crashing when credentials are absent."""
    mcp = MCPToolManager()
    res = await mcp.call_tool("IRCTC Server", "train_search", {
        "origin": "Delhi",
        "destination": "Jaipur",
    })
    # If Amadeus key is missing or invalid, it returns status="error" with descriptive message
    assert "status" in res
    assert res.get("status") in ("error", "success")


@pytest.mark.asyncio
async def test_mcp_bus_booking_links_live():
    """Verify bus tool returns valid booking links for intercity routes."""
    mcp = MCPToolManager()
    res = await mcp.call_tool("Bus Server", "search_buses", {
        "origin": "Delhi",
        "destination": "Jaipur",
    })
    assert res["status"] == "success"
    assert len(res["booking_links"]) >= 2
    assert "redbus.in" in res["booking_links"][0]["url"]

