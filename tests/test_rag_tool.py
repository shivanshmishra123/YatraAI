"""Unit tests for Phase 2: ChromaDB TravelRAGTool knowledge base."""

import pytest
from pathlib import Path
from yatra_agent.tools.rag_tool import TravelRAGTool


@pytest.fixture
def rag_tool(tmp_path):
    """Create a temporary TravelRAGTool instance for isolated testing."""
    tool = TravelRAGTool(persist_dir=str(tmp_path / "test_vectordb"))
    return tool


def test_rag_tool_initialization(rag_tool):
    """Verify ChromaDB collection initializes properly."""
    assert rag_tool.collection is not None
    stats = rag_tool.get_stats()
    assert stats["total_chunks"] == 0


def test_rag_add_and_search_text(rag_tool):
    """Verify adding plain text, chunking, and searching."""
    sample_text = (
        "# Jaipur Travel Guide\n\n"
        "Jaipur is the capital of Rajasthan, famous for Amber Fort, Hawa Mahal, and Dal Baati Churma. "
        "The best time to visit Jaipur is from October to March."
    )
    added = rag_tool.add_text(sample_text, source="test_jaipur.md", metadata={"destination": "Jaipur"})
    assert added >= 1

    stats = rag_tool.get_stats()
    assert stats["total_chunks"] >= 1
    assert "test_jaipur.md" in stats["sources"]

    # Search
    docs = rag_tool.search("What is Dal Baati Churma in Jaipur?", top_k=2)
    assert len(docs) >= 1
    assert "Amber Fort" in docs[0].page_content
    assert docs[0].metadata.get("source") == "test_jaipur.md"


def test_rag_idempotent_deduplication(rag_tool):
    """Verify re-adding identical text with same source produces identical MD5 IDs without duplicates."""
    text = "Exploring the backwaters of Alleppey in Kerala on a houseboat."
    c1 = rag_tool.add_text(text, source="kerala.md")
    count_after_first = rag_tool.get_stats()["total_chunks"]

    c2 = rag_tool.add_text(text, source="kerala.md")
    count_after_second = rag_tool.get_stats()["total_chunks"]

    assert count_after_first == count_after_second


def test_rag_delete_by_source(rag_tool):
    """Verify deleting chunks by document source."""
    rag_tool.add_text("Manali snow guide and Rohtang pass.", source="manali.md")
    assert "manali.md" in rag_tool.get_stats()["sources"]

    deleted = rag_tool.delete_by_source("manali.md")
    assert deleted == 1
    assert "manali.md" not in rag_tool.get_stats()["sources"]
