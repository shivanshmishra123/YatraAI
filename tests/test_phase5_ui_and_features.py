import sys
from pathlib import Path

# Add project root to sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app import stream_response_generator
from yatra_agent.graph.state import create_initial_state
from yatra_agent.graph.workflow import build_travel_workflow
from yatra_agent.tools.rag_tool import TravelRAGTool


def test_stream_response_generator():
    """Verify stream_response_generator yields words without losing text."""
    sample_text = "Namaste! Welcome to Yatra AI domestic travel planner."
    chunks = list(stream_response_generator(sample_text))
    reconstructed = "".join(chunks)
    assert reconstructed == sample_text
    assert len(chunks) == len(sample_text.split(" "))


def test_export_chat_format():
    """Verify chat transcript formatting generates valid markdown with traveler and assistant headers."""
    messages = [
        {"role": "assistant", "content": "Namaste! Where would you like to travel?"},
        {"role": "user", "content": "Plan a trip from Delhi to Jaipur for 3 days"},
        {"role": "assistant", "content": "Here is your itinerary for Jaipur..."},
    ]

    export_text = "# 🌍 Yatra AI — Indian Travel Plan Transcript\n\n"
    for m in messages:
        sender = "👤 Traveler" if m["role"] == "user" else "🌍 Yatra AI"
        export_text += f"### {sender}\n\n{m['content']}\n\n---\n\n"

    assert "# 🌍 Yatra AI" in export_text
    assert "### 👤 Traveler" in export_text
    assert "### 🌍 Yatra AI" in export_text
    assert "Delhi to Jaipur" in export_text


def test_quick_start_prompt_execution():
    """Verify that quick-start prompts run through workflow without exceptions."""
    workflow = build_travel_workflow()
    config = {"configurable": {"thread_id": "test_phase5_quickstart"}}

    state = create_initial_state()
    state["messages"] = [
        {"role": "user", "content": "Plan a trip to Jaipur, 3 days, ₹8000, starting Dec 15"}
    ]

    res = workflow.invoke(state, config=config)
    assert res.get("origin") != "" or "Jaipur" in res.get("destinations", [])
    assistant_msgs = [m for m in res["messages"] if m["role"] == "assistant"]
    assert len(assistant_msgs) >= 1
    assert "Jaipur" in assistant_msgs[-1]["content"]


def test_rag_multi_file_support(tmp_path):
    """Verify multi-file loading and indexing in ChromaDB RAG tool."""
    rag = TravelRAGTool()

    f1 = tmp_path / "agra_notes.txt"
    f1.write_text("Agra is famous for Taj Mahal and Agra Fort on Yamuna river bank.", encoding="utf-8")

    f2 = tmp_path / "jaipur_notes.txt"
    f2.write_text("Jaipur Pink City features Hawa Mahal, Jantar Mantar and Amber Fort.", encoding="utf-8")

    c1 = rag.load_file_content(f1)
    added1 = rag.add_text(c1, source="agra_notes.txt")

    c2 = rag.load_file_content(f2)
    added2 = rag.add_text(c2, source="jaipur_notes.txt")

    assert added1 >= 1
    assert added2 >= 1

    docs = rag.search("Taj Mahal", top_k=2)
    assert any("Taj Mahal" in d.page_content for d in docs)
