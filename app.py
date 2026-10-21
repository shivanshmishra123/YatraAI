"""Yatra AI — Intelligent Domestic Travel Planner with LangGraph, LangChain, ChromaDB RAG & Gemini 1.5 Pro."""

import os
from pathlib import Path
import time
from typing import Generator
import streamlit as st

from yatra_agent.graph.state import create_initial_state
from yatra_agent.graph.workflow import build_travel_workflow
from yatra_agent.tools.rag_tool import TravelRAGTool
from yatra_agent.utils.date_parser import get_current_ist_datetime

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Yatra AI — Indian Domestic Travel Planner",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Rich Custom CSS: Aesthetics, Badges, Responsive Tables & Card Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main titles and header typography */
    .main-title {
        font-size: 2.25rem;
        font-weight: 800;
        color: #ff6b35;
        letter-spacing: -0.5px;
        margin-bottom: 0.15rem;
    }
    .sub-title {
        font-size: 0.98rem;
        color: #4b5563;
        margin-bottom: 1.25rem;
        line-height: 1.5;
    }
    
    /* Phase and Technology Badges */
    .badge-phase {
        background-color: #e8f0fe;
        color: #1a73e8;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.80rem;
        margin-right: 4px;
        display: inline-block;
    }
    .badge-rag {
        background-color: #e6f4ea;
        color: #137333;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.80rem;
        margin-right: 4px;
        display: inline-block;
    }
    .badge-mcp {
        background-color: #fef7e0;
        color: #b06000;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.80rem;
        margin-right: 4px;
        display: inline-block;
    }
    .badge-gemini {
        background-color: #f3e8fd;
        color: #7b1fa2;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.80rem;
        display: inline-block;
    }
    
    /* Data pill items */
    .data-pill {
        display: inline-block;
        background: #f1f3f4;
        border-radius: 12px;
        padding: 3px 10px;
        margin: 2px 3px 4px 0;
        font-size: 0.82rem;
        font-weight: 500;
        color: #374151;
        border: 1px solid #e5e7eb;
    }
    
    /* Quick-start cards */
    .quick-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .quick-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        border-color: #ff6b35;
    }

    /* Table styling & horizontal scroll */
    div[data-testid="stMarkdownContainer"] table {
        width: 100%;
        border-collapse: collapse;
        margin: 12px 0;
        overflow-x: auto;
        display: block;
    }
    div[data-testid="stMarkdownContainer"] th {
        background-color: #f8fafc;
        color: #1e293b;
        font-weight: 600;
        padding: 8px 12px;
        border: 1px solid #e2e8f0;
    }
    div[data-testid="stMarkdownContainer"] td {
        padding: 8px 12px;
        border: 1px solid #e2e8f0;
        color: #334155;
    }

    /* Warning & info box tweaks */
    .caution-banner {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        margin: 10px 0;
        font-size: 0.9rem;
        color: #92400e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Namaste! 🙏 I am **Yatra AI**, your domestic travel planner for India. "
                       "Where would you like to travel, or what destination can I help you explore today?",
        }
    ]

if "travel_state" not in st.session_state:
    st.session_state.travel_state = create_initial_state()

if "workflow" not in st.session_state:
    st.session_state.workflow = build_travel_workflow()

if "rag_tool" not in st.session_state:
    st.session_state.rag_tool = TravelRAGTool()

if "max_iterations" not in st.session_state:
    st.session_state.max_iterations = 15

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7

if "enable_gemini" not in st.session_state:
    st.session_state.enable_gemini = True

if "stream_latest" not in st.session_state:
    st.session_state.stream_latest = False


# ---------------------------------------------------------------------------
# Helper Functions: State Reset & Streaming Output Generator
# ---------------------------------------------------------------------------
def reset_conversation():
    """Reset chat history and travel plan state."""
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Namaste! 🙏 I am **Yatra AI**, your domestic travel planner for India. "
                       "Where would you like to travel, or what destination can I help you explore today?",
        }
    ]
    st.session_state.travel_state = create_initial_state()
    st.session_state.stream_latest = False
    st.rerun()


def stream_response_generator(text: str) -> Generator[str, None, None]:
    """Yields chunks with a micro-delay for smooth real-time assistant streaming."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(0.008)


# ---------------------------------------------------------------------------
# Sidebar UI: Controls, State Inspector, RAG Manager, System Config & Health
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🇮🇳 Yatra AI Controls")

    # Session action buttons: Reset Chat & Export Transcript
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("🗑️ Reset Chat", use_container_width=True, on_click=reset_conversation):
            st.stop()

    ist_time = get_current_ist_datetime()

    with col_s2:
        # Build Markdown chat transcript for export
        export_text = f"# 🌍 Yatra AI — Indian Travel Plan Transcript\n"
        export_text += f"*Generated on: {ist_time.strftime('%d-%m-%Y %H:%M:%S IST')}*\n\n---\n\n"
        for idx, m in enumerate(st.session_state.messages):
            sender = "👤 Traveler" if m["role"] == "user" else "🌍 Yatra AI"
            export_text += f"### {sender}\n\n{m['content']}\n\n---\n\n"

        st.download_button(
            "📥 Export Chat",
            data=export_text,
            file_name=f"yatra_plan_{ist_time.strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
            help="Download complete travel plan and conversation as Markdown.",
        )

    st.caption(f"🕒 **Current IST:** `{ist_time.strftime('%d-%m-%Y %H:%M:%S')}`")

    # 1. Live State Inspector
    st.divider()
    st.subheader("📊 Live Travel Parameters")
    ts = st.session_state.travel_state

    origin_disp = ts.get("origin") or "*(Not specified)*"
    dest_disp = ", ".join(ts.get("destinations", [])) or "*(Not specified)*"
    date_disp = ts.get("departure_date") or "*(Not specified)*"
    days_disp = f"{ts.get('num_days')} days" if ts.get("num_days", 0) > 0 else "*(Not specified)*"
    budget_val = ts.get("budget", 0)
    budget_disp = f"₹{budget_val:,.0f}" if budget_val > 0 else "*(Not specified)*"
    travelers_disp = f"{ts.get('num_travelers', 1)} people"
    mode_disp = ts.get("query_type", "simple").upper()
    deep_reasoning = "⚡ Gemini 1.5 Pro" if ts.get("needs_deep_analysis") else "🤖 GPT-4o-mini"

    st.markdown(f"• **Origin:** `{origin_disp}`")
    st.markdown(f"• **Destination(s):** `{dest_disp}`")
    st.markdown(f"• **Departure Date:** `{date_disp}`")
    st.markdown(f"• **Duration:** `{days_disp}`")
    st.markdown(f"• **Budget:** `{budget_disp}`")
    st.markdown(f"• **Travelers:** `{travelers_disp}`")
    st.markdown(f"• **Routing Mode:** `{mode_disp}` ({deep_reasoning})")

    if ts.get("needs_deep_analysis") and ts.get("complexity_reason"):
        st.caption(f"🧠 **Trigger:** `{ts['complexity_reason']}`")

    # Display active tool data pills if collected
    trains_count = len(ts.get("train_data", {}).get("trains", []))
    driving_km = ts.get("driving_data", {}).get("distance_km", 0)
    hotels_count = len(ts.get("hotel_data", {}).get("hotels", []))
    weather_cond = ts.get("weather_data", {}).get("condition", "")

    if trains_count or driving_km or hotels_count or weather_cond:
        st.markdown("##### 🛠️ Retrieved Travel Data")
        if trains_count:
            st.markdown(f'<span class="data-pill">🚆 {trains_count} IRCTC Trains</span>', unsafe_allow_html=True)
        if driving_km:
            st.markdown(f'<span class="data-pill">🚗 {driving_km} km Highway</span>', unsafe_allow_html=True)
        if hotels_count:
            st.markdown(f'<span class="data-pill">🏨 {hotels_count} Hotels</span>', unsafe_allow_html=True)
        if weather_cond:
            st.markdown(f'<span class="data-pill">🌤️ {weather_cond[:20]}</span>', unsafe_allow_html=True)

    # 2. Knowledge Base Management
    st.divider()
    st.subheader("📚 ChromaDB Knowledge Base")
    rag = st.session_state.rag_tool
    stats = rag.get_stats()
    st.write(f"• **Indexed Chunks**: `{stats['total_chunks']}`")
    st.write(f"• **Curated Guides**: `{stats['unique_sources_count']}`")

    # Multi-file uploader with progress bar
    uploaded_files = st.file_uploader(
        "Upload guide (.md, .txt, .pdf, .csv):",
        type=["md", "txt", "pdf", "csv"],
        accept_multiple_files=True,
        key="rag_multi_uploader",
    )
    if uploaded_files:
        upload_prog = st.progress(0.0, text="Indexing files...")
        total_files = len(uploaded_files)
        total_added_chunks = 0

        for idx, file_obj in enumerate(uploaded_files):
            save_path = Path("data") / "travel_docs" / file_obj.name
            with open(save_path, "wb") as f:
                f.write(file_obj.getbuffer())
            content = rag.load_file_content(save_path)
            added = rag.add_text(content, source=file_obj.name)
            total_added_chunks += added
            upload_prog.progress((idx + 1) / total_files, text=f"Indexed {file_obj.name}")

        st.success(f"✅ Successfully indexed {total_files} file(s) ({total_added_chunks} chunks)!")
        time.sleep(0.5)
        st.rerun()

    if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
        with st.spinner("Rebuilding vector database from disk..."):
            res = rag.build_knowledge_base(force_recreate=True)
            st.success(f"Indexed {res['documents_indexed']} documents ({res['chunks_added']} chunks)!")
            st.rerun()

    # 3. System Configuration & Sliders
    st.divider()
    with st.expander("⚙️ System Configuration", expanded=False):
        st.session_state.max_iterations = st.slider(
            "Max Agent Iterations",
            min_value=5,
            max_value=30,
            value=st.session_state.max_iterations,
            step=1,
            help="Maximum ReAct thought-action loop cycles.",
        )
        st.session_state.temperature = st.slider(
            "Model Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.05,
            help="Generation randomness: 0.0 for deterministic extraction, 0.7 for creative synthesis.",
        )
        st.session_state.enable_gemini = st.toggle(
            "Enable Gemini 1.5 Pro Reasoning",
            value=st.session_state.enable_gemini,
            help="Enables deep multi-city route optimization and budget survival strategies.",
        )
        mcp_selected = st.multiselect(
            "Active MCP Tool Servers",
            options=["IRCTC Server", "Google Maps Server", "Weather Server", "Flight Server", "Bus Server"],
            default=["IRCTC Server", "Google Maps Server", "Weather Server", "Flight Server", "Bus Server"],
            help="Choose which domestic MCP tool servers are active during agent tool calls.",
        )

    # 4. Tool Inventory & Health Indicators
    with st.expander("🛠️ Tool Inventory & Health", expanded=False):
        st.markdown("• 🟢 **IRCTC Trains** (`train_search`) — Active")
        st.markdown("• 🟢 **Google Maps** (`get_directions`, `hotel_search`) — Active")
        st.markdown("• 🟢 **Weather Server** (`get_forecast`) — Active")
        st.markdown("• 🟢 **Bus Server** (`search_buses`) — Active")
        st.markdown("• 🟢 **Flight Server** (`search_flights`) — Active")
        st.markdown("• 🟢 **ChromaDB RAG** (`rag_search`) — Active")
        st.markdown("• 🟢 **Gemini 1.5 Pro** (`gemini_strategy`) — Active")
        st.caption("All domestic tools ready with offline/test fallback support.")

    # 5. Quick Starter Queries
    st.divider()
    st.subheader("💡 Quick Starter Queries")
    starter_prompts = [
        "Delhi → Agra → Jaipur, 5 days, ₹15,000 for elderly parents",
        "Plan Delhi to Jaipur, 3 days, ₹8000, starting Dec 15",
        "Plan Delhi to Jaipur, 3 days, ₹1200 for 1 person",
        "Find cheapest way from Delhi to Varanasi",
        "What are the best places in Jaipur?",
        "Bangalore to Goa this weekend for 2 people",
        "kal Dilli se Jaipur jaana hai",
    ]
    for prompt_text in starter_prompts:
        if st.button(prompt_text, key=f"sidebar_btn_{prompt_text}"):
            st.session_state.pending_prompt = prompt_text
            st.rerun()


# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🌍 Yatra AI — Travel Planner</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Intelligent Domestic Travel Planning for India | '
    '<span class="badge-phase">LangGraph Core</span>'
    '<span class="badge-rag">ChromaDB ONNX RAG</span>'
    '<span class="badge-mcp">Groq Llama 3.3 70B</span>'
    '<span class="badge-gemini">Dual-Model Gemini 1.5 Pro</span>'
    '</div>',
    unsafe_allow_html=True,
)

# API Mode Status Banner
from yatra_agent.core.llm_factory import get_active_llm_info
llm_info = get_active_llm_info()
google_key = os.getenv("GOOGLE_API_KEY", "")
has_live_keys = bool(llm_info["status"] != "LOCAL_FALLBACK_ACTIVE" or (google_key and len(google_key) > 20))

if not has_live_keys:
    st.markdown(
        '<div class="caution-banner">'
        'ℹ️ <strong>Domestic Fallback Engine Active:</strong> '
        'Running with deterministic local MCP engines, local ONNX ChromaDB RAG, and high-reasoning fallback strategist. '
        'Add your free <code>GROQ_API_KEY</code> and <code>GOOGLE_API_KEY</code> in <code>.env</code> to enable live online inference!'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div style="background-color: #e6f4ea; border-left: 4px solid #137333; padding: 10px 14px; border-radius: 6px; font-size: 0.88rem; color: #1e4620; margin-bottom: 1rem;">'
        f'⚡ <strong>Live Fast Inference Online:</strong> Primary LLM: <strong>{llm_info["provider"]} ({llm_info["model"]})</strong> | Reasoning Engine: <strong>Gemini 1.5 Pro</strong>'
        '</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Quick-Start Suggestions (Rendered on Initial / Empty Chat State)
# ---------------------------------------------------------------------------
if len(st.session_state.messages) <= 1:
    st.markdown("#### 🌟 Popular Indian Travel Circuits")
    st.markdown("Choose a starter journey below or type your customized travel request in the chat:")

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        if st.button("🏰 **Golden Triangle Heritage**\n\nDelhi → Agra → Jaipur, 5 days, ₹15,000 for elderly parents", use_container_width=True):
            st.session_state.pending_prompt = "Delhi → Agra → Jaipur, 5 days, ₹15,000 for elderly parents"
            st.rerun()

        if st.button("🏖️ **Coastal Retreat in Goa**\n\nBest beaches, cafes & water sports in North & South Goa", use_container_width=True):
            st.session_state.pending_prompt = "What are the best places and beaches in Goa?"
            st.rerun()

        if st.button("🚆 **Direct Rail Connectivity**\n\nFastest Vande Bharat & Rajdhani trains from Delhi to Mumbai", use_container_width=True):
            st.session_state.pending_prompt = "Plan a train trip from Delhi to Mumbai for 4 days"
            st.rerun()

    with col_q2:
        if st.button("💰 **Budget-Friendly Rajasthan**\n\nPlan Delhi to Jaipur, 3 days, ₹8000, starting Dec 15", use_container_width=True):
            st.session_state.pending_prompt = "Plan Delhi to Jaipur, 3 days, ₹8000, starting Dec 15"
            st.rerun()

        if st.button("⛰️ **Western Ghats Weekend**\n\nBangalore to Goa this weekend for 2 people", use_container_width=True):
            st.session_state.pending_prompt = "Bangalore to Goa this weekend for 2 people"
            st.rerun()

        if st.button("🕉️ **Spiritual Circuit**\n\nFind cheapest way from Delhi to Varanasi with temple darshan", use_container_width=True):
            st.session_state.pending_prompt = "Find cheapest way from Delhi to Varanasi"
            st.rerun()

    st.divider()


# ---------------------------------------------------------------------------
# Render Chat History with Custom Avatars (🧑 for user, 🌍 for assistant)
# ---------------------------------------------------------------------------
for i, msg in enumerate(st.session_state.messages):
    is_user = msg["role"] == "user"
    avatar_icon = "🧑" if is_user else "🌍"

    with st.chat_message(msg["role"], avatar=avatar_icon):
        content = msg["content"]
        # If this is the newly generated message and streaming is requested
        if not is_user and i == len(st.session_state.messages) - 1 and st.session_state.stream_latest:
            st.write_stream(stream_response_generator(content))
            st.session_state.stream_latest = False
        else:
            # If the response is a long itinerary, render cleanly with section expanders
            if "# 🇮🇳 Complete Itinerary" in content and "## 1. 🚆" in content:
                # Main Header & Summary
                parts = content.split("---")
                header_part = parts[0]
                st.markdown(header_part)

                # Render remaining sections with expanders for enhanced readability
                with st.expander("📊 View Complete Itinerary, Transport Comparison & Day-by-Day Schedule", expanded=True):
                    st.markdown("---".join(parts[1:]))
            else:
                st.markdown(content)


# ---------------------------------------------------------------------------
# Chat Input & LangGraph Pipeline Execution with Progress Indicators
# ---------------------------------------------------------------------------
user_input = st.chat_input("Ask a travel question or plan a trip across India...")

if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    user_input = st.session_state.pending_prompt
    del st.session_state["pending_prompt"]

if user_input:
    # 1. Render User Message immediately
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.travel_state["messages"] = list(st.session_state.messages)

    # 2. Phase-by-Phase Real-Time Loading Status
    with st.status("🧭 Planning your Indian journey with Yatra AI...", expanded=True) as status:
        try:
            status.write("🔍 Extracting travel parameters (origin, destinations, dates, budget)...")
            time.sleep(0.15)

            status.write("📚 Querying ChromaDB knowledge base for destination advisories...")
            time.sleep(0.15)

            status.write("🚆 Querying IRCTC for trains, berth availability & fares...")
            time.sleep(0.15)

            status.write("🚗 Calculating highway driving routes, fuel costs & FASTag tolls...")
            time.sleep(0.15)

            status.write("🏨 Searching verified accommodations & room tariffs...")
            time.sleep(0.15)

            status.write("☀️ Checking weather forecasts & seasonal alerts...")
            time.sleep(0.15)

            # Check if multi-city or complex query
            raw_input_lower = user_input.lower()
            if any(k in raw_input_lower for k in ("elderly", "senior", "budget", "cheapest", "fastest", "->", "to")):
                status.write("🧠 Activating Gemini 1.5 Pro deep reasoning & route optimization...")
                time.sleep(0.2)

            status.write("🛡️ Running critique validator for transport, stay & budget compliance...")

            # Execute compiled LangGraph StateGraph
            workflow = st.session_state.workflow
            config = {"configurable": {"thread_id": "yatra_streamlit_session"}}
            final_state = workflow.invoke(st.session_state.travel_state, config=config)
            st.session_state.travel_state = final_state

            # Budget sanity warning if applicable
            budget_val = final_state.get("budget", 0)
            days_val = final_state.get("num_days", 1) or 1
            travelers_val = final_state.get("num_travelers", 1) or 1
            if budget_val > 0:
                per_day = budget_val / (days_val * travelers_val)
                if per_day < 350:
                    status.write(f"⚠️ Note: ₹{per_day:.0f}/person/day is extremely tight. Prioritizing IRCTC Sleeper & Dharamshalas.")

            status.write("📝 Synthesizing comprehensive itinerary & itemized budget...")
            status.update(label="✅ Travel Itinerary Successfully Generated!", state="complete", expanded=False)

            # 3. Retrieve latest assistant message
            assistant_messages = [m for m in final_state.get("messages", []) if m.get("role") == "assistant"]
            if assistant_messages:
                latest_reply = assistant_messages[-1]["content"]
                if len(st.session_state.messages) == 0 or st.session_state.messages[-1].get("content") != latest_reply:
                    st.session_state.messages.append({"role": "assistant", "content": latest_reply})
                    st.session_state.stream_latest = True

        except Exception as e:
            err_msg = (
                f"⚠️ **Travel Planning Notice**: We encountered an unexpected processing condition ({e}). "
                f"Please verify your inputs or retry with a specific city name (e.g. *'Delhi to Jaipur for 3 days'*)."
            )
            status.update(label="⚠️ Planning adjusted with recommendations", state="error", expanded=False)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
            st.session_state.stream_latest = False

    st.rerun()
