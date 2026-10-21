"""Yatra AI — Streamlit Application Entrypoint."""

import os
import streamlit as st
from yatra_agent.graph.state import create_initial_state
from yatra_agent.graph.workflow import build_travel_workflow
from yatra_agent.utils.date_parser import get_current_ist_datetime

st.set_page_config(
    page_title="Yatra AI — Intelligent Travel Planner",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ff6b35;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #666;
        margin-bottom: 1.2rem;
    }
    .badge-phase1 {
        background-color: #e8f0fe;
        color: #1a73e8;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .param-box {
        background-color: #f8f9fa;
        border-left: 3px solid #ff6b35;
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: 2px;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Namaste! 🙏 I am **Yatra AI**, your Indian travel planning assistant. "
                       "Where would you like to travel, or what destination can I help you explore today?",
        }
    ]

if "travel_state" not in st.session_state:
    st.session_state.travel_state = create_initial_state()

if "workflow" not in st.session_state:
    st.session_state.workflow = build_travel_workflow()


# Function to clear chat and reset state
def reset_conversation():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Namaste! 🙏 I am **Yatra AI**, your Indian travel planning assistant. "
                       "Where would you like to travel, or what destination can I help you explore today?",
        }
    ]
    st.session_state.travel_state = create_initial_state()
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar UI: State Inspector & Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("⚙️ Yatra AI Controls")
    
    if st.button("🗑️ Clear Chat & Reset State", use_container_width=True, on_click=reset_conversation):
        st.stop()

    ist_time = get_current_ist_datetime()
    st.caption(f"🕒 **Current IST:** `{ist_time.strftime('%d-%m-%Y %H:%M:%S')}`")

    st.divider()
    st.subheader("📊 Live Travel Parameters")
    ts = st.session_state.travel_state

    origin_display = ts.get("origin") or "*(Not specified)*"
    dest_display = ", ".join(ts.get("destinations", [])) or "*(Not specified)*"
    date_display = ts.get("departure_date") or "*(Not specified)*"
    days_display = f"{ts.get('num_days')} days" if ts.get("num_days", 0) > 0 else "*(Not specified)*"
    budget_val = ts.get("budget", 0)
    budget_display = f"₹{budget_val:,.0f}" if budget_val > 0 else "*(Not specified)*"
    travelers_display = f"{ts.get('num_travelers', 1)} people"
    mode_display = ts.get("query_type", "simple").upper()
    deep_reasoning = "⚡ Gemini 1.5 Pro" if ts.get("needs_deep_analysis") else "🤖 GPT-4o-mini"

    st.markdown(f"**Origin:** {origin_display}")
    st.markdown(f"**Destination(s):** {dest_display}")
    st.markdown(f"**Departure Date:** {date_display}")
    st.markdown(f"**Duration:** {days_display}")
    st.markdown(f"**Budget:** {budget_display}")
    st.markdown(f"**Travelers:** {travelers_display}")
    st.markdown(f"**Mode:** `{mode_display}` ({deep_reasoning})")

    st.divider()
    st.subheader("💡 Quick Starter Queries")
    starter_prompts = [
        "Plan Delhi to Jaipur, 3 days, ₹8000, starting Dec 15",
        "What are the best places in Jaipur?",
        "Bangalore to Goa this weekend for 2 people",
        "kal Dilli se Jaipur jaana hai",
    ]
    for prompt_text in starter_prompts:
        if st.button(prompt_text, key=f"btn_{prompt_text}"):
            st.session_state.pending_prompt = prompt_text
            st.rerun()


# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🌍 Yatra AI — Travel Planner</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Intelligent Domestic Travel Planning for India | '
    '<span class="badge-phase1">Phase 1: Multi-turn NLU & StateGraph</span></div>',
    unsafe_allow_html=True,
)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Handle user input from chat_input or starter buttons
user_input = st.chat_input("Ask a travel question or plan a trip across India...")

if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    user_input = st.session_state.pending_prompt
    del st.session_state["pending_prompt"]

if user_input:
    # 1. Render user message in UI
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.travel_state["messages"] = list(st.session_state.messages)

    # 2. Invoke LangGraph Workflow
    with st.spinner("Processing with Yatra AI NLU engine..."):
        try:
            workflow = st.session_state.workflow
            config = {"configurable": {"thread_id": "yatra_streamlit_session"}}
            
            # Execute compiled LangGraph StateGraph
            final_state = workflow.invoke(st.session_state.travel_state, config=config)
            st.session_state.travel_state = final_state

            # 3. Retrieve latest assistant message
            assistant_messages = [m for m in final_state.get("messages", []) if m.get("role") == "assistant"]
            if assistant_messages:
                latest_reply = assistant_messages[-1]["content"]
                # Update session messages if reply was newly generated
                if len(st.session_state.messages) == 0 or st.session_state.messages[-1].get("content") != latest_reply:
                    st.session_state.messages.append({"role": "assistant", "content": latest_reply})
                    with st.chat_message("assistant"):
                        st.markdown(latest_reply)
        except Exception as e:
            err_msg = f"⚠️ Processing error: {e}"
            st.error(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})

    st.rerun()
