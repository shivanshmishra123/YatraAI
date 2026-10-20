"""Yatra AI — Streamlit Application Entrypoint."""

import streamlit as st
from yatra_agent.utils.city_resolver import resolve_city_name
from yatra_agent.utils.date_parser import get_current_ist_datetime

st.set_page_config(
    page_title="Yatra AI — Intelligent Travel Planner",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom header styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #ff6b35;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        background-color: #e6f4ea;
        color: #137333;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🌍 Yatra AI — Intelligent Travel Planner</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI-powered travel planning for the Indian domestic ecosystem | '
    '<span class="status-badge">Phase 0: Scaffolded</span></div>',
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("System initialized with LangGraph + LangChain hybrid architecture.")
    
    ist_time = get_current_ist_datetime()
    st.write(f"🕒 **Current IST:** `{ist_time.strftime('%d-%m-%Y %H:%M:%S')}`")
    
    st.divider()
    st.subheader("🔌 Connected MCP Servers (5)")
    st.markdown("- 🚆 **IRCTC Train Server**")
    st.markdown("- 🚗 **Google Maps Server**")
    st.markdown("- ☀️ **Weather Server**")
    st.markdown("- ✈️ **Flight Server**")
    st.markdown("- 🚌 **Bus Server**")

# Main Page Area
st.subheader("👋 Welcome to Yatra AI")
st.write(
    "Phase 0 environment setup is complete! The system skeleton, package directories, "
    "configurations, and dependencies are now initialized."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧩 Quick Sanity Checks")
    test_city = st.text_input("Test City Resolver:", value="bombay")
    if test_city:
        resolved = resolve_city_name(test_city)
        st.success(f"Canonical Indian City: **{resolved}**")

with col2:
    st.markdown("### 🗺️ Next Steps (Phase 1)")
    st.markdown("- [ ] **Core LLM Pipeline**: GPT-4o-mini structured parameter extraction")
    st.markdown("- [ ] **Prompt Templates**: Extraction, planning, and synthesis prompts")
    st.markdown("- [ ] **Conversational Chat UI**: Dynamic message rendering & streaming")

st.divider()
st.caption("Yatra AI — Open Source Domestic Indian Travel Assistant | Powered by LangGraph + LangChain")
