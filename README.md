# 🌍 Yatra AI — Intelligent Travel Planning Assistant

**Yatra AI** is an AI-powered travel planning assistant tailored for the **Indian domestic travel ecosystem**. It features a **hybrid LangGraph + LangChain Agent** architecture paired with **Streamlit**, integrating dual-model reasoning (**Google Gemini 1.5 Pro** + **GPT-4o-mini**), local **RAG knowledge retrieval** via ChromaDB, and real-time transit data through **MCP (Model Context Protocol)** servers.

---

## 🚀 Key Features

- **Dual-Model Collaboration**:
  - **GPT-4o-mini**: Fast NLU parameter extraction, LangChain ReAct tool invocation, and dynamic response synthesis.
  - **Gemini 1.5 Pro**: Deep reasoning for multi-city route optimization, tight budget allocation, and constraint resolution.
- **Hybrid LangGraph + LangChain Architecture**:
  - **LangGraph**: Workflow orchestration, conversation state persistence (`MemorySaver`), complexity routing, and constraint validation feedback cycles.
  - **LangChain**: ReAct autonomous tool-calling loops, prompt templates, and ChromaDB RAG.
- **Real-Time Data via MCP**:
  - 🚆 **IRCTC Trains**: Station codes, schedules, seat availability, and fare classes (SL/3A/2A/1A/CC).
  - 🚗 **Google Maps**: Driving routes, live traffic duration, fuel estimates, tolls, and hotels/places.
  - ☀️ **Weather**: Destination forecasts and season-specific warnings (Monsoon, Fog, Heatwave).
  - ✈️ **Domestic Flights**: Schedule and fare queries via Amadeus API.
  - 🚌 **Intercity Buses**: Route coverage and bus types via RedBus API.
- **RAG Knowledge Base**: Curated local travel tips, cultural etiquette, Tatkal booking guides, and city recommendations stored locally in ChromaDB.
- **Streamlit Chat UI**: Clean conversational interface with real-time thought streaming, budget breakdowns, and transport comparison tables.

---

## 📁 Repository Structure

```
yatra-ai/
├── yatra_agent/                    # Core Python package
│   ├── config/                     # Prompts, global settings, MCP server configs
│   ├── core/                       # LangChain ReAct agent executor
│   ├── graph/                      # LangGraph StateGraph, nodes, and state definitions
│   ├── tools/                      # MCP manager, RAG tools, and tool registries
│   └── utils/                      # IST date parser, city resolver
├── data/
│   ├── travel_docs/                # Curated knowledge documents for RAG
│   └── travel_vectordb/            # Local ChromaDB vector database
├── app.py                          # Streamlit UI entry point
├── check_mcp_health.py             # MCP server connectivity & health check tool
├── requirements.txt                # Python dependencies
├── setup.py                        # Editable package installer
├── PROJECT_SPEC.md                 # Full project specification
├── ARCHITECTURE.md                 # System architecture & sequence diagrams
├── PLAN.md                         # 6-phase implementation roadmap
└── CLAUDE.md                       # Coding conventions & developer guide
```

---

## ⚡ Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/yatra-ai.git
cd yatra-ai

# Create and activate virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Configure Environment Variables
Copy the template and provide your API keys:
```bash
cp .env.example .env
```
Fill in `.env`:
```ini
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIzaSy...
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## 📖 Documentation
- [Project Specification](PROJECT_SPEC.md)
- [System Architecture](ARCHITECTURE.md)
- [Development Plan](PLAN.md)
- [Agent Guidelines](CLAUDE.md)

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
