# 🤝 Contributing to Yatra AI

Thank you for your interest in contributing to **Yatra AI**! This project is an agentic domestic travel planning platform for India built using **LangGraph**, **LangChain**, **Google Gemini 1.5 Pro**, and **ChromaDB**.

---

## 🧭 Core Principles & Constraints

1. **Strictly Indian Domestic Context**:
   - Currency is always Indian Rupees (**INR / `₹`**).
   - Rail transit is powered by **IRCTC** station codes, berths (1A, 2A, 3A, 3E, SL, CC), and Tatkal guidelines.
   - Road calculations must use Indian National Highway benchmarks (fuel efficiency, FASTag toll costs, monsoon/fog alerts).
2. **Zero Astrology / No Panchang**:
   - Do not include Panchang, Kundali, or astrological timing constraints.
3. **Zero-Failure Architecture**:
   - Any external API failure must fall back gracefully to the domestic local engine without crashing the UI or pipeline.
4. **Hybrid LangGraph + LangChain**:
   - LangGraph manages workflow state, routing, validation guardrails, and checkpoints.
   - LangChain manages individual tool execution, callbacks, and prompt templates.

---

## 🛠️ Local Development Setup

### 1. Clone the repository
```bash
git clone https://github.com/shivanshmishra123/YatraAI.git
cd YatraAI
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies in editable mode
```bash
pip install -e .
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env with your keys, or leave blank to use the domestic offline engine
```

---

## 🧪 Running Tests

Yatra AI has a test suite covering unit tests, scenario tests, and hardening:

```bash
# Run all tests
pytest -v

# Run a specific test module
pytest tests/test_scenarios.py -v
pytest tests/test_hardening.py -v
pytest tests/test_gemini_and_phase4.py -v
```

Before opening a pull request, ensure that **all 98+ tests pass** with zero errors:
```bash
pytest
```

---

## 📁 Repository Structure

```text
├── app.py                     # Streamlit frontend & session coordinator
├── check_mcp_health.py        # CLI health probe for MCP tool servers
├── yatra_agent/
│   ├── agent/                 # LangChain ReAct agent executor
│   ├── config/                # Prompts, settings, servers_config.json
│   ├── graph/                 # LangGraph StateGraph, nodes, and state definitions
│   ├── tools/                 # MCP tool manager, RAG tool, Gemini reasoning tool
│   └── utils/                 # City resolver, date parser, Indian NLP helpers
├── data/
│   ├── travel_docs/           # Curated markdown guides for ChromaDB RAG
│   └── travel_vectordb/       # Persistent ChromaDB vector database
└── tests/                     # 10 comprehensive pytest test modules
```

---

## 📜 Pull Request Process

1. Fork the repo and create a feature branch (`git checkout -b feature/amazing-feature`).
2. Implement your changes following clean code standards and docstrings.
3. Add corresponding unit or scenario tests in `tests/`.
4. Run `pytest` to verify 100% pass rate.
5. Commit using conventional commit format (`feat: ...`, `fix: ...`, `docs: ...`).
6. Push to your branch and open a Pull Request against `main`.
