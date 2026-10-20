# Yatra AI — Claude Code / Agentic Coding Assistant Instructions

> This file provides comprehensive context for any AI coding assistant (Claude, Cursor, Copilot, Windsurf, etc.) working on the Yatra AI project. Read this file FIRST before writing any code.

---

## Project Identity

- **Name:** Yatra AI — Intelligent Travel Planning Assistant
- **Domain:** Indian domestic travel planning
- **Architecture:** Hybrid LangGraph StateGraph (macro workflow, state persistence, validation loops) + LangChain ReAct Agent (micro tool execution, MCP integrations, RAG)
- **Language:** Python 3.11+
- **Primary LLM:** GPT-4o-mini (OpenAI) — fast extraction, tool-calling, synthesis
- **Reasoning LLM:** Gemini 1.5 Pro (Google AI) — complex multi-city/tight-budget optimization
- **Embeddings:** OpenAI text-embedding-3-small
- **Vector DB:** ChromaDB (local, persistent)
- **UI:** Streamlit
- **External Data:** 5 MCP servers (IRCTC trains, Google Maps, Weather, Domestic Flights, Intercity Buses)

---

## Reference Documents

Before writing code, read these project documents for full context:

| Document | Path | What It Covers |
|---|---|---|
| **Project Spec** | `PROJECT_SPEC.md` | Complete feature specifications, extracted fields, MCP tool schemas, output format, edge cases |
| **Architecture** | `ARCHITECTURE.md` | System design, data flows, component structure, API details, security, deployment |
| **Development Plan** | `PLAN.md` | Phased build strategy, task checklists, definitions of done |
| **This File** | `CLAUDE.md` | Coding conventions, patterns, do's and don'ts |

---

## Directory Structure

```
yatra-ai/
├── yatra_agent/                    # Main Python package
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── prompts.py              # ALL LLM prompt templates
│   │   ├── settings.py             # Global constants and configuration
│   │   └── servers_config.json     # MCP server URLs and agent config
│   ├── core/
│   │   ├── __init__.py
│   │   └── agent_executor.py       # LangChain Agent setup and configuration
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py                # TravelPlanState TypedDict definition
│   │   ├── workflow.py             # LangGraph StateGraph compilation and routing
│   │   └── nodes.py                # All processing nodes (extract, analyze, query, synthesize)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── rag_tool.py             # ChromaDB RAG search tool
│   │   ├── mcp_tools.py            # MCP connection manager with retry/timeout
│   │   ├── gemini_tool.py          # Gemini 1.5 Pro wrapper for complex reasoning
│   │   └── tool_registry.py        # ToolDefinition dataclass + all tool schemas
│   └── utils/
│       ├── __init__.py
│       ├── date_parser.py          # Relative date resolution (IST timezone)
│       └── city_resolver.py        # Indian city name alias mapping
├── data/
│   ├── travel_docs/                # Raw travel guide documents (txt, md, pdf, csv)
│   └── travel_vectordb/            # ChromaDB persistence directory (gitignored)
├── app.py                          # Streamlit UI entry point
├── check_mcp_health.py             # MCP server health check script
├── requirements.txt
├── setup.py
├── .env.example
├── .env                            # API keys (gitignored)
├── .gitignore
├── LICENSE
├── README.md
├── PROJECT_SPEC.md
├── ARCHITECTURE.md
├── PLAN.md
└── CLAUDE.md                       # This file
```

---

## Coding Conventions

### Python Style

- **Python version:** 3.11+ (use `match/case`, `|` union types, `TypedDict`)
- **Formatter:** Black (line length 100)
- **Linter:** Ruff
- **Type hints:** Required on all public functions. Use `typing` for complex types.
- **Docstrings:** Google-style docstrings on all public classes and functions.
- **Imports:** Standard library → Third-party → Local, separated by blank lines. Use absolute imports.
- **Variable naming:** `snake_case` for variables/functions, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- **String formatting:** f-strings preferred. Template strings for prompts.
- **Error handling:** Specific exceptions, never bare `except:`. Always log errors.

```python
# ✅ Good
async def search_trains(
    from_station: str,
    to_station: str, 
    date: str,
    travel_class: str | None = None,
) -> dict[str, Any]:
    """Search for trains between two stations on a given date.
    
    Args:
        from_station: Origin station code (e.g., "NDLS").
        to_station: Destination station code (e.g., "JP").
        date: Travel date in DD-MM-YYYY format.
        travel_class: Optional class filter (SL, 3A, 2A, 1A, CC).
    
    Returns:
        Dictionary containing train list with fares and availability.
    
    Raises:
        MCPTimeoutError: If IRCTC server doesn't respond within 120s.
        MCPConnectionError: If IRCTC server is unreachable.
    """
    ...

# ❌ Bad
async def search_trains(f, t, d, c=None):
    """search trains"""
    ...
```

### Async Patterns

- **All MCP tool calls are async.** Use `async/await` throughout the tool layer.
- **Use `asyncio.wait_for()` for timeouts**, not `asyncio.sleep()` polling.
- **Use `asyncio.gather()` for parallel tool calls** where tools are independent (e.g., weather + driving, or train + flight queries).
- **Never block the event loop.** Use `asyncio.to_thread()` for synchronous operations (file I/O, ChromaDB).

```python
# ✅ Parallel independent calls
weather_task = asyncio.create_task(weather_query(state))
driving_task = asyncio.create_task(driving_query(state))
weather_result, driving_result = await asyncio.gather(
    weather_task, driving_task, return_exceptions=True
)

# ✅ Sequential dependent calls
station_code = await get_station_code(city)
trains = await search_trains(station_code, dest_code, date)
```

### Error Handling Patterns

```python
# ✅ MCP tool call with retry
async def safe_tool_call(
    manager: MCPToolManager,
    server: str,
    tool: str,
    params: dict,
    fallback: Any = None,
) -> Any:
    """Call MCP tool with retry logic and graceful fallback."""
    try:
        result = await manager.call_tool(server, tool, params)
        if "error" in result:
            logger.warning(f"Tool {tool} returned error: {result['error']}")
            return fallback
        return result
    except MCPTimeoutError:
        logger.warning(f"Tool {tool} timed out after all retries")
        return fallback
    except MCPConnectionError:
        logger.error(f"Cannot reach {server}")
        return fallback

# ✅ Graceful degradation in synthesis
if state["train_data"] is None:
    plan_sections.append("⚠️ Train data unavailable — please check IRCTC directly.")
else:
    plan_sections.append(format_train_table(state["train_data"]))
```

### Streamlit Patterns

```python
# ✅ Session state initialization (do at top of app.py)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "travel_state" not in st.session_state:
    st.session_state.travel_state = get_initial_state()

# ✅ Chat message display
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🌍"):
        st.markdown(message["content"])

# ✅ Streaming response
with st.chat_message("assistant", avatar="🌍"):
    with st.status("Planning your trip...", expanded=True) as status:
        st.write("🔍 Extracting travel details...")
        state = await extract_info(user_input, state)
        
        st.write("🚆 Querying IRCTC for trains...")
        state = await train_query(state)
        
        st.write("📝 Generating your plan...")
        plan = await synthesize(state)
        
        status.update(label="✅ Plan ready!", state="complete")
    
    st.markdown(plan)

# ❌ Never do this (breaks Streamlit reruns)
result = some_function()  # This runs on EVERY rerun
```

---

## Key Design Decisions (and Why)

### 1. Hybrid LangGraph + LangChain Architecture (Cohesive Symbiosis)

**Decision:** Use **both** LangGraph and LangChain together — LangGraph for macro workflow orchestration and LangChain for micro ReAct tool execution.

**Why:**
- **LangGraph** provides deterministic control over multi-stage workflows: state persistence (`MemorySaver`), explicit conditional routing (simple query vs Gemini deep planning), and cyclic constraint validation loops (e.g., ensuring budget limits or transit feasibility before synthesis).
- **LangChain** provides expressive abstractions for ReAct agents inside execution nodes, wrapping MCP SSE tools as `StructuredTool` instances, managing structured prompt templates, and driving ChromaDB RAG retrieval.
- Combining both avoids unbounded agent drift while retaining flexible, autonomous tool execution inside defined stages.

**Implication:** LangGraph defines the `StateGraph` in `workflow.py`. Within execution nodes in `nodes.py`, use LangChain's ReAct agent (`create_react_agent`) to interactively call tools.

### 2. GPT-4o-mini + Gemini 1.5 Pro (not DeepSeek + Qwen)

**Decision:** Replace DeepSeek R1 with Gemini 1.5 Pro, replace Qwen3 with GPT-4o-mini.

**Why:**
- DeepSeek R1 and Qwen3 (DashScope) are China-optimized; less reliable from Indian servers
- GPT-4o-mini has the best cost/performance for fast tool-calling tasks
- Gemini 1.5 Pro has a 1M token context window — perfect for multi-city planning with lots of data
- Both have reliable Indian API access with low latency
- OpenAI and Google AI both have free tiers suitable for development

**Implication:** All prompts should be in English (with Hindi/Hinglish understanding). Don't write prompts in Chinese.

### 3. Google Maps over Gaode Maps (Amap)

**Decision:** Use Google Maps Platform APIs instead of Gaode (高德地图).

**Why:**
- Gaode Maps only covers China
- Google Maps has comprehensive Indian coverage (directions, places, hotels)
- Google Maps has generous free tier ($200/month credit)
- Well-documented, widely used in India

**Implication:** Use Google Maps API conventions (place_id, lat/lng, etc.).

### 4. IRCTC over 12306

**Decision:** Use IRCTC (Indian Railways) MCP server instead of China's 12306.

**Why:**
- 12306 is China's railway booking system
- IRCTC is the Indian equivalent — handles 700+ million train bookings per year
- Indian Railways has unique features: station codes (4-letter), multiple classes (SL/3A/2A/1A/CC), Tatkal quota, RAC/WL status

**Implication:** Station codes are 3-4 letter codes (NDLS, CSTM, SBC), not numeric. Date format is DD-MM-YYYY.

### 5. Transport Logistics Focus over Astrological Calendars (Excluding Panchang/Bazi)

**Decision:** Exclude Chinese Bazi (八字) and Hindu Panchang calendars; focus exclusively on verifiable transport, lodging, weather, and budget logistics.

**Why:**
- The original Chinese project included Bazi astrology. While Panchang exists in India, travel planning applications are fundamentally operational utilities.
- Users need high-accuracy train seat availability, road driving times, bus schedules, flight fares, and weather alerts.
- Excluding astrology/Panchang eliminates API latency, external dependencies, and potential hallucinated auspicious dates, keeping prompt tokens and reasoning focused on budget feasibility and transit logistics.

**Implication:** Do not build or connect a Panchang MCP server. The 5 MCP servers focus strictly on IRCTC, Google Maps, Weather, Flights, and Buses.

### 6. OpenAI Embeddings over DashScope Embeddings

**Decision:** Use OpenAI `text-embedding-3-small` instead of DashScope `text-embedding-v3`.

**Why:**
- OpenAI embeddings are widely supported and well-documented
- Better multilingual support (English + Hindi)
- ChromaDB has native OpenAI embedding function support
- Very cheap ($0.02/1M tokens)

**Implication:** Embedding dimension is 1536. ChromaDB collection uses cosine similarity.

---

## Indian-Specific Context (IMPORTANT)

When writing code or prompts, keep these India-specific details in mind:

### Currency
- Always use `₹` (Indian Rupee symbol) or `INR`
- Format: `₹1,500` (comma after thousands, Indian numbering: ₹1,00,000 for lakh)
- Budget ranges are very different from Western countries:
  - Budget: ₹500-1,500/night
  - Mid-range: ₹1,500-4,000/night
  - Luxury: ₹4,000-15,000+/night

### Railway System
- India has one of the world's largest rail networks (68,000+ km)
- Station codes: 3-4 uppercase letters (NDLS = New Delhi, CSTM = Mumbai CST, SBC = Bangalore City)
- Train numbers: 5 digits (12016 = Ajmer Shatabdi, 12952 = Mumbai Rajdhani)
- Classes: 1A (First AC), 2A (2-tier AC), 3A (3-tier AC), SL (Sleeper), CC (Chair Car), 2S (Second Sitting)
- Quotas: General (GN), Tatkal (TQ), Ladies (LD), Senior Citizen (SS)
- Availability: AVL (available), RAC (reservation against cancellation), WL (waitlist)
- Major cities have MULTIPLE stations — always resolve to the right one

### Cities
- Many cities have multiple names: Mumbai/Bombay, Bengaluru/Bangalore, Chennai/Madras, Kolkata/Calcutta
- Spelling variations: Varanasi/Banaras/Benares/Kashi
- State capitals matter: "Hyderabad" could be Telangana (India) or Sindh (Pakistan)
- Tier classification matters for hotel/food pricing:
  - Tier 1: Delhi, Mumbai, Bangalore, Chennai, Kolkata, Hyderabad, Pune, Ahmedabad
  - Tier 2: Jaipur, Lucknow, Chandigarh, Indore, Coimbatore, Kochi, Goa
  - Tier 3: Udaipur, Rishikesh, Manali, Ooty, Hampi, Alleppey

### Dates & Time
- Timezone: IST (UTC+5:30) — `Asia/Kolkata`
- Date format: DD-MM-YYYY (Indian standard) or DD/MM/YYYY
- 12-hour time format is common ("4:30 PM")
- Hindi relative dates: "kal" (tomorrow), "parso" (day after tomorrow), "agle hafte" (next week)

### Travel Seasons
- **Peak season:** October-March (winter, pleasant weather across most of India)
- **Monsoon:** June-September (heavy rains, some areas inaccessible)
- **Summer:** April-June (extreme heat in plains, hill stations popular)
- **Festival season:** October-November (Navratri, Dussehra, Diwali — heavy crowding and price surge)

### Food
- India is predominantly vegetarian-friendly — always note veg/non-veg options
- "Pure veg" restaurants are common and important for many travelers
- Regional cuisines vary dramatically: South Indian (dosa, idli), North Indian (roti, dal), coastal (fish curry), Rajasthani (dal baati), Bengali (fish, sweets)
- "Dhaba" = highway roadside eatery (budget, authentic)
- "Thali" = set meal (good value)

### Transport Cost Benchmarks
- **Train (SL class):** ₹0.5-1 per km
- **Train (3A class):** ₹1.5-2.5 per km
- **Train (2A class):** ₹2.5-4 per km
- **Bus (ordinary):** ₹1-1.5 per km
- **Bus (Volvo AC):** ₹2-3 per km
- **Flight (budget):** ₹3-6 per km (but cheaper on long routes due to base fare)
- **Driving (petrol):** ~₹7 per km (fuel only) + tolls
- **Auto-rickshaw:** ₹15-25 per km (city transport)
- **Uber/Ola:** ₹10-20 per km (city transport)

---

## Prompt Engineering Guidelines

### Language
- All system prompts should be in **English**
- The Agent should understand **Hindi and Hinglish** user input natively (GPT-4o-mini and Gemini both support this)
- Responses should be in **English** by default
- If user writes in Hindi, respond in **Hinglish** (mixed Hindi-English)

### System Prompt Structure
Every system prompt should include:
1. **Role definition:** "You are Yatra AI, an expert Indian travel planning assistant."
2. **Context:** What data is available, what tools can be called
3. **Indian-specific instructions:** Currency in ₹, dates in DD-MM-YYYY, IST timezone
4. **Output format:** Markdown with tables, emojis, sections
5. **Constraints:** Don't make up data, cite sources, handle missing data gracefully

### Prompt Template Pattern
```python
SYNTHESIZER_PROMPT_TEMPLATE = """You are Yatra AI, an expert Indian travel planning assistant.

Generate a comprehensive travel plan based on the following gathered data.

## Trip Details
- Origin: {origin}
- Destination: {destination}
- Dates: {departure_date} to {return_date} ({num_days} days)
- Budget: ₹{budget} ({num_travelers} travelers)
- Preferences: {preferences}

## Gathered Data
### Train Options
{train_data}

### Driving Route
{driving_data}

### Hotels
{hotel_data}

### Weather Forecast
{weather_data}

### Knowledge Base
{rag_results}

## Instructions
1. Create a structured travel plan in Markdown
2. Include transport comparison table
3. Include hotel recommendations within budget
4. Include day-wise itinerary with morning/afternoon/evening slots
5. Include budget breakdown table
6. Include pro tips specific to the destination
7. All prices in ₹ (INR)
8. Flag any weather warnings or travel advisories
9. Highlight transit convenience and schedule feasibility
10. If any data is missing, acknowledge it and provide general advice
"""
```

---

## MCP Server Configuration

### `servers_config.json` Format

```json
{
    "mcp_servers": [
        {
            "name": "IRCTC Server",
            "url": "https://your-irctc-mcp-server/sse",
            "timeout": 120,
            "required": true
        },
        {
            "name": "Google Maps Server",
            "url": "https://your-gmaps-mcp-server/sse",
            "timeout": 30,
            "required": true
        },
        {
            "name": "Weather Server",
            "url": "https://your-weather-mcp-server/sse",
            "timeout": 30,
            "required": true
        },
        {
            "name": "Flight Server",
            "url": "https://your-flight-mcp-server/sse",
            "timeout": 30,
            "required": false
        },
        {
            "name": "Bus Server",
            "url": "https://your-bus-mcp-server/sse",
            "timeout": 30,
            "required": false
        }
    ],
    "agent": {
        "name": "YatraAI",
        "instructions": "You are Yatra AI, a professional Indian travel planning assistant. You help users plan domestic trips across India by querying trains (IRCTC), driving routes (Google Maps), hotels, weather forecasts, domestic flights, and intercity buses. Generate detailed, budget-conscious travel plans with day-wise itineraries."
    }
}
```

### Key MCP Implementation Notes
- All MCP connections use **SSE (Server-Sent Events)** protocol
- URLs end with `/sse`
- IRCTC server gets 120-second timeout (notoriously slow)
- All others get 30-second timeout
- Auto-retry: 2 retries with exponential backoff (1s, 3s)
- `required: true` servers cause a warning if unreachable; `required: false` servers are silently skipped

---

## Testing Strategy

### Unit Tests (pytest)
```bash
pytest tests/ -v
```

**Test files:**
- `tests/test_date_parser.py` — relative date parsing
- `tests/test_city_resolver.py` — city alias resolution
- `tests/test_state.py` — state management
- `tests/test_rag_tool.py` — ChromaDB operations

### Integration Tests
```bash
pytest tests/integration/ -v --timeout=120
```

**Test files:**
- `tests/integration/test_mcp_connectivity.py` — server reachability
- `tests/integration/test_agent_simple.py` — simple query end-to-end
- `tests/integration/test_agent_planning.py` — full planning end-to-end

### Manual Test Scenarios

Run these 10 scenarios after any significant change:

| # | Query | Expected Behavior |
|---|---|---|
| 1 | "Best places in Jaipur" | Simple mode, RAG + POI search |
| 2 | "Delhi to Jaipur, 3 days, ₹8000, Dec 15" | Full plan with trains, hotels, weather |
| 3 | "Bangalore to Goa this weekend" | Relative date, bus + flight suggestions |
| 4 | "Mumbai to Pune fastest way" | Driving + train comparison |
| 5 | "Delhi → Agra → Jaipur, 5 days, ₹15000" | Multi-city, Gemini-led mode |
| 6 | "Family of 4, elderly parents, Varanasi, ₹10000" | Special constraints, Gemini analysis |
| 7 | "kal Dilli se Jaipur jaana hai" (Hindi) | Hindi input handling |
| 8 | "Compare bus vs train Bangalore to Goa" | Multi-modal transit comparison (IRCTC + RedBus) |
| 9 | "Cheapest way Delhi to Bangalore" | Optimization goal, flight triggered |
| 10 | "Tell me about Hampi" (no docs in RAG) | Graceful fallback, use LLM knowledge |

---

## Common Pitfalls to Avoid

### ❌ Don't
- Don't hardcode API keys anywhere — always use `os.getenv()`
- Don't use synchronous HTTP calls in async context — use `httpx.AsyncClient`
- Don't catch bare `Exception` without logging — always `logger.error(f"...: {e}")`
- Don't assume MCP servers are always available — always have fallback
- Don't use Chinese text in prompts, comments, or UI — this is an Indian-context project
- Don't use 12306, Gaode Maps, DashScope, or Bazi APIs — use Indian equivalents
- Don't format dates as YYYY-MM-DD in user-facing output — use DD MMM YYYY ("15 Dec 2026")
- Don't assume USD pricing — all prices in INR (₹)
- Don't include astrological or calendar calculations (Panchang/Bazi) — focus purely on real-time transit and logistics
- Don't bypass LangGraph for state management — use LangGraph for workflow routing & validation, and LangChain ReAct within execution nodes
- Don't store user data to disk — privacy by default

### ✅ Do
- Do use `async/await` for all I/O operations
- Do add type hints to every function
- Do log every tool call (input + output summary)
- Do handle timeouts explicitly with `asyncio.wait_for()`
- Do validate extracted dates are in the future
- Do resolve city aliases before making API calls
- Do show loading states in Streamlit during long operations
- Do test with both English and Hindi inputs
- Do calculate fuel costs using Indian prices (₹100/L petrol, 15 km/L)
- Do handle multiple railway stations per city
- Do include IST timezone in all date/time operations
- Do cite RAG sources in responses
- Do add `st.session_state` checks before accessing session variables

---

## Environment Variables Reference

| Variable | Required | Example | Used By |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | `sk-proj-abc123...` | GPT-4o-mini, embeddings |
| `GOOGLE_API_KEY` | ✅ | `AIzaSyAbc123...` | Gemini 1.5 Pro |
| `GMAPS_API_KEY` | ✅ | `AIzaSyDef456...` | Google Maps MCP server (if self-hosted) |
| `WEATHER_API_KEY` | ✅ | `abc123def456` | OpenWeatherMap |
| `IRCTC_API_KEY` | ⭕ | `irctc_key_123` | IRCTC MCP server (if required by provider) |
| `AMADEUS_API_KEY` | ⭕ | `abc123` | Amadeus flight search |
| `AMADEUS_API_SECRET` | ⭕ | `def456` | Amadeus flight search |
| `LANGCHAIN_TRACING_V2` | ⭕ | `true` | LangSmith debugging |
| `LANGCHAIN_API_KEY` | ⭕ | `ls-abc123` | LangSmith debugging |
| `MCP_CONFIG_PATH` | ⭕ | `yatra_agent/config/servers_config.json` | MCP server config location |
| `CHROMA_PERSIST_DIR` | ⭕ | `./data/travel_vectordb` | ChromaDB storage path |

---

## Quick Commands

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e .
cp .env.example .env            # Edit with your API keys

# Run
streamlit run app.py

# Health check
python check_mcp_health.py

# Import travel docs to RAG
python -c "from yatra_agent.tools.rag_tool import TravelRAGTool; t = TravelRAGTool(); t.build_knowledge_base('./data/travel_docs')"

# Run tests
pytest tests/ -v

# Format code
black . --line-length 100
ruff check .
```

---

## Version History

| Version | Date | Changes |
|---|---|---|
| 0.1.0 | Phase 0-1 | Project scaffold + core LLM pipeline |
| 0.2.0 | Phase 2 | RAG knowledge base |
| 0.3.0 | Phase 3 | MCP server integration |
| 0.4.0 | Phase 4 | Dual-model + complex planning |
| 0.5.0 | Phase 5 | UI polish |
| 1.0.0 | Phase 6 | Production-ready release |
