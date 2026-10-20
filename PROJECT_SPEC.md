# 🌍 Yatra AI — Intelligent Travel Planning Assistant

## PROJECT SPECIFICATION (Full Detail)

---

## 1. Executive Summary

**Yatra AI** is an intelligent, AI-powered travel planning assistant built for the **Indian travel ecosystem**. It uses a **hybrid LangGraph + LangChain Agent** architecture with **Streamlit** as the frontend, combining **dual-model collaboration** (Google Gemini 1.5 Pro for complex reasoning + GPT-4o-mini for fast extraction/tool-calling), **RAG (Retrieval-Augmented Generation)** over a curated Indian travel knowledge base, and **real-time data integration** via **MCP (Model Context Protocol)** servers for IRCTC trains, Indian railways PNR, Google Maps directions/places, weather, domestic flights, and intercity buses.

The system accepts natural language queries in English or Hindi (Hinglish supported), extracts structured travel parameters, orchestrates multiple external API calls, and synthesizes a complete, budget-aware, culturally-relevant travel itinerary — all through a conversational chat interface.

---

## 2. Problem Statement

Planning a trip in India involves juggling multiple disconnected platforms:

- **IRCTC** for train bookings (notoriously complex UI)
- **Google Maps** or **MapMyIndia** for driving routes and distances
- **MakeMyTrip / Goibibo / Cleartrip** for flights
- **OYO / Booking.com / Goibibo** for hotel search
- **Weather** apps for forecasts
- **Bus & Flight Portals** (RedBus, MakeMyTrip, Skyscanner) for intercity transit comparison
- **Travel blogs / forums** (TripAdvisor, Indiamike, Reddit r/IndiaTravelling) for local tips

A user must open 5-10 tabs, cross-reference prices, manually calculate budgets, and stitch everything together. **Yatra AI** collapses this entire workflow into a single conversational interface.

---

## 3. Target Users

| User Segment | Use Case |
|---|---|
| **Budget-conscious families** | "Plan a 3-day Jaipur trip for 4 people under ₹8,000" |
| **Solo backpackers** | "Cheapest way from Bangalore to Hampi this weekend" |
| **Senior citizens / pilgrimage groups** | "Varanasi trip for 2 elderly, need wheelchair-friendly hotels" |
| **Corporate travelers** | "Mumbai to Pune day trip, need fastest option" |
| **Festival / seasonal travelers** | "Best time to visit Ladakh? What are road conditions in October?" |
| **Multi-city planners** | "Delhi → Agra → Jaipur → Udaipur in 7 days, ₹25,000 budget" |

---

## 4. Core Features — Detailed Breakdown

### 4.1 Intelligent Information Extraction

The system uses an LLM (GPT-4o-mini) to extract structured parameters from free-form natural language:

**Extracted Fields:**

| Field | Type | Examples | Handling |
|---|---|---|---|
| `origin` | string | "Delhi", "Bengaluru", "from Mumbai" | Fuzzy city name resolution; handles alternate names (Bombay→Mumbai, Bangalore→Bengaluru, Calcutta→Kolkata) |
| `destinations` | list[string] | "Jaipur", "Agra and Jaipur", "Manali via Chandigarh" | Supports multi-destination; comma/and/via parsing |
| `departure_date` | date | "Dec 15", "tomorrow", "next Friday", "15 taareekh ko" | Relative date resolution using current IST time; Hindi date parsing |
| `return_date` | date | "Dec 18", "3 days later" | Inferred from `num_days` if not explicit |
| `num_days` | int | "3 days", "a week", "5 nights" | Converts "nights" to days+1 |
| `budget` | float | "₹5000", "5k", "5000 rupees", "budget: 10,000" | Extracts numeric value; handles k/K suffix; per-person vs total detection |
| `num_travelers` | int | "2 people", "family of 4", "couple" | Defaults to 1 if unspecified |
| `traveler_details` | string | "2 elderly, 1 child (5yo)", "with pet dog" | Special needs flagging |
| `preferences` | list[string] | "vegetarian food", "no flights", "adventure activities" | Constraint extraction |
| `optimization_goal` | string | "cheapest", "fastest", "most scenic" | Optimization objective |
| `travel_mode_preference` | string | "train only", "no bus", "prefer flight" | Transport constraint |

**Multi-turn Context:**
- The system maintains a `TravelPlanState` dictionary across conversation turns
- If user says "I want to go to Jaipur" then "from Delhi" then "for 3 days" — all three turns accumulate into the same state
- The system prompts for missing required fields before triggering the full planning pipeline

**Missing Information Prompting:**
- If the user provides a destination but no origin → "Where will you be starting your journey from?"
- If no date → "When are you planning to travel?"
- If no budget → "Do you have a budget in mind? (This helps me recommend appropriate hotels and transport)"

### 4.2 Dual Query Modes

#### 4.2.1 Simple Query Mode

**Trigger:** User asks a general question without providing full travel details.

**Examples:**
- "What are the best places to visit in Kerala?"
- "Tell me about Hampi"
- "Is Ladakh open in December?"
- "Best street food in Old Delhi"

**Pipeline:**
1. Query goes to RAG knowledge base (ChromaDB)
2. If RAG has relevant docs → return enriched answer
3. Optionally call Google Maps POI search for real-time place data
4. GPT-4o-mini synthesizes a response from RAG + POI results

**Tools Used:** RAG search, Google Maps Places (optional)
**Models Used:** GPT-4o-mini only
**Response Time:** 3-8 seconds

#### 4.2.2 Full Planning Mode

**Trigger:** User provides sufficient details (origin + destination + dates/duration + optionally budget).

**Examples:**
- "Plan a trip from Delhi to Jaipur, 3 days, budget ₹8000, starting Dec 15"
- "I want to go from Bangalore to Ooty this weekend with my family of 4, budget 12k"

**Pipeline (Orchestrated by LangGraph StateGraph with LangChain ReAct Execution):**

1. **Pre-Analysis & Routing** (`pre_analyze_node`)
   - Detect query complexity (simple / complex / multi-destination)
   - Count destinations
   - Check budget tightness (per-person-per-day calculation)
   - Detect special constraints (elderly, children, disabled, pets)
   - Decide state transition: LangChain Agent direct path vs Gemini 1.5 Pro strategy decomposition

2. **RAG Knowledge Retrieval** (`rag_retrieval_node`)
   - Search ChromaDB for destination guides
   - Return top-K (default 5) relevant chunks

3. **Train Query** (IRCTC MCP Server)
   - Resolve station codes (e.g., "Delhi" → "NDLS", "New Delhi" → "NDLS", "Old Delhi" → "DLI")
   - Query available trains for the date
   - Return: train number, name, departure/arrival times, classes available, fares (SL/3A/2A/1A), availability status
   - Auto-attach driving route as a comparison

4. **Driving Route** (Google Maps MCP Server)
   - Calculate road distance, estimated driving time, toll costs
   - Factor in national highway vs state highway
   - Return: distance (km), time (hours), estimated fuel cost (based on ₹100/L petrol, 15 km/L average), toll estimates

5. **Hotel Search** (Google Maps MCP Server)
   - Budget-based keyword selection:
     - Budget > ₹3,000/night → "5-star hotel", "luxury resort"
     - Budget ₹1,500-3,000/night → "3-star hotel", "branded hotel"
     - Budget ₹800-1,500/night → "OYO", "budget hotel"
     - Budget < ₹800/night → "hostel", "dharamshala", "guest house"
   - Search near destination city center or tourist area
   - Return: hotel name, rating, price range, address, photos

6. **Weather Forecast** (Weather API MCP Server)
   - Fetch weather for destination city for travel dates
   - Up to 7-day forecast
   - Return: temperature (°C), conditions, humidity, rain probability
   - Special alerts: monsoon warnings, extreme cold, heatwave

7. **Flight Query** (Flight MCP Server — conditional)
   - Triggered only when distance > 500 km or user prefers flights
   - Query available flights (via Amadeus/Skyscanner API)
   - Return: airline, flight number, departure/arrival, price, duration

8. **Bus Query** (RedBus MCP Server — conditional)
   - For routes well-served by buses (e.g., Bangalore-Goa, Delhi-Jaipur)
   - Return: operator, type (AC/Non-AC/Sleeper/Volvo), price, duration

9. **Deep Analysis & Strategy** (Gemini 1.5 Pro — conditional node)
    - Activated for complex scenarios (multi-city, tight budget, conflicting constraints)
    - Generates optimized route order, budget allocation, time distribution
    - Returns structured `query_plan` executed by the LangChain ReAct agent

10. **Critique & Validation Loop** (`critique_validator_node`)
    - LangGraph conditional edge verifies budget constraints, transit feasibility, and weather viability
    - If critical violations exist, loops back with feedback (max 2 iterations) before proceeding

11. **Synthesis** (`synthesizer_node`)
    - Combines ALL gathered data into a formatted travel plan
    - Sections: Overview, Transport Comparison, Hotel Recommendations, Day-wise Itinerary, Budget Breakdown, Weather Advisory, Pro Tips

**Tools Used:** 5 MCP servers + RAG
**Models Used:** GPT-4o-mini (extraction, ReAct tool calls, synthesis) + Gemini 1.5 Pro (complex reasoning)
**Response Time:** 15-60 seconds depending on complexity

### 4.3 Dual-Model Collaboration Strategy

#### Primary Model: GPT-4o-mini (via OpenAI API)
- **Role:** Fast information extraction, tool-calling decisions, response synthesis
- **Strengths:** Low latency, excellent instruction following, good at structured output
- **Used for:** 80% of queries (simple + moderate complexity)

#### Deep Reasoning Model: Google Gemini 1.5 Pro (via Google AI API)
- **Role:** Complex multi-constraint optimization, route planning, budget allocation
- **Strengths:** Long context window (1M tokens), strong reasoning, good with math/logistics
- **Used for:** 20% of queries (multi-destination, tight budgets, complex constraints)

**When Gemini 1.5 Pro is triggered (any ONE condition):**

| Condition | Example | Why Gemini? |
|---|---|---|
| Multi-city route (≥3 cities) | "Delhi → Agra → Jaipur → Udaipur → Mumbai" | Route ordering optimization (TSP-like) |
| Tight budget | "4 people, 5 days, ₹6,000 total" (₹300/person/day) | Precise allocation across transport/hotel/food/tickets |
| Conflicting constraints | "Elderly + child + adventure activities + low budget" | Trade-off analysis |
| Explicit optimization | "Fastest route", "cheapest option", "maximize sightseeing" | Multi-objective optimization |
| Long duration + multiple activities | "10-day Kerala trip covering beaches, backwaters, hill stations, temples" | Complex scheduling and logistics |

**When GPT-4o-mini handles alone:**

- Simple destination queries
- Single-city, comfortable budget trips
- Straightforward A→B travel plans
- Follow-up questions on existing plans

### 4.4 Real-Time Data Integration via MCP

**MCP (Model Context Protocol)** is used to connect the LangChain Agent to external data sources. Each MCP server exposes tools that the Agent can call.

#### MCP Server 1: IRCTC Train Query Server

**Tools exposed:**

| Tool Name | Parameters | Returns |
|---|---|---|
| `get_station_code` | `city_name: str` | Station code(s) — e.g., "Delhi" → ["NDLS", "DLI", "DEE", "ANVT"] |
| `search_trains` | `from_station: str, to_station: str, date: str (DD-MM-YYYY), class: str (optional)` | List of trains with times, classes, fares, availability |
| `get_train_schedule` | `train_number: str` | Full route schedule with all stops and timings |
| `check_availability` | `train_number: str, date: str, class: str, quota: str` | Seat availability status (AVL/RAC/WL) |

**Special handling:**
- Auto-retry on timeout (IRCTC is notoriously slow) — 2 retries, 120-second timeout
- Station code fuzzy matching ("Banglore" → "Bengaluru" → "SBC/KSR")
- Handles Rajdhani/Shatabdi/Duronto premium trains separately
- Tatkal availability check if travel date is within 1 day

#### MCP Server 2: Google Maps Server

**Tools exposed:**

| Tool Name | Parameters | Returns |
|---|---|---|
| `get_directions` | `origin: str, destination: str, mode: str (driving/transit)` | Distance (km), duration, route steps, toll info |
| `search_places` | `query: str, location: str, radius: int, type: str` | List of places with name, rating, address, price level, photos |
| `get_place_details` | `place_id: str` | Full details including reviews, hours, phone |
| `search_hotels` | `location: str, keyword: str, min_price: int, max_price: int` | Filtered hotel list |
| `geocode` | `address: str` | Lat/lng coordinates |

**Special handling:**
- Indian-specific place types: "dhaba", "temple", "ashram", "dharamshala"
- Distance calculation factors in Indian road conditions (1.3x time multiplier for non-highway)
- Fuel cost estimation: ₹100/L petrol, 15 km/L car average, ₹95/L diesel

#### MCP Server 3: Weather Server

**Tools exposed:**

| Tool Name | Parameters | Returns |
|---|---|---|
| `get_forecast` | `city: str, days: int (1-7)` | Daily forecast: temp (°C), conditions, humidity, rain%, wind |
| `get_current_weather` | `city: str` | Current conditions |
| `get_weather_alerts` | `city: str` | Active weather warnings (cyclone, heatwave, heavy rain) |

**Special handling:**
- Monsoon season awareness (June-September): automatic rain warnings
- Hill station temperature adjustments
- Fog alerts for North India (November-January)

#### MCP Server 4: Flight Query Server (Conditional)

**Tools exposed:**

| Tool Name | Parameters | Returns |
|---|---|---|
| `search_flights` | `origin_iata: str, dest_iata: str, date: str, adults: int` | Available flights: airline, number, times, price, duration |
| `get_airport_code` | `city: str` | IATA code — "Delhi" → "DEL", "Mumbai" → "BOM" |

**Trigger conditions:**
- Distance between origin and destination > 500 km
- User explicitly mentions flights
- Travel time by train > 12 hours

**Special handling:**
- Budget airline preference for budget travelers (IndiGo, SpiceJet, Air India Express)
- Full-service airline recommendation for business/luxury travelers (Air India, Vistara)
- Tatkal-like "last minute" fare alerts

#### MCP Server 5: Bus Query Server (Conditional)

**Tools exposed:**

| Tool Name | Parameters | Returns |
|---|---|---|
| `search_buses` | `from: str, to: str, date: str` | Available buses: operator, type, departure, arrival, price |
| `get_bus_types` | `route: str` | Available bus types (Volvo AC, Sleeper, Semi-Sleeper, Ordinary) |

**Well-served routes:** Delhi-Jaipur, Bangalore-Goa, Mumbai-Pune, Hyderabad-Bangalore, Chennai-Pondicherry

### 4.5 RAG Knowledge Base

**Purpose:** Store and retrieve curated Indian travel guides, tips, and local knowledge that LLMs don't reliably have.

**Vector Database:** ChromaDB (local, persistent)

**Embedding Model:** OpenAI `text-embedding-3-small` (1536 dimensions)

**Document Types Supported:**
- `.txt` — Plain text travel guides
- `.md` — Markdown-formatted guides
- `.pdf` — PDF travel books, government tourism brochures
- `.csv` — Structured data (hotel lists, train routes, tourist place databases)

**Knowledge Base Contents (Recommended):**

| Category | Examples |
|---|---|
| City guides | "Complete guide to Jaipur", "48 hours in Varanasi" |
| State tourism | Rajasthan Tourism guide, Kerala Backwaters guide |
| Budget travel tips | "How to travel India under ₹1000/day" |
| Food guides | "Best street food in each Indian city" |
| Cultural etiquette | "Temple visit dos and don'ts", "Dress codes for religious places" |
| Seasonal guides | "Best time to visit each Indian destination" |
| Transport tips | "How to book Tatkal tickets", "Indian Railways travel classes explained" |
| Safety information | "Solo female travel safety tips for India" |
| Pilgrimage routes | "Char Dham Yatra guide", "Amarnath Yatra preparation" |
| Adventure travel | "Trekking in Himachal", "Scuba diving in Andaman" |

**Chunking Strategy:**
- `chunk_size`: 500 characters
- `chunk_overlap`: 50 characters
- Splitter: `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", ". ", " "]`

**UUID Generation:**
```
UUID = MD5(f"{source_file}:{chunk_index}:{content[:100]}")
```
- Enables deduplication on re-import
- Supports incremental updates
- Allows per-source deletion

**Retrieval:**
- Top-K: 5 (configurable)
- Similarity metric: Cosine similarity
- Results include source file metadata for attribution

### 4.6 Conversation Management

**Multi-turn State:**
- Full conversation history maintained in Streamlit `session_state`
- LangChain `ChatMessageHistory` for Agent context
- State dictionary (`TravelPlanState`) persists extracted parameters across turns

**Context Window Management:**
- Last 20 messages kept in active context
- Older messages summarized by GPT-4o-mini and stored as a "conversation summary" message
- RAG results and tool outputs are injected as system messages

**Chat Reset:**
- "Clear Chat" button in sidebar resets all state
- Previous conversation data is not persisted to disk (privacy by default)

### 4.7 Output Format

The final travel plan is rendered in Markdown within the Streamlit chat interface:

```markdown
# 🗺️ Your Travel Plan: Delhi → Jaipur

## 📋 Trip Overview
- **Route:** Delhi → Jaipur
- **Dates:** December 15-17, 2026 (3 days, 2 nights)
- **Travelers:** 2 adults
- **Budget:** ₹8,000 (₹4,000 per person)
- **Weather:** 12-24°C, Clear skies

## 🚆 Transport Options

### Option 1: Train (Recommended ✅)
| Train | Departure | Arrival | Duration | Class | Fare (per person) |
|---|---|---|---|---|---|
| 12016 Ajmer Shatabdi | 06:00 NDLS | 10:30 JP | 4h 30m | CC | ₹785 |
| 12958 ADI SJ Rajdhani | 19:55 NDLS | 00:35 JP | 4h 40m | 3A | ₹1,215 |
| 12462 Mandore Express | 21:10 DLI | 05:05 JP | 7h 55m | SL | ₹350 |

### Option 2: Driving
- **Distance:** 281 km via NH-48
- **Estimated Time:** 4h 45m (with traffic)
- **Fuel Cost:** ~₹1,870 (petrol) / ~₹1,400 (diesel)
- **Tolls:** ~₹735

### Option 3: Bus
| Operator | Type | Departure | Price |
|---|---|---|---|
| RSRTC Volvo | AC Seater | 06:00, 08:00, 10:00... | ₹800 |
| Private (Neeta/SRS) | AC Sleeper | 22:00 | ₹900-1,200 |

## 🏨 Hotel Recommendations
(2-3 hotel options with name, price, rating, address)

## 📅 Day-wise Itinerary
### Day 1 (Dec 15) — Arrival & Old City
(Detailed morning/afternoon/evening breakdown)

### Day 2 (Dec 16) — Forts & Palaces
(Detailed morning/afternoon/evening breakdown)

### Day 3 (Dec 17) — Shopping & Departure
(Detailed morning/afternoon/evening breakdown)

## 💰 Budget Breakdown
| Category | Amount (₹) | Per Person |
|---|---|---|
| Transport (Shatabdi CC × 2, round trip) | ₹3,140 | ₹1,570 |
| Hotel (2 nights × ₹1,200) | ₹2,400 | ₹1,200 |
| Food (3 days × ₹400/person) | ₹1,600 | ₹800 |
| Attractions & Entry Fees | ₹600 | ₹300 |
| Miscellaneous | ₹260 | ₹130 |
| **Total** | **₹8,000** | **₹4,000** |

## 💡 Pro Tips
- 🎫 Book Shatabdi tickets on IRCTC at least 30 days in advance
- 🕌 Hawa Mahal is best visited early morning for photography
- 🍛 Try Dal Baati Churma at LMB on Johari Bazaar
- 🛍️ Bargain at Bapu Bazaar — start at 40% of quoted price
```

---

## 5. Non-Functional Requirements

### 5.1 Performance
- Simple queries: < 5 seconds
- Full planning (single city): < 30 seconds
- Complex planning (multi-city): < 90 seconds
- MCP tool timeout: 120 seconds (IRCTC), 30 seconds (others)
- Auto-retry: 2 retries on MCP failure with exponential backoff

### 5.2 Reliability
- Single tool failure must not crash the pipeline — graceful degradation
- If IRCTC is down → skip train data, mention "IRCTC data unavailable, try later"
- If weather API fails → use general seasonal knowledge from RAG
- All MCP connections use SSE with keep-alive

### 5.3 Scalability
- Single-user Streamlit deployment (MVP)
- ChromaDB handles up to 100K document chunks locally
- Stateless LLM calls — can be horizontally scaled behind a load balancer in future

### 5.4 Security
- API keys stored in `.env` file, never committed to Git
- `.gitignore` includes `.env`, `data/travel_vectordb/`, `__pycache__/`
- No user data persisted beyond the session
- No authentication required (single-user deployment)

### 5.5 Localization
- Primary language: English
- Hindi / Hinglish support in input (understood by LLMs natively)
- City name aliases handled (Bombay↔Mumbai, Bangalore↔Bengaluru, Madras↔Chennai, Calcutta↔Kolkata, Trivandrum↔Thiruvananthapuram)
- Currency: Indian Rupees (₹ / INR) throughout
- Date format: DD-MM-YYYY (Indian standard)
- Time format: 12-hour with IST
- Distance: Kilometers
- Temperature: Celsius

### 5.6 Cost Optimization
- GPT-4o-mini used for 80% of queries (cheap)
- Gemini 1.5 Pro used only when genuinely needed (20% of queries)
- RAG retrieval is free (local ChromaDB)
- MCP calls cost varies by provider
- Estimated cost per full planning query: ~₹2-5 ($0.02-0.06) in API costs

---

## 6. Edge Cases & Error Handling

| Scenario | Handling |
|---|---|
| User asks about international destination | "This planner is optimized for Indian domestic travel. For international trips, I can provide general suggestions." |
| IRCTC server unreachable | Skip train data; suggest user check IRCTC directly; continue with bus/drive/flight options |
| No trains available on date | Suggest adjacent dates; offer alternative transport |
| Budget impossibly low | Warn user: "₹500 for 3 days in Goa may be very tight. Consider increasing budget or reducing days." |
| Destination is restricted area | Flag: "Certain areas in Arunachal Pradesh/Sikkim require Inner Line Permits" |
| Monsoon season travel | Warning: "Heavy rainfall expected. Roads to Ladakh may be closed. Consider alternative dates." |
| Festival crowds | Alert: "Diwali week — expect heavy crowds and surge pricing at hotels in Jaipur" |
| Ambiguous city names | Ask: "Did you mean Hyderabad (Telangana) or Hyderabad (Sindh, Pakistan)?" |
| Multiple stations in a city | "Delhi has multiple railway stations: New Delhi (NDLS), Old Delhi (DLI), Hazrat Nizamuddin (NZM), Anand Vihar (ANVT). I'll check the most relevant ones." |
| User provides Hindi input | Process natively — GPT-4o-mini handles Hindi/Hinglish well |

---

## 7. Future Enhancements (Post-MVP)

- **User authentication** and trip history persistence
- **Booking integration** — direct IRCTC / MakeMyTrip booking links
- **Real-time train tracking** using NTES/RailYatri APIs
- **Cab/auto fare estimation** (Uber/Ola API integration)
- **Local food recommendations** using Zomato/Swiggy API
- **Photo gallery** from Google Places for visual inspiration
- **Shareable itineraries** — export as PDF or share via WhatsApp
- **Voice input** — speak your travel plan in Hindi/English
- **Mobile-responsive PWA** version
- **Multi-language support** — Tamil, Telugu, Bengali, Marathi, Gujarati
- **Travel insurance** recommendations
- **Visa/permit checker** for restricted areas (Ladakh, Arunachal, Andaman)
- **Carbon footprint** calculator for each transport mode
- **Group trip collaboration** — shared planning for friend groups

---

## 8. Glossary

| Term | Definition |
|---|---|
| **IRCTC** | Indian Railway Catering and Tourism Corporation — the online ticketing platform for Indian Railways |
| **MCP** | Model Context Protocol — a protocol for connecting LLMs to external tools/APIs |
| **RAG** | Retrieval-Augmented Generation — enhancing LLM responses with retrieved documents |
| **ReAct** | Reasoning + Acting — an agent paradigm where the LLM reasons about what tool to call next |
| **LangGraph** | Multi-actor stateful orchestration framework extending LangChain with cyclic graph workflows, state persistence, and guardrail routing |
| **Tatkal** | Emergency quota booking system on Indian Railways (opens 1 day before travel) |
| **Shatabdi/Rajdhani** | Premium Indian Railways train categories (chair car / AC sleeper) |
| **SL/3A/2A/1A** | Indian Railways travel classes: Sleeper, 3-tier AC, 2-tier AC, 1st AC |
| **Dharamshala** | Budget pilgrimage accommodation, often attached to temples |
| **Dhaba** | Roadside eatery, typically on highways |
| **IST** | Indian Standard Time (UTC+5:30) |
| **Hinglish** | Mixed Hindi-English colloquial language |
