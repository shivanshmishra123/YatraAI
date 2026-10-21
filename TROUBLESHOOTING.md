# 🔧 Yatra AI — Troubleshooting Guide

This guide covers common issues, root causes, and verified fixes when developing, running, or deploying **Yatra AI**.

---

## 1. Streamlit Port Conflicts (`Address already in use`)

### Symptom:
```text
OSError: [Errno 10048] error while attempting to bind on address ('0.0.0.0', 8501): only one usage of each socket address is normally permitted
```

### Solution:
A previous Streamlit process is still bound to port `8501`. Either specify an alternative port or terminate the existing process:
```powershell
# Option A: Run on a different port
.\.venv\Scripts\streamlit run app.py --server.port 8502

# Option B: Stop existing Streamlit process on Windows
Get-Process -Name python* | Stop-Process -Force
```

---

## 2. Event Loop Errors in Synchronous Tool Wrappers

### Symptom:
```text
RuntimeError: There is no current event loop in thread 'ThreadPoolExecutor-0_0'.
# or
RuntimeError: This event loop is already running
```

### Cause:
In Python 3.10+, calling `asyncio.get_event_loop()` inside secondary worker threads (like LangChain ReAct workers or Streamlit runners) raises an exception if no event loop has been bound to that thread.

### Solution:
All tool adapters in `yatra_agent/tools/tool_registry.py` and `yatra_agent/agent/agent_executor.py` use the thread-safe `_run_sync` helper:
```python
def _run_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)
```
Always use `_run_sync(async_coroutine())` when bridging async MCP calls to synchronous LangChain `BaseTool` interfaces.

---

## 3. ChromaDB SQLite File Locks or Corrupt Collections

### Symptom:
```text
OperationalError: database is locked
# or
error returned from database: (code: 26) file is not a database
```

### Cause:
On Windows, if a Python process is abruptly terminated, SQLite write-ahead-log (`.sqlite3-wal`) locks can remain held on disk in `./data/travel_vectordb/`.

### Solution:
1. `TravelRAGTool` automatically detects database corruption on initialization and recovers cleanly.
2. To manually reset the vector database:
```powershell
# Remove old vector store directory
Remove-Item -Recurse -Force ./data/travel_vectordb

# Rebuild via app UI button "🔄 Rebuild Knowledge Base" or python CLI:
python -c "from yatra_agent.tools.rag_tool import TravelRAGTool; rag = TravelRAGTool(); print(rag.build_knowledge_base())"
```

---

## 4. Missing API Keys & Offline Mode Notice

### Symptom:
App displays:
```text
ℹ️ Domestic Offline Engine Active: Running with deterministic local MCP engines...
```

### Explanation:
This is **expected and intentional** when `OPENAI_API_KEY` or `GOOGLE_API_KEY` are not configured in your `.env`. Yatra AI gracefully falls back to authentic domestic timetable models, highway formulas, and local vector RAG so all features work out-of-the-box.

To activate live web APIs:
1. Copy `.env.example` to `.env`.
2. Add your keys:
   ```bash
   OPENAI_API_KEY=sk-...
   GOOGLE_API_KEY=AIzaSy...
   ```
3. Restart the Streamlit app.

---

## 5. Deprecation Warning: `asyncio.iscoroutinefunction`

### Symptom:
```text
DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
```

### Explanation:
This warning originates from ChromaDB's internal OpenTelemetry telemetry dependency. It does not affect functionality and can be safely ignored. To silence it in tests, run:
```bash
pytest -W ignore::DeprecationWarning
```

---

## 6. City Name Typo or Unrecognized Small Town

### Symptom:
An obscure town is entered (e.g. *"Dhanbad"* or *"Hazaribagh"*), and train search returns general circuit advice.

### Solution:
Yatra AI's city resolver ([city_resolver.py](file:///c:/Users/13shi/OneDrive/Desktop/traverlplanner/yatra_agent/utils/city_resolver.py)) contains 50+ major Indian travel hubs. If traveling to an unlisted destination, specify the nearest major railway junction:
- e.g. instead of *"Bodhgaya village"*, write *"Bodh Gaya"* or *"Gaya"*.
- instead of *"Jim Corbett interior"*, write *"Ramnagar"* or *"Nainital"*.
