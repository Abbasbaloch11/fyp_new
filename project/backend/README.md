# Registrar AI — FastAPI Backend

This wraps your existing Supervisor + 6 domain agents (unchanged) in a REST
API your new frontend can call directly.

## What changed vs. your original zip

- `main_agent.py.py` → moved to `agents/main_agent.py`
- `multi_agents.py.py` → moved to `agents/multi_agents.py`
  (renamed only — the double `.py.py` extension isn't a valid Python module
  name, so it had to change for clean imports. Logic is untouched.)
- `agents/main_agent.py`'s import line updated:
  `from multi_agents import (...)` → `from agents.multi_agents import (...)`
- New: `backend/main.py` — the FastAPI app
- New: `requirements.txt` — added fastapi/uvicorn/pydantic on top of what
  you already had

Nothing in `src/`, `data/`, or `faiss_store_agents/` was touched.

## Run it

```bash
cd RAG                       # project root — must contain data/, agents/, src/, backend/
pip install -r requirements.txt --break-system-packages   # or use a venv
uvicorn backend.main:app --reload --port 8000
```

Your `.env` (`GOOGLE_API_KEY=...`) is loaded the same way it always was, by
`agents/multi_agents.py` and `agents/main_agent.py`.

Visit `http://localhost:8000/docs` for interactive Swagger docs of every
endpoint below.

## Endpoints

| Method | Path                          | Purpose                                        |
|--------|-------------------------------|-------------------------------------------------|
| GET    | `/api/health`                 | Health check                                    |
| POST   | `/api/chat`                   | Supervisor mode — auto-routes to the right agent |
| POST   | `/api/agents/{agent}/query`   | Direct-desk mode — skips Supervisor              |
| GET    | `/api/agents`                 | List of the 6 agents + status                    |
| GET    | `/api/dashboard`               | Total students / active agents / queries today   |
| GET    | `/api/student/{student_id}`   | Merged profile pulled straight from the CSVs (no LLM, can't hallucinate) |
| GET    | `/api/activity?limit=20`      | Recent query log                                 |

`agent` in `/api/agents/{agent}/query` is one of:
`admission`, `examination`, `attendance`, `finance`, `admin`, `analytics`.

### Example

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the CGPA of Zain Hassan?"}'
```

```json
{
  "answer": "Zain Hassan's CGPA is 2.95.",
  "agent": "Examination Agent",
  "agent_key": "examination"
}
```

## Notes / things worth knowing

- **`/api/chat`'s `agent` field is best-effort.** It's read from which tool
  the Supervisor actually called mid-response. It works, but if you want a
  bulletproof guarantee, have each sub-agent tool return a small JSON
  envelope (`{"agent": "...", "answer": "..."}`) instead of plain text, and
  the Supervisor's final answer will carry that through more reliably.
- **`/api/student/{id}` bypasses Gemini entirely** and reads the CSVs
  directly with pandas. This matches the recommendation in your plan (point
  9): never hard-code a student card in the frontend, always fetch from a
  backend endpoint that reflects the current CSV.
- **Activity log is in-memory** (resets on server restart, holds the last
  200 entries). Swap `ACTIVITY_LOG` for a SQLite table (`registrar.db`) when
  you want it to persist — the dict shape already matches a row, so it's a
  drop-in change.
- **CORS is wide open (`allow_origins=["*"]`)** for local development. Lock
  this down to your actual frontend origin before deploying anywhere real.
- This does **not** touch your RAG/agent logic — `structured_search` in
  `src/search.py` is still the ID → structured → FAISS fallback chain you
  already had. If you want name/father-name matching hardened further (per
  point 14 of your plan), that's a separate, smaller change to
  `src/search.py` and I'm happy to do that next.
