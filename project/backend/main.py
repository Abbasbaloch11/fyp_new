"""
Registrar AI — FastAPI backend
================================

This is the API layer that sits between the new frontend (Registrar AI UI)
and the existing Supervisor / 6-agent / RAG backend that already lives in
`agents/` and `src/`. Nothing about the agent logic is rewritten here — we
just expose it over HTTP with structured JSON responses.

Run from the project root (the folder that contains `data/`, `agents/`,
`src/`, `backend/`) so the relative FAISS / CSV paths used by RAGSearch
still resolve correctly:

    cd RAG
    pip install -r requirements.txt
    uvicorn backend.main:app --reload --port 8000

Then point the frontend at http://localhost:8000
"""

import os
import glob
import time
import uuid
import pandas as pd

from typing import Optional, List
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.main_agent import supervisor_agent
from agents.multi_agents import AGENTS


# =============================================================
# APP SETUP
# =============================================================

app = FastAPI(title="Registrar AI Backend", version="1.0.0")

# Allow the frontend (any origin during development) to call this API.
# Tighten allow_origins to your real frontend URL before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

# In-memory activity log (swap for SQLite later — see note at bottom of file)
ACTIVITY_LOG: List[dict] = []
MAX_ACTIVITY_LOG = 200

AGENT_DISPLAY_NAMES = {
    "admission": "Admission Agent",
    "examination": "Examination Agent",
    "attendance": "Attendance Agent",
    "finance": "Finance Agent",
    "admin": "Admin Agent",
    "analytics": "Analytics Agent",
}

DATA_DIRS = {
    "admission": "data/admission_agent_data",
    "examination": "data/examination_agent_data",
    "attendance": "data/attendance_agent_data",
    "finance": "data/finance_agent_data",
    "admin": "data/admin_agent_data",
    "analytics": "data/analytics_agent_data",
}


# =============================================================
# REQUEST / RESPONSE MODELS
# =============================================================

class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    agent: Optional[str] = None
    agent_key: Optional[str] = None


class AgentQueryRequest(BaseModel):
    query: str


# =============================================================
# HELPERS
# =============================================================

def _first_csv(data_dir: str) -> Optional[str]:
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    return files[0] if files else None


def _load_csv(agent_key: str) -> Optional[pd.DataFrame]:
    data_dir = DATA_DIRS.get(agent_key)
    if not data_dir:
        return None
    csv_path = _first_csv(data_dir)
    if not csv_path:
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None


def _find_id_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        low = col.lower()
        if "unique id" in low or "unique_id" in low or "student id" in low or low == "id":
            return col
    return None


def _log_activity(query: str, agent_key: str, agent_name: str, status: str = "success"):
    entry = {
        "id": str(uuid.uuid4()),
        "query": query,
        "agent": agent_name,
        "agent_key": agent_key,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
    }
    ACTIVITY_LOG.insert(0, entry)
    del ACTIVITY_LOG[MAX_ACTIVITY_LOG:]


def _guess_agent_from_tool_calls(response: dict) -> str:
    """
    Best-effort: look at which tool the Supervisor actually called so we can
    tell the frontend which agent handled the query, instead of guessing
    from keywords again on the API side.
    """
    tool_to_key = {
        "admission_registrar_agent": "admission",
        "examination_agent": "examination",
        "attendance_agent": "attendance",
        "finance_agent": "finance",
        "admin_agent": "admin",
        "analytics_agent": "analytics",
    }

    for msg in response.get("messages", []):
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                name = call.get("name") if isinstance(call, dict) else None
                if name in tool_to_key:
                    return tool_to_key[name]
    return "unknown"


# =============================================================
# HEALTH
# =============================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 1),
    }


# =============================================================
# CHAT (Supervisor mode — routes to the right agent automatically)
# =============================================================

@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    try:
        response = supervisor_agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })
    except Exception as exc:
        _log_activity(query, "unknown", "Supervisor", status="error")
        raise HTTPException(status_code=500, detail=f"Supervisor error: {exc}")

    answer = response["messages"][-1].content
    agent_key = _guess_agent_from_tool_calls(response)
    agent_name = AGENT_DISPLAY_NAMES.get(agent_key, "Supervisor")

    _log_activity(query, agent_key, agent_name)

    return ChatResponse(answer=answer, agent=agent_name, agent_key=agent_key)


# =============================================================
# DIRECT DESK MODE (bypasses Supervisor, calls one agent directly)
# =============================================================

@app.post("/api/agents/{agent_key}/query", response_model=ChatResponse)
def query_agent(agent_key: str, payload: AgentQueryRequest):
    if agent_key not in AGENTS:
        raise HTTPException(status_code=404, detail=f"Unknown agent '{agent_key}'")

    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    try:
        response = AGENTS[agent_key].invoke({
            "messages": [{"role": "user", "content": query}]
        })
    except Exception as exc:
        _log_activity(query, agent_key, AGENT_DISPLAY_NAMES.get(agent_key, agent_key), status="error")
        raise HTTPException(status_code=500, detail=f"{agent_key} agent error: {exc}")

    answer = response["messages"][-1].content
    agent_name = AGENT_DISPLAY_NAMES.get(agent_key, agent_key)

    _log_activity(query, agent_key, agent_name)

    return ChatResponse(answer=answer, agent=agent_name, agent_key=agent_key)


# =============================================================
# AGENTS LIST / STATUS
# =============================================================

@app.get("/api/agents")
def list_agents():
    return {
        "agents": [
            {"key": key, "name": name, "status": "online"}
            for key, name in AGENT_DISPLAY_NAMES.items()
        ]
    }


# =============================================================
# DASHBOARD STATS
# =============================================================

@app.get("/api/dashboard")
def dashboard():
    # Total students = distinct student IDs across the admission CSV
    # (the most authoritative source of "who is an enrolled student")
    total_students = 0
    df = _load_csv("admission")
    if df is not None:
        id_col = _find_id_column(df)
        if id_col:
            total_students = df[id_col].nunique()
        else:
            total_students = len(df)

    queries_today = sum(
        1 for entry in ACTIVITY_LOG
        if entry["timestamp"][:10] == datetime.now(timezone.utc).date().isoformat()
    )

    return {
        "total_students": int(total_students),
        "active_agents": len(AGENTS),
        "queries_today": queries_today,
        "system_status": "Operational",
    }


# =============================================================
# STUDENT PROFILE
# =============================================================

@app.get("/api/student/{student_id}")
def get_student(student_id: str):
    """
    Direct structured lookup (no LLM) across every domain CSV that has a
    student-id column, merged into one profile. This is deliberately NOT
    routed through Gemini/RAG — for a known ID, a plain pandas lookup is
    faster and can't hallucinate.
    """
    student_id = student_id.strip().upper()
    profile: dict = {"student_id": student_id, "found": False}

    for agent_key, data_dir in DATA_DIRS.items():
        df = _load_csv(agent_key)
        if df is None:
            continue

        id_col = _find_id_column(df)
        if not id_col:
            continue

        matches = df[df[id_col].astype(str).str.upper().str.strip() == student_id]
        if matches.empty:
            continue

        profile["found"] = True
        row = matches.iloc[0].to_dict()

        if agent_key == "admission" and "name" not in profile:
            profile["name"] = row.get("Student Name")
            profile["program"] = row.get("Program")
            profile["enrollment_status"] = row.get("Enrollment Status")
            profile["eligibility_status"] = row.get("Eligibility Status")

        if agent_key == "examination":
            profile["name"] = profile.get("name") or row.get("Student Name")
            semester_cols = [c for c in df.columns if "semester" in c.lower() and "cgpa" in c.lower()]
            profile["semester_cgpa"] = {c: row.get(c) for c in semester_cols}
            non_null = [row.get(c) for c in semester_cols if pd.notna(row.get(c))]
            profile["latest_cgpa"] = non_null[-1] if non_null else None

        if agent_key == "attendance":
            attendance_rows = matches.to_dict(orient="records")
            profile["attendance"] = attendance_rows

        if agent_key == "finance":
            finance_rows = matches.to_dict(orient="records")
            profile["finance"] = finance_rows

        if agent_key == "analytics":
            profile["risk_level"] = row.get("Risk Level")
            profile["cgpa_trend"] = row.get("CGPA Trend")
            profile["attendance_trend"] = row.get("Attendance Trend")
            profile["prediction_note"] = row.get("Prediction Note")

    if not profile["found"]:
        raise HTTPException(status_code=404, detail=f"No student found with ID {student_id}")

    return profile


# =============================================================
# ACTIVITY LOG
# =============================================================

@app.get("/api/activity")
def get_activity(limit: int = 20):
    return {"activity": ACTIVITY_LOG[:limit]}


# =============================================================
# NOTE ON SCALING THIS UP
# =============================================================
# - Swap ACTIVITY_LOG for a SQLite table (registrar.db) once you need it to
#   survive a server restart — same shape of dict works as a row.
# - The /api/chat "agent" field is best-effort (read from which tool the
#   Supervisor actually called). If you want 100% certainty, have each
#   sub-agent tool return a small JSON envelope instead of plain text.
