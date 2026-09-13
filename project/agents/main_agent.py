import os
from dotenv import load_dotenv
from langchain.agents import create_agent

from agents.multi_agents import (
    admission_agent,
    examination_agent,
    attendance_agent,
    finance_agent,
    admin_agent,
    analytics_agent,
    AGENTS,
)

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

MODEL = "google_genai:gemini-2.5-flash"


# =========================================================
# HELPER: turn a sub-agent into a callable tool for the Supervisor
# =========================================================

def make_agent_tool(sub_agent, name: str, description: str):
    """
    Wraps an already-built LangChain agent (admission_agent, examination_agent, etc.)
    as a single tool function the Supervisor can call.
    """

    def tool_fn(query: str) -> str:
        result = sub_agent.invoke({
            "messages": [
                {"role": "user", "content": query}
            ]
        })
        return result["messages"][-1].content

    tool_fn.__name__ = name
    tool_fn.__doc__ = description

    return tool_fn


# =========================================================
# WRAP EACH DOMAIN AGENT AS A TOOL
# =========================================================

admission_tool = make_agent_tool(
    admission_agent,
    "admission_registrar_agent",
    (
        "Call this agent for anything related to student admissions, "
        "eligibility, enrollment status, or academic history records. "
        "Example queries: 'Is this student eligible for admission?', "
        "'Show enrollment details of student X', 'What is the academic "
        "history of student X?'."
    ),
)

examination_tool = make_agent_tool(
    examination_agent,
    "examination_agent",
    (
        "Call this agent for anything related to student results, GPA/CGPA, "
        "or transcripts. Example queries: 'What is the CGPA of student X?', "
        "'Show the transcript for student X', 'What are the semester-wise "
        "results of student X?'."
    ),
)

attendance_tool = make_agent_tool(
    attendance_agent,
    "attendance_agent",
    (
        "Call this agent for anything related to student attendance, "
        "attendance percentage, or attendance shortage. Example queries: "
        "'What is the attendance percentage of student X?', 'Which students "
        "have short attendance?'."
    ),
)

finance_tool = make_agent_tool(
    finance_agent,
    "finance_agent",
    (
        "Call this agent for anything related to student fees, challans, "
        "dues, or payment history. Example queries: 'What is the fee status "
        "of student X?', 'Has student X paid this semester's challan?'."
    ),
)

admin_tool = make_agent_tool(
    admin_agent,
    "admin_agent",
    (
        "Call this agent for anything related to faculty records, staff "
        "management, or system-level administrative data. Example queries: "
        "'List all faculty members in the CS department', 'Show staff "
        "details of employee X'."
    ),
)

analytics_tool = make_agent_tool(
    analytics_agent,
    "analytics_agent",
    (
        "Call this agent for anything related to performance prediction, "
        "attendance trends, or student risk analysis. Example queries: "
        "'Which students are at risk of failing?', 'Predict the performance "
        "trend of student X'."
    ),
)


# =========================================================
# SUPERVISOR / ORCHESTRATOR AGENT
# =========================================================

SUPERVISOR_SYSTEM_PROMPT = """
You are the Supervisor / Orchestrator Agent of a University Management
Multi-Agent System. You do not answer questions yourself and you do not
have direct access to any data. Your only job is to understand the user's
query and route it to the ONE correct specialist agent (tool) that can
answer it.

You manage these specialist agents:
1. admission_registrar_agent -> admissions, eligibility, enrollment, academic history
2. examination_agent -> results, GPA/CGPA, transcripts
3. attendance_agent -> attendance records, attendance percentage, shortage cases
4. finance_agent -> fees, challans, dues, payment history
5. admin_agent -> faculty records, staff management, admin data
6. analytics_agent -> performance prediction, attendance trends, risk analysis

STRICT RULES:
1. Read the user's query carefully and decide which ONE agent (or, only if
   truly necessary, a small number of agents) is relevant to it.
2. Call ONLY the relevant agent(s). NEVER call every agent "just in case."
3. If the query clearly needs only one agent, call exactly one tool, not more.
4. If a query genuinely spans two domains (e.g. "check attendance and fee
   status of student X"), call each relevant agent once, then combine both
   answers into a single clear reply.
5. If the query does not match any agent's domain, tell the user politely
   that this system cannot help with that request instead of guessing or
   calling a random agent.
6. Never call the same agent more than once for the same query.
7. Do not fabricate information. Only report what the specialist agent
   returned to you.
8. Keep your final answer clear, concise, and organized for the user.
"""

supervisor_agent = create_agent(
    model=MODEL,
    tools=[
        admission_tool,
        examination_tool,
        attendance_tool,
        finance_tool,
        admin_tool,
        analytics_tool,
    ],
    system_prompt=SUPERVISOR_SYSTEM_PROMPT,
)


# =========================================================
# QUICK MANUAL TEST
# =========================================================

if __name__ == "__main__":

    query = "What is your main purpose and how many agents are connected with you and tell me the purpose of each agent?"

    print(f"\n[TEST] Supervisor handling query: {query}\n")

    response = supervisor_agent.invoke({
        "messages": [
            {"role": "user", "content": query}
        ]
    })

    print(response["messages"][-1].content)