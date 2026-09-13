import os
from dotenv import load_dotenv
from langchain.agents import create_agent

from src.search import RAGSearch

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

MODEL = "google_genai:gemini-2.5-flash"


# =========================================================
# HELPER: build a RAGSearch engine + tool + agent for one domain
# =========================================================

def build_domain_agent(agent_key: str, data_subdir: str, tool_description: str, system_prompt: str):
    """
    agent_key     -> short name, e.g. 'admission' (used for its private faiss store)
    data_subdir   -> folder name under data/, e.g. 'admission_agent_data'
    tool_description -> docstring shown to Gemini so it knows when to call this tool
    system_prompt -> persona/instructions for this specific agent
    """

    engine = RAGSearch(
        persist_dir=f"faiss_store_agents/{agent_key}",
        data_dir=f"data/{data_subdir}",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="gemini-2.5-flash",
    )

    def tool_fn(query: str) -> str:
        return engine.search_and_summarize(query, top_k=5)

    tool_fn.__name__ = f"{agent_key}_search_tool"
    tool_fn.__doc__ = tool_description

    agent = create_agent(
        model=MODEL,
        tools=[tool_fn],
        system_prompt=system_prompt,
    )

    return agent


# =========================================================
# 1. ADMISSION & REGISTRAR AGENT
# =========================================================

admission_agent = build_domain_agent(
    agent_key="admission",
    data_subdir="admission_agent_data",
    tool_description=(
        "Search the admissions & registrar knowledge base. Use this for "
        "questions about student admissions, eligibility criteria, "
        "enrollment status, and academic history records."
    ),
    system_prompt=(
        "You are the Admission & Registrar Agent of a university management "
        "system. Your job is to handle student admissions, validate "
        "eligibility, and answer questions about student records and academic "
        "history. Always use your search tool to find the answer in the "
        "admissions knowledge base before responding. If the information is "
        "not found, clearly say so instead of guessing."
    ),
)


# =========================================================
# 2. EXAMINATION AGENT
# =========================================================

examination_agent = build_domain_agent(
    agent_key="examination",
    data_subdir="examination_agent_data",
    tool_description=(
        "Search the examination knowledge base. Use this for questions "
        "about student results, GPA/CGPA calculations, and transcripts."
    ),
    system_prompt=(
        "You are the Examination Agent of a university management system. "
        "Your job is to manage and report student results, calculate "
        "GPA/CGPA, and provide transcript information. Always use your "
        "search tool to find the answer in the examination knowledge base "
        "before responding. If the information is not found, clearly say so."
    ),
)


# =========================================================
# 3. ATTENDANCE AGENT
# =========================================================

attendance_agent = build_domain_agent(
    agent_key="attendance",
    data_subdir="attendance_agent_data",
    tool_description=(
        "Search the attendance knowledge base. Use this for questions about "
        "student attendance records, attendance percentages, and short "
        "attendance / shortage cases."
    ),
    system_prompt=(
        "You are the Attendance Agent of a university management system. "
        "Your job is to track attendance, calculate attendance percentages, "
        "and detect short-attendance cases. Always use your search tool to "
        "find the answer in the attendance knowledge base before responding. "
        "If the information is not found, clearly say so."
    ),
)


# =========================================================
# 4. FINANCE AGENT
# =========================================================

finance_agent = build_domain_agent(
    agent_key="finance",
    data_subdir="finance_agent_data",
    tool_description=(
        "Search the finance knowledge base. Use this for questions about "
        "student fee status, challans, dues, and payment history."
    ),
    system_prompt=(
        "You are the Finance Agent of a university management system. Your "
        "job is to handle fee management, challan generation details, and "
        "payment tracking. Always use your search tool to find the answer "
        "in the finance knowledge base before responding. If the information "
        "is not found, clearly say so."
    ),
)


# =========================================================
# 5. ADMIN AGENT
# =========================================================

admin_agent = build_domain_agent(
    agent_key="admin",
    data_subdir="admin_agent_data",
    tool_description=(
        "Search the administrative knowledge base. Use this for questions "
        "about faculty records, staff management, and system-level "
        "administrative data."
    ),
    system_prompt=(
        "You are the Admin Agent of a university management system. Your "
        "job is to manage system-level operations such as faculty "
        "management and administrative data control. Always use your "
        "search tool to find the answer in the admin knowledge base before "
        "responding. If the information is not found, clearly say so."
    ),
)


# =========================================================
# 6. ANALYTICS AGENT
# =========================================================

analytics_agent = build_domain_agent(
    agent_key="analytics",
    data_subdir="analytics_agent_data",
    tool_description=(
        "Search the analytics knowledge base. Use this for questions about "
        "performance prediction, attendance trends, and student risk "
        "analysis."
    ),
    system_prompt=(
        "You are the Analytics Agent of a university management system. "
        "Your job is to provide intelligent insights such as performance "
        "prediction, attendance trends, and risk analysis. Always use your "
        "search tool to find the answer in the analytics knowledge base "
        "before responding. If the information is not found, clearly say so."
    ),
)


# =========================================================
# REGISTRY (handy for looking agents up by name)
# =========================================================

AGENTS = {
    "admission": admission_agent,
    "examination": examination_agent,
    "attendance": attendance_agent,
    "finance": finance_agent,
    "admin": admin_agent,
    "analytics": analytics_agent,
}


# =========================================================
# QUICK MANUAL TEST
# =========================================================

if __name__ == "__main__":

    # Change this to test a different agent / query
    agent_name = "examination"
    query = "What is the CGPA of Zain Hassan?"

    print(f"\n[TEST] Running '{agent_name}' agent with query: {query}\n")

    response = AGENTS[agent_name].invoke({
        "messages": [
            {"role": "user", "content": query}
        ]
    })

    print(response["messages"][-1].content)