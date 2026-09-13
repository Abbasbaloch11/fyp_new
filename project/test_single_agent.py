"""
Interactive CLI to talk to ONE specialist agent directly,
bypassing the Supervisor. Useful for testing/debugging a
single domain (admission, examination, attendance, finance,
admin, analytics) without going through the Supervisor's
own routing decision call.

NOTE: Since each entry in AGENTS (from multi_agents.py) is a
full LangChain agent (not a raw RAGSearch engine), each
question here still costs ~3 Gemini calls internally
(agent's tool-call decision + RAGSearch's own answer +
agent's final response). Keep that in mind re: your daily
free-tier quota.
"""

from multi_agents import AGENTS
from src.utils import extract_text

AGENT_DESCRIPTIONS = {
    "admission": "Admissions, eligibility, enrollment status, academic history",
    "examination": "Results, GPA/CGPA, transcripts",
    "attendance": "Attendance records, attendance percentage, shortage cases",
    "finance": "Fees, challans, dues, payment history",
    "admin": "Faculty records, staff management, admin data",
    "analytics": "Performance prediction, attendance trends, risk analysis",
}


def print_menu():
    print("\n" + "=" * 55)
    print("SELECT AN AGENT TO TALK TO DIRECTLY")
    print("=" * 55)
    for i, key in enumerate(AGENTS.keys(), start=1):
        print(f"{i}. {key.capitalize()} Agent  -  {AGENT_DESCRIPTIONS[key]}")
    print("0. Exit")
    print("=" * 55)


def main():
    agent_keys = list(AGENTS.keys())

    while True:
        print_menu()
        choice = input("Enter the number of the agent: ").strip()

        if choice == "0":
            print("Goodbye.")
            break

        if not choice.isdigit() or not (1 <= int(choice) <= len(agent_keys)):
            print("Invalid choice, try again.")
            continue

        selected_key = agent_keys[int(choice) - 1]
        agent = AGENTS[selected_key]

        print(f"\n--- Now talking to: {selected_key.upper()} AGENT ---")
        print("(type 'back' to return to the agent menu)\n")

        while True:
            query = input(f"[{selected_key}] Your question: ").strip()

            if query.lower() == "back":
                break

            if not query:
                continue

            response = agent.invoke({
                "messages": [
                    {"role": "user", "content": query}
                ]
            })

            answer = extract_text(response["messages"][-1].content)

            print("\n" + "-" * 55)
            print(f"{selected_key.upper()} AGENT ANSWER:")
            print("-" * 55)
            print(answer)
            print("-" * 55 + "\n")


if __name__ == "__main__":
    main()