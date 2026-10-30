"""Task 7 - the 3-agent CrewAI crew run via .kickoff(), with both tools invoked
on different queries.

Run:  python -m demos.task07_crew
"""
import config

from crew.crew import run_crew
from demos._transcript import transcript
from rag.generate import answer

ORDER_STATUS_ONLY = ["Where is my order NYK-1009?", "What is the status of NYK-1004?",
                     "Has NYK-1033 shipped yet?", "Track my order NYK-1012 please"]
ORDER_PLUS_POLICY = ["Can I cancel my order NYK-1004?", "I want to return order NYK-1002, how many days do I have?",
                     "My order NYK-1009 is delayed, can I escalate?", "When will I get the refund for NYK-1020?"]

DEMO_QUERIES = [
    ("policy question  -> RAG tool only", "How many reward points can I use on an order?"),
    ("order question   -> lookup tool", "Where is my order NYK-1009?"),
    ("order + policy   -> both tools", "Can I cancel my order NYK-1004?"),
    ("unknown order ID -> lookup reports not found", "What is the status of order NYK-9999?"),
    ("out of scope     -> fallback", "Can you recommend a good movie for tonight?"),
]


def calibrate_order_policy_threshold() -> None:
    print("Calibrating ORDER_POLICY_MIN_SIM (when to append policy to an order answer):")
    groups = {}
    for label, queries in (("status-only", ORDER_STATUS_ONLY), ("order+policy", ORDER_PLUS_POLICY)):
        groups[label] = []
        for q in queries:
            r = answer(q)
            groups[label].append(r.top_similarity)
            print(f"  {label:12s} top1={r.top_similarity:.3f} -> {r.sources} | {q}")
    hi, lo = max(groups["status-only"]), min(groups["order+policy"])
    print(f"  gap ({hi:.3f}, {lo:.3f}) midpoint {(hi + lo) / 2:.3f} -> ORDER_POLICY_MIN_SIM = {config.ORDER_POLICY_MIN_SIM}\n")
    assert hi < config.ORDER_POLICY_MIN_SIM < lo


def main() -> None:
    calibrate_order_policy_threshold()
    invoked = {"rag_lookup": set(), "check_order_status": set()}
    for label, q in DEMO_QUERIES:
        run = run_crew(q)
        print("=" * 100)
        print(f"[{label}] Q: {q}")
        print("  agent decisions (MockLLM ReAct steps):")
        for d in run.decisions:
            print(f"    {d}")
        print(f"  tool invocations: {run.tool_calls}")
        print(f"  LLM calls: {run.llm_calls}")
        print(f"  final draft ({type(run.response).__name__}):")
        print("    " + run.response.model_dump_json(indent=2).replace("\n", "\n    "))
        for tool, args in run.tool_calls.items():
            if args:
                invoked[tool].add(q)
    print("=" * 100)
    print("Queries on which each tool was actually invoked via .kickoff():")
    for tool, qs in invoked.items():
        print(f"  {tool}: {sorted(qs)}")
    assert "How many reward points can I use on an order?" in invoked["rag_lookup"]
    assert "How many reward points can I use on an order?" not in invoked["check_order_status"]
    assert "Where is my order NYK-1009?" in invoked["check_order_status"]


if __name__ == "__main__":
    with transcript("task07_crew"):
        main()
