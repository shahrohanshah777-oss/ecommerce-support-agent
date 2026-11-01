"""Four-layer AI governance: least autonomy (application layer), risk
classification, and a per-request token/cost budget (runtime layer).

Run:  python -m demos.governance_controls
"""
import config

from fastapi.testclient import TestClient

import pipeline
from api.main import app
from crew.crew import COMPOSER, LOOKUP, RETRIEVAL, build_agents
from demos._transcript import transcript
from governance import (
    RISK_JUSTIFICATION, RISK_LEVEL, TOOL_PERMISSIONS, ToolPermissionError, check_budget,
)
from tools.crew_tools import OrderStatusTool


def attempt(label: str, fn) -> None:
    try:
        fn()
        raise AssertionError(f"{label}: was NOT blocked")
    except ToolPermissionError as e:
        print(f"  BLOCKED  {label}\n           -> ToolPermissionError: {e}")


def least_autonomy() -> None:
    print("APPLICATION LAYER - least autonomy")
    print(f"  allow-list: { {k: sorted(v) for k, v in TOOL_PERMISSIONS.items()} }")
    agents = build_agents()
    for a in agents.values():
        print(f"  {a.role:18s} tools={[t.name for t in a.tools]}")
    print()
    attempt("wiring a check_order_status tool to the Response Composer (build time)",
            lambda: build_agents(extra_tools={"composer": [OrderStatusTool(bound_role=COMPOSER)]}))
    attempt("handing the Lookup Agent's own tool instance to the Retrieval Agent (build time)",
            lambda: build_agents(extra_tools={"retrieval": [OrderStatusTool(bound_role=LOOKUP)]}))
    attempt("invoking check_order_status from a tool issued to the Retrieval Agent (call time)",
            lambda: OrderStatusTool(bound_role=RETRIEVAL)._run(record_id="NYK-1009"))
    attempt("invoking an unbound check_order_status tool (call time, fail closed)",
            lambda: OrderStatusTool()._run(record_id="NYK-1009"))
    ok = OrderStatusTool(bound_role=LOOKUP)._run(record_id="NYK-1009")
    print(f"  ALLOWED  Lookup Agent -> check_order_status('NYK-1009'): {ok[:80]}...\n")


def risk() -> None:
    print(f"RISK CLASSIFICATION: {RISK_LEVEL}")
    words, line = RISK_JUSTIFICATION.split(), "  "
    for w in words:
        if len(line) + len(w) > 98:
            print(line)
            line = "  "
        line += w + " "
    print(line + "\n")


def budget() -> None:
    print(f"RUNTIME LAYER - per-request token budget = {config.TOKEN_BUDGET:,} tokens "
          f"(estimate = {config.BUDGET_OVERHEAD_TOKENS:,} + {config.BUDGET_QUERY_REPEATS} x query tokens; "
          f"illustrative cost Rs {config.COST_PER_1K_TOKENS_INR}/1K tokens)")
    normal = "Can I cancel order NYK-1001?"
    r = pipeline.handle(normal, "budget-normal")
    print(f"  normal request : {normal!r}")
    print(f"    pre-flight   : {r.budget.reason} (estimated {r.budget.estimated_tokens:,} tokens, "
          f"Rs {r.budget.estimated_cost_inr})")
    print(f"    metered      : {r.tokens_used:,} tokens actually used (Rs {r.cost_inr}), status={r.status}")
    assert r.status == "answered" and r.tokens_used <= r.budget.estimated_tokens <= config.TOKEN_BUDGET

    pasted = "Please read my whole complaint history carefully before answering. " * 180
    oversized = pasted + "Where is my order NYK-1009?"
    r = pipeline.handle(oversized, "budget-oversized")
    print(f"\n  oversized request: {len(oversized):,} characters (a pasted complaint log + a question)")
    print(f"    pre-flight   : {r.budget.reason}")
    print(f"    status       : {r.status} | tools invoked: {r.tool_calls or 'none'} | "
          f"tokens used: {r.tokens_used} | review/crew ran: {r.review is not None}")
    print(f"    answer       : {r.answer}")
    assert r.status == "rejected_budget" and r.tokens_used == 0 and not r.tool_calls

    resp = TestClient(app).post("/ask", json={"query": oversized})
    print(f"\n  same request via POST /ask -> HTTP {resp.status_code} "
          f"({resp.json()['detail'][0]['msg']}) - the API's max_length is a second, earlier cap")
    edge = "x" * 2000
    print(f"  largest request the API accepts (2,000 chars): {check_budget(edge).reason} "
          f"(estimated {check_budget(edge).estimated_tokens:,})")
    assert resp.status_code == 422 and check_budget(edge).allowed


def main() -> None:
    least_autonomy()
    risk()
    budget()


if __name__ == "__main__":
    with transcript("governance_controls"):
        main()
