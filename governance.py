"""Task 15 - AI governance: application-layer least autonomy, risk classification,
and the runtime-layer per-request token/cost budget.

Application layer - least autonomy
    TOOL_PERMISSIONS is the single allow-list of which agent role may use which
    tool. It is enforced twice:
      1. at crew build time: enforce_least_autonomy() refuses to start a crew in
         which any agent holds a tool it is not allowed to use;
      2. at call time: every governed tool is bound to the role it was issued to
         and calls authorize() before doing anything, so a tool that is wired to
         the wrong agent some other way still fails closed.

Runtime layer - budget
    check_budget() runs before any LLM call: a request whose *estimated* token
    use exceeds config.TOKEN_BUDGET is rejected up front instead of being run
    and silently overspending. Actual usage is metered on every request.
"""
from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel

import config

TOOL_PERMISSIONS: dict[str, frozenset[str]] = {
    "check_order_status": frozenset({"Lookup Agent"}),
    "rag_lookup": frozenset({"Retrieval Agent"}),
}


class ToolPermissionError(PermissionError):
    pass


def authorize(tool_name: str, agent_role: str | None) -> None:
    allowed = TOOL_PERMISSIONS.get(tool_name, frozenset())
    if agent_role not in allowed:
        raise ToolPermissionError(
            f"least-autonomy policy: agent {agent_role!r} may not use tool {tool_name!r} "
            f"(allowed: {sorted(allowed) or 'nobody'})"
        )


def enforce_least_autonomy(agents: Iterable) -> None:
    for agent in agents:
        for tool in agent.tools or []:
            authorize(tool.name, agent.role)
            bound = getattr(tool, "bound_role", None)
            if bound != agent.role:
                raise ToolPermissionError(
                    f"tool {tool.name!r} was issued to {bound!r} but is wired to {agent.role!r}"
                )


# --- Risk classification ----------------------------------------------------------
RISK_LEVEL = "Medium"
RISK_JUSTIFICATION = (
    "Under the Low / Medium / High scheme this system is Medium risk. Its job is customer-support "
    "ticket handling, which the scheme places at Medium: it answers policy questions and reports "
    "order status, and a wrong answer can mislead a customer about a refund or return deadline, "
    "but it cannot act on that answer. It does not decide or move money: it issues no refunds, "
    "processes no payments and only reads a fabricated order table. It does handle financial-"
    "adjacent data, though (order values, card last-4 digits typed by customers), which is the "
    "trait the scheme lists under High. That is why High-tier controls are applied to that data "
    "even though the system is Medium overall: card digits and phone numbers are masked before "
    "any model, memory or log sees them; the order tool is restricted to one agent; every answer "
    "must pass a groundedness check and an independent review; and each request has a hard token "
    "budget. If the agent were ever allowed to issue refunds or change orders, it would become "
    "High risk and would need human approval for those actions."
)


# --- Runtime budget -----------------------------------------------------------------
class BudgetDecision(BaseModel):
    allowed: bool
    estimated_tokens: int
    budget: int
    estimated_cost_inr: float
    reason: str


def estimate_tokens(query: str) -> int:
    """Pre-flight estimate: fixed per-request overhead (prompts, tool output, review)
    plus the query, which is repeated in roughly QUERY_REPEATS model calls."""
    query_tokens = (len(query) + 3) // 4
    return config.BUDGET_OVERHEAD_TOKENS + config.BUDGET_QUERY_REPEATS * query_tokens


def estimated_cost_inr(tokens: int) -> float:
    return round(tokens / 1000 * config.COST_PER_1K_TOKENS_INR, 4)


def check_budget(query: str, budget: int | None = None) -> BudgetDecision:
    budget = config.TOKEN_BUDGET if budget is None else budget
    est = estimate_tokens(query)
    allowed = est <= budget
    return BudgetDecision(
        allowed=allowed, estimated_tokens=est, budget=budget, estimated_cost_inr=estimated_cost_inr(est),
        reason=("within budget" if allowed else
                f"estimated {est:,} tokens exceeds the per-request budget of {budget:,}"),
    )


BUDGET_REFUSAL = ("Your message is too long for me to process in one request. Please shorten it "
                  "to the key question (for example, your order ID and what you need).")
