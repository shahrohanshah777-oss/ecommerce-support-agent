"""The full request path.

    handle(query, session_id)
      1. mask_pii          phone / card digits replaced before anything else sees them
      2. detect_injection  blocked requests never reach the agents or the memory
      3. check_budget      runtime governance: oversized requests rejected before any LLM call
      4. memory chain      history-aware query resolution (RunnableWithMessageHistory)
      5. run_crew          Retrieval -> Lookup -> Composer, SupportResponse draft
      6. Autogen review    Policy-Compliance-Reviewer -> Final-Editor: approve or revise
      7. groundedness      every sentence of the final answer must be backed by evidence
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

import config
from crew.crew import CrewRun, run_crew, template_sentences
from crew.schemas import OrderInfo, SupportResponse
from governance import BUDGET_REFUSAL, BudgetDecision, check_budget, estimated_cost_inr
from guardrails import (
    INJECTION_REFUSAL, UNGROUNDED_REFUSAL, GroundednessResult,
    check_groundedness, detect_injection, mask_pii,
)
from memory import build_memory_chain, get_session_history
from review.autogen_review import ReviewVerdict, review_draft


class PipelineResult(BaseModel):
    session_id: str
    status: Literal["answered", "blocked_injection", "refused_ungrounded", "rejected_budget"]
    answer: str
    masked_query: str
    resolved_query: str | None = None
    pii_masked: dict[str, int] = {}
    injection_matched: list[str] = []
    budget: BudgetDecision | None = None
    draft_answer: str | None = None
    review: ReviewVerdict | None = None
    groundedness: GroundednessResult | None = None
    response: SupportResponse | None = None
    tool_calls: dict[str, list[str]] = {}
    context: list[str] = []  # retrieved chunks the answer may draw on (for evaluation)
    tokens_used: int = 0
    cost_inr: float = 0.0
    history_messages: int = 0


def _tool_order(run: CrewRun) -> tuple[OrderInfo | None, str | None]:
    lookup = run.lookup
    order = OrderInfo(**{k: lookup[k] for k in OrderInfo.model_fields}) if lookup.get("found") else None
    missing = lookup.get("record_id") if lookup.get("record_id") and not lookup.get("found") else None
    return order, missing


def verify_draft(run: CrewRun) -> GroundednessResult:
    """Output-side groundedness check against the tools' own outputs (not the draft's)."""
    order, missing = _tool_order(run)
    r = run.response
    return check_groundedness(
        answer=r.answer,
        intent=r.intent,
        top_similarity=r.retrieval_top_similarity,
        threshold=config.IDK_THRESHOLD,
        context=run.retrieval.get("context", []),
        allowed_sentences=template_sentences(order, missing),
    )


def review_run(run: CrewRun):
    """Task 14: Autogen review of the Composer's draft against the original context."""
    order, missing = _tool_order(run)
    return review_draft(run.response.answer, run.retrieval.get("context", []), template_sentences(order, missing))


REVIEW_ENABLED = True  # demos may switch the Autogen stage off to isolate the Task 10 guardrail


def _respond(resolved_query: str) -> dict:
    run = run_crew(resolved_query)
    draft = run.response.answer
    outcome = review_run(run) if REVIEW_ENABLED else None
    if outcome and not outcome.verdict.approved:
        run.response = run.response.model_copy(update={"answer": outcome.verdict.final_answer})
    verdict = verify_draft(run)
    return {
        "answer": run.response.answer if verdict.passed else UNGROUNDED_REFUSAL,
        "status": "answered" if verdict.passed else "refused_ungrounded",
        "draft_answer": draft,
        "review": outcome.verdict if outcome else None,
        "groundedness": verdict,
        "response": run.response,
        "tool_calls": run.tool_calls,
        "context": run.retrieval.get("context", []),
        "tokens_used": run.tokens_used + (outcome.tokens_used if outcome else 0),
    }


_chain = build_memory_chain(_respond)


def handle(query: str, session_id: str = "default") -> PipelineResult:
    masked = mask_pii(query)
    base = dict(session_id=session_id, masked_query=masked.text, pii_masked=masked.findings)
    injection = detect_injection(query)
    if injection.is_injection:
        return PipelineResult(**base, status="blocked_injection", answer=INJECTION_REFUSAL,
                              injection_matched=injection.matched,
                              history_messages=len(get_session_history(session_id).messages))
    budget = check_budget(masked.text)
    if not budget.allowed:
        return PipelineResult(**base, status="rejected_budget", answer=BUDGET_REFUSAL, budget=budget,
                              history_messages=len(get_session_history(session_id).messages))

    out = _chain.invoke({"input": masked.text}, config={"configurable": {"session_id": session_id}})
    return PipelineResult(
        **base,
        status=out["status"],
        answer=out["answer"],
        resolved_query=out["resolved_query"],
        budget=budget,
        draft_answer=out["draft_answer"],
        review=out["review"],
        groundedness=out["groundedness"],
        response=out["response"],
        tool_calls=out["tool_calls"],
        context=out["context"],
        tokens_used=out["tokens_used"],
        cost_inr=estimated_cost_inr(out["tokens_used"]),
        history_messages=len(get_session_history(session_id).messages),
    )
