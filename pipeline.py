"""The full request path: guardrails -> memory -> crew -> groundedness check.

    handle(query, session_id)
      1. mask_pii          phone / card digits replaced before anything else sees them
      2. detect_injection  blocked requests never reach the agents or the memory
      3. memory chain      history-aware query resolution (RunnableWithMessageHistory)
      4. run_crew          Retrieval -> Lookup -> Composer, SupportResponse draft
      5. groundedness      every answer sentence must be backed by evidence
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

import config
from crew.crew import CrewRun, run_crew, template_sentences
from crew.schemas import OrderInfo, SupportResponse
from guardrails import (
    INJECTION_REFUSAL, UNGROUNDED_REFUSAL, GroundednessResult,
    check_groundedness, detect_injection, mask_pii,
)
from memory import build_memory_chain, get_session_history


class PipelineResult(BaseModel):
    session_id: str
    status: Literal["answered", "blocked_injection", "refused_ungrounded"]
    answer: str
    masked_query: str
    resolved_query: str | None = None
    pii_masked: dict[str, int] = {}
    injection_matched: list[str] = []
    groundedness: GroundednessResult | None = None
    response: SupportResponse | None = None
    tool_calls: dict[str, list[str]] = {}
    history_messages: int = 0


def verify_draft(run: CrewRun) -> GroundednessResult:
    """Output-side groundedness check against the tools' own outputs (not the draft's)."""
    lookup = run.lookup
    tool_order = OrderInfo(**{k: lookup[k] for k in OrderInfo.model_fields}) if lookup.get("found") else None
    missing_id = lookup.get("record_id") if lookup.get("record_id") and not lookup.get("found") else None
    r = run.response
    return check_groundedness(
        answer=r.answer,
        intent=r.intent,
        top_similarity=r.retrieval_top_similarity,
        threshold=config.IDK_THRESHOLD,
        context=run.retrieval.get("context", []),
        allowed_sentences=template_sentences(tool_order, missing_id),
    )


def _respond(resolved_query: str) -> dict:
    run = run_crew(resolved_query)
    verdict = verify_draft(run)
    return {
        "answer": run.response.answer if verdict.passed else UNGROUNDED_REFUSAL,
        "status": "answered" if verdict.passed else "refused_ungrounded",
        "groundedness": verdict,
        "response": run.response,
        "tool_calls": run.tool_calls,
    }


_chain = build_memory_chain(_respond)


def handle(query: str, session_id: str = "default") -> PipelineResult:
    masked = mask_pii(query)
    injection = detect_injection(query)
    if injection.is_injection:
        return PipelineResult(
            session_id=session_id, status="blocked_injection", answer=INJECTION_REFUSAL,
            masked_query=masked.text, pii_masked=masked.findings, injection_matched=injection.matched,
            history_messages=len(get_session_history(session_id).messages),
        )
    out = _chain.invoke({"input": masked.text}, config={"configurable": {"session_id": session_id}})
    return PipelineResult(
        session_id=session_id,
        status=out["status"],
        answer=out["answer"],
        masked_query=masked.text,
        resolved_query=out["resolved_query"],
        pii_masked=masked.findings,
        groundedness=out["groundedness"],
        response=out["response"],
        tool_calls=out["tool_calls"],
        history_messages=len(get_session_history(session_id).messages),
    )
