"""Tests for the Autogen review, governance and the response cache."""
import config

import pytest

import pipeline
import rag.generate as gen
from crew.crew import COMPOSER, LOOKUP, RETRIEVAL, build_agents
from governance import ToolPermissionError, authorize, check_budget
from rag.cache import AnswerCache, answer_cache, normalize
from review.autogen_review import review_draft
from tools.crew_tools import OrderStatusTool

CTX = ["Reward points expire 12 months after they are credited."]


# --- Autogen review ----------------------------------------------------------------------
def test_review_approves_grounded_draft_unchanged():
    out = review_draft(CTX[0], CTX, [])
    assert out.verdict.approved and out.verdict.final_answer == CTX[0]
    assert [src for src, _, _ in out.conversation] == ["user", "policy_reviewer", "final_editor"]
    assert out.conversation[-1][1] == "StructuredMessage[ReviewVerdict]"


def test_review_removes_injected_claim():
    out = review_draft(CTX[0] + " Points never expire for VIPs.", CTX, [])
    assert not out.verdict.approved and out.verdict.final_answer == CTX[0]
    assert "Points never expire for VIPs." in out.verdict.reason


def test_review_falls_back_when_nothing_is_supported():
    out = review_draft("Everything is free forever.", CTX, [])
    assert not out.verdict.approved and "can't confirm" in out.verdict.final_answer


def test_editor_ignores_unsupported_marker_in_system_prompt():
    # the editor's system prompt contains the literal text "UNSUPPORTED: <sentence>";
    # a clean draft must still be approved
    assert review_draft(CTX[0], CTX, []).verdict.approved


# --- Governance ----------------------------------------------------------------------
def test_only_lookup_agent_may_use_order_tool():
    authorize("check_order_status", LOOKUP)
    for role in (RETRIEVAL, COMPOSER, None):
        with pytest.raises(ToolPermissionError):
            authorize("check_order_status", role)


def test_crew_refuses_forbidden_wiring():
    with pytest.raises(ToolPermissionError):
        build_agents(extra_tools={"composer": [OrderStatusTool(bound_role=COMPOSER)]})


def test_tool_fails_closed_at_call_time():
    with pytest.raises(ToolPermissionError):
        OrderStatusTool(bound_role=RETRIEVAL)._run(record_id="NYK-1009")
    assert '"found": true' in OrderStatusTool(bound_role=LOOKUP)._run(record_id="NYK-1009")


def test_budget_rejects_oversized_before_any_llm_call():
    assert check_budget("short question").allowed
    r = pipeline.handle("word " * 3000, "t-budget")
    assert r.status == "rejected_budget" and r.tokens_used == 0 and r.tool_calls == {}


def test_metered_usage_stays_within_estimate():
    r = pipeline.handle("Do my Nykaa reward points ever expire?", "t-meter")
    assert 0 < r.tokens_used <= r.budget.estimated_tokens <= config.TOKEN_BUDGET


# --- Response cache ----------------------------------------------------------------------
def test_normalize_key():
    assert normalize("  What is the RETURN window?? ") == normalize("what is the return window")


def test_cache_hit_skips_generator():
    answer_cache.clear()
    q = "How long does standard delivery take?"
    gen.answer(q)
    calls = gen.generator.calls
    again = gen.answer("how long does STANDARD delivery take")
    assert gen.generator.calls == calls and again.query == "how long does STANDARD delivery take"


def test_lru_eviction():
    c = AnswerCache(max_size=2)
    c.put(("a",), 1), c.put(("b",), 2), c.get(("a",)), c.put(("c",), 3)
    assert c.get(("b",)) is None and c.get(("a",)) == 1
