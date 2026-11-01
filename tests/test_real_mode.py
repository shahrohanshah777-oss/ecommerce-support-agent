"""Real-LLM mode (MOCK_LLM=false) wiring, tested offline: nothing here calls a model API.

The one live test at the bottom runs only when MOCK_LLM=false and an API key is set.
"""
import os

import config

import pytest

import guardrails
import pipeline
from crew import crew as crew_mod
from crew.crew import CrewRun, build_agents
from crew.schemas import SupportResponse
from guardrails import check_groundedness, numbers_in, premise_units
from llm.providers import MissingAPIKeyError, api_key, plain_text
from review.autogen_review import REVIEW_FALLBACK, ReviewOutcome, ReviewVerdict

CTX = ["Apparel and Footwear can be returned within 15 days.", "Electronics can be returned within 7 days."]


@pytest.fixture
def real_mode(monkeypatch):
    monkeypatch.setattr(config, "MOCK_LLM", False)
    monkeypatch.setattr(config, "LLM_API_KEY_ENV", "NYKAA_TEST_FAKE_KEY")
    monkeypatch.setenv("NYKAA_TEST_FAKE_KEY", "test-key-not-real")


# --- configuration -----------------------------------------------------------------
def test_missing_key_fails_with_a_clear_message(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY_ENV", "NYKAA_TEST_UNSET_KEY")
    monkeypatch.delenv("NYKAA_TEST_UNSET_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="NYKAA_TEST_UNSET_KEY"):
        api_key()


def test_mock_is_the_default():
    assert os.environ.get("MOCK_LLM", "true").lower() == "true" and config.MOCK_LLM


# --- crew ----------------------------------------------------------------------------
def test_real_mode_agents_use_the_configured_endpoint(real_mode):
    agents = build_agents()
    for agent in agents.values():
        assert type(agent.llm).__name__ == "OpenAICompletion"
        assert agent.llm.model == config.LLM_MODEL
        assert str(agent.llm.base_url).rstrip("/") == config.LLM_BASE_URL.rstrip("/")
    # tool JSON reaches the Composer verbatim; least autonomy is unchanged
    assert [t.name for t in agents["retrieval"].tools] == ["rag_lookup"]
    assert [t.name for t in agents["lookup"].tools] == ["check_order_status"]
    assert agents["composer"].tools == []
    assert all(t.result_as_answer for a in agents.values() for t in a.tools)


def test_mock_mode_agents_are_unchanged():
    agents = build_agents()
    assert all(type(a.llm).__name__ == "MockLLM" for a in agents.values())
    assert not any(t.result_as_answer for a in agents.values() for t in a.tools)


def test_real_mode_instructions_follow_the_query_line(real_mode):
    text = crew_mod._describe("Do the task.\nCustomer query: hi", "composer")
    assert text.startswith("Do the task.\nCustomer query: hi\n\n")
    assert "Never add policies" in text


def test_mock_mode_task_text_is_unchanged():
    assert crew_mod._describe("Do the task.\nCustomer query: hi", "lookup") == "Do the task.\nCustomer query: hi"


# --- groundedness ----------------------------------------------------------------------
def test_numbers_compare_by_value():
    assert numbers_in("Rs 6,269 and 0.70, NYK-1004") == {6269.0, 0.7, 1004.0}
    assert numbers_in("v1.5.9") == set()


def test_plain_text_maps_typographic_unicode():
    assert plain_text("15 days, 5‑7 days—it’s") == "15 days, 5-7 days - it's"


def test_json_mode_reply_is_not_normalised(monkeypatch):
    # curly quotes inside a JSON string must survive, or json.loads fails
    import json
    from types import SimpleNamespace

    from llm import providers

    reply = '{"rationale": "says “15 days”"}'
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))], usage=None)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
    monkeypatch.setattr(providers, "_client", fake)
    assert json.loads(providers.complete("p", json_mode=True).text)["rationale"] == "says “15 days”"
    assert providers.complete("p").text == '{"rationale": "says "15 days""}'


def test_nli_inputs_ignore_thousands_separators():
    assert guardrails._nli_text("Rs 6,269 of 1,2345") == "Rs 6269 of 1,2345"


def test_premise_units_include_pairs_and_the_whole_evidence():
    units = premise_units(CTX, ["Order NYK-1 is Placed."])
    assert CTX[0] in units and f"{CTX[0]} {CTX[1]}" in units
    assert " ".join(CTX + ["Order NYK-1 is Placed."]) in units


def _fake_nli(prob):
    def support(sentences, premises, known):
        return [(s, prob, numbers_in(s) - known) for s in sentences]
    return support


def test_entailed_paraphrase_passes(monkeypatch):
    monkeypatch.setattr(guardrails, "entailment_support", _fake_nli(0.99))
    r = check_groundedness("You can return shoes within 15 days.", "policy", 0.6, 0.34, CTX, [], semantic=True)
    assert r.passed and r.entailment == {"You can return shoes within 15 days.": 0.99}


def test_low_entailment_is_refused(monkeypatch):
    monkeypatch.setattr(guardrails, "entailment_support", _fake_nli(0.2))
    r = check_groundedness("You can return shoes whenever you like.", "policy", 0.6, 0.34, CTX, [], semantic=True)
    assert not r.passed and r.unsupported == ["You can return shoes whenever you like."]


def test_low_entailment_is_advisory_when_the_reviewer_ran(monkeypatch):
    monkeypatch.setattr(guardrails, "entailment_support", _fake_nli(0.01))
    r = check_groundedness("Standard delivery to Bangalore is fast.", "policy", 0.6, 0.34, CTX, [],
                           semantic=True, require_entailment=False)
    assert r.passed and r.entailment == {"Standard delivery to Bangalore is fast.": 0.01}
    r = check_groundedness("Shoes can be returned within 30 days.", "policy", 0.6, 0.34, CTX, [],
                           semantic=True, require_entailment=False)
    assert not r.passed  # the numbers rule is always enforced


def test_pipeline_enforces_nli_only_without_review(monkeypatch):
    seen = {}
    monkeypatch.setattr(pipeline, "check_groundedness", lambda **kw: seen.update(kw) or None)
    monkeypatch.setattr(config, "MOCK_LLM", False)
    for review_on in (True, False):
        monkeypatch.setattr(pipeline, "REVIEW_ENABLED", review_on)
        pipeline.verify_draft(_fake_run("x"))
        assert seen["semantic"] and seen["require_entailment"] is (not review_on)


def test_invented_number_is_refused_even_if_entailed(monkeypatch):
    monkeypatch.setattr(guardrails, "entailment_support", _fake_nli(0.99))
    r = check_groundedness("You can return shoes within 30 days.", "policy", 0.6, 0.34, CTX, [], semantic=True)
    assert not r.passed


def test_number_from_the_customer_query_is_allowed(monkeypatch):
    monkeypatch.setattr(guardrails, "entailment_support", _fake_nli(0.99))
    r = check_groundedness("After 20 days, shoes can no longer be returned.", "policy", 0.6, 0.34, CTX, [],
                           semantic=True, query="I bought shoes 20 days ago, can I return them?")
    assert r.passed


def test_empty_answer_is_never_grounded():
    assert not check_groundedness("", "policy", 0.6, 0.34, CTX, []).passed


def test_mock_mode_never_loads_the_nli_model(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("NLI used in mock mode")
    monkeypatch.setattr(guardrails, "entailment_support", boom)
    assert not check_groundedness("Shoes can be returned any time.", "policy", 0.6, 0.34, CTX, []).passed


# --- review stage ------------------------------------------------------------------------
def _fake_run(answer: str) -> CrewRun:
    response = SupportResponse(answer=answer, intent="policy", grounded=True, sources=["01_return_window"],
                               order=None, escalate=False, retrieval_top_similarity=0.6)
    retrieval = {"grounded": True, "top_similarity": 0.6, "context": CTX, "sources": ["01_return_window"]}
    return CrewRun(query="q", response=response, retrieval=retrieval, lookup={}, tool_calls={})


@pytest.mark.parametrize("approved, edited, expected", [
    (True, CTX[0], CTX[0]),          # edited but still "approved": the edit is applied
    (False, "", REVIEW_FALLBACK),    # editor removed everything: fallback, never an empty reply
])
def test_editor_output_is_applied(monkeypatch, approved, edited, expected):
    draft = f"{CTX[0]} Free gift wrapping is included."
    monkeypatch.setattr(pipeline, "run_crew", lambda q: _fake_run(draft))
    monkeypatch.setattr(pipeline, "review_run", lambda run: ReviewOutcome(
        verdict=ReviewVerdict(approved=approved, final_answer=edited, reason="r"), conversation=[]))
    out = pipeline._respond("q")
    assert out["response"].answer == expected and out["draft_answer"] == draft
    if expected == REVIEW_FALLBACK:  # nothing survived review: reported as a refusal, not "answered"
        assert out["status"] == "refused_ungrounded"


# --- live (opt-in) ---------------------------------------------------------------------------
@pytest.mark.skipif(config.MOCK_LLM or not os.environ.get(config.LLM_API_KEY_ENV),
                    reason="live test: needs MOCK_LLM=false and an API key")
def test_live_policy_question_is_answered_and_grounded():
    r = pipeline.handle("Can I cancel my order NYK-1004?", session_id="live-test")
    assert r.status == "answered" and r.response.order and r.response.order.record_id == "NYK-1004"
