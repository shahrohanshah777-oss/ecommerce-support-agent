"""Tests for Task 6 (order tool), Task 7 (crew) and Task 9 (schema)."""
import config  # noqa: F401

import pytest

import dataset
from crew.crew import compose_response, json_objects, run_crew
from crew.schemas import SupportResponse
from tools.order_status import ESCALATION_THRESHOLD, check_order_status, escalation_score


# --- Task 6 -----------------------------------------------------------------
def test_escalation_formula_combines_both_signals():
    assert escalation_score(False, 0) == 0.0
    assert escalation_score(False, 30) == 0.4  # age alone can never escalate
    assert escalation_score(True, 0) == 0.6
    assert escalation_score(True, 15) == 0.8
    assert escalation_score(True, 30) == 1.0


def test_threshold_means_delayed_and_past_sla():
    assert escalation_score(True, 7) < ESCALATION_THRESHOLD <= escalation_score(True, 8)


def test_check_order_status_lookup_and_not_found():
    rec = dataset.ORDERS[0]
    out = check_order_status(rec["record_id"].lower())
    assert out["found"] and out["status"] == rec["status"]
    assert out["order_value_inr"] == rec["order_value_inr"]
    assert 0.0 <= out["escalation_score"] <= 1.0
    assert check_order_status("NYK-9999")["found"] is False


# --- Task 7 / 9 -------------------------------------------------------------
def test_policy_query_invokes_only_rag_tool():
    run = run_crew("How many reward points can I use on an order?")
    assert run.tool_calls["rag_lookup"] and not run.tool_calls["check_order_status"]
    assert run.response.intent == "policy" and run.response.sources == ["07_loyalty_points"]


def test_order_query_invokes_lookup_tool():
    run = run_crew("Where is my order NYK-1009?")
    assert run.tool_calls["check_order_status"] == ["NYK-1009"]
    assert run.response.order.record_id == "NYK-1009" and run.response.escalate is True
    assert isinstance(SupportResponse.model_validate(run.response.model_dump()), SupportResponse)


def test_composer_never_invents_an_order():
    retrieval = {"grounded": True, "answer": "x", "sources": ["01"], "top_similarity": 0.9}
    resp = compose_response(retrieval, {"found": False, "record_id": "NYK-9999", "error": "nope"})
    assert resp.order is None and "couldn't find" in resp.answer and resp.sources == []


def test_json_objects_extracts_embedded_blobs():
    text = 'noise {"a": 1} more {"b": {"c": 2}} {bad'
    assert json_objects(text) == [{"a": 1}, {"b": {"c": 2}}]


def test_schema_rejects_extra_fields():
    with pytest.raises(Exception):
        SupportResponse.model_validate({"answer": "x", "intent": "policy", "grounded": True,
                                        "retrieval_top_similarity": 0.5, "coupon": "FREE"})
