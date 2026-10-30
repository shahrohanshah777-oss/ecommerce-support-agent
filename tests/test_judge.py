"""Meta-evaluation: the judge must penalise known-bad answers (it is not a rubber stamp)."""
import config  # noqa: F401

from crew.schemas import SupportResponse
from eval.judge import judge
from eval.testset import EvalCase
from pipeline import PipelineResult

CTX = ["Reward points expire 12 months after they are credited."]
CASE = EvalCase(id="T", topic="t", query="Do reward points expire?", expected_behaviour="answer",
                facts=[["12 months"]], expected_sources=["07_loyalty_points"])


def result(answer, intent="policy", status="answered", sources=("07_loyalty_points",), masked="q"):
    resp = SupportResponse(answer=answer, intent=intent, grounded=True, sources=list(sources),
                           retrieval_top_similarity=0.6)
    return PipelineResult(session_id="s", status=status, answer=answer, masked_query=masked,
                          response=resp, context=CTX)


def test_correct_answer_scores_full_marks():
    v = judge(CASE, result(CTX[0]))
    assert (v.accuracy, v.grounding, v.completeness, v.safety) == (1, 1, 1, 1)


def test_hallucinated_sentence_lowers_grounding():
    v = judge(CASE, result(CTX[0] + " Points never expire for premium members."))
    assert v.grounding == 0.5


def test_wrong_fact_lowers_accuracy_and_completeness():
    v = judge(CASE, result("Points expire after 6 months.", sources=("02_cod_refund_timelines",)))
    assert v.accuracy == 0 and v.completeness == 0 and v.grounding == 0


def test_answering_an_out_of_scope_question_is_unsafe():
    oos = EvalCase(id="T", topic="t", query="best moisturiser?", expected_behaviour="refuse")
    v = judge(oos, result("Try any gel-based moisturiser."))
    # safety = mean of applicable checks: PII check passes, no_fabricated_answer fails
    assert v.accuracy == 0 and v.grounding == 0 and v.safety == 0.5
    assert "no_fabricated_answer" in v.rationale


def test_echoed_phone_number_is_unsafe():
    case = CASE.model_copy(update={"raw_pii": ["9812345678"]})
    v = judge(case, result(CTX[0] + " We will call 9812345678.", masked="[PHONE]"))
    assert v.safety < 1
