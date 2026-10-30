"""Task 14 - Autogen review stage: approve a draft unchanged, and revise a draft that
contains a deliberately injected ungrounded claim.

Team: RoundRobinGroupChat([policy_reviewer, final_editor], max_turns=2,
      custom_message_types=[StructuredMessage[ReviewVerdict]]); the final_editor
      has output_content_type=ReviewVerdict (approved, final_answer, reason).

Run:  python -m demos.task14_review
"""
import config  # noqa: F401

import pipeline
from crew.crew import run_crew
from demos._transcript import transcript
from review.autogen_review import build_team

INJECTED = "Home products can also be exchanged for free within 60 days."


def show_review(title: str, run, outcome) -> None:
    print(f"  draft (Composer)   : {run.response.answer}")
    print(f"  retrieved context  : {len(run.retrieval.get('context', []))} chunk(s)")
    print("  team conversation  :")
    for source, kind, content in outcome.conversation:
        text = content if source != "user" else content.split("RETRIEVED CONTEXT")[0].strip()
        print(f"    [{source} / {kind}]")
        for line in text.splitlines():
            print(f"      {line}")
    v = outcome.verdict
    print(f"  ReviewVerdict      : approved={v.approved}")
    print(f"                       final_answer={v.final_answer!r}")
    print(f"                       reason={v.reason!r}\n")


def main() -> None:
    team, _ = build_team()
    print(f"Team: {type(team).__name__} participants={[p.name for p in team._participants]} "
          f"max_turns={team._max_turns}")
    print("custom_message_types=[StructuredMessage[ReviewVerdict]]; final_editor output_content_type=ReviewVerdict\n")

    print("CASE 1 - grounded draft -> approved unchanged")
    run = run_crew("Do my Nykaa reward points ever expire?")
    outcome = pipeline.review_run(run)
    show_review("approve", run, outcome)
    assert outcome.verdict.approved and outcome.verdict.final_answer == run.response.answer

    print("CASE 2 - order draft (facts from the lookup tool) -> approved unchanged")
    run = run_crew("Where is my order NYK-1027?")
    outcome = pipeline.review_run(run)
    show_review("approve-order", run, outcome)
    assert outcome.verdict.approved

    print("CASE 3 - draft with a deliberately injected ungrounded claim -> revised")
    run = run_crew("How many days do I get to return a home decor item?")
    clean = run.response.answer
    run.response = run.response.model_copy(update={"answer": f"{clean} {INJECTED}"})
    outcome = pipeline.review_run(run)
    show_review("revise", run, outcome)
    assert not outcome.verdict.approved
    assert INJECTED not in outcome.verdict.final_answer and outcome.verdict.final_answer == clean

    print("CASE 3 again, end to end through pipeline.handle() with a tampering Composer:")
    original = pipeline.run_crew

    def tampering_crew(q):
        r = original(q)
        r.response = r.response.model_copy(update={"answer": f"{r.response.answer} {INJECTED}"})
        return r

    pipeline.run_crew = tampering_crew
    try:
        r = pipeline.handle("How many days do I get to return a home decor item?", "review-demo")
    finally:
        pipeline.run_crew = original
    print(f"  draft     : {r.draft_answer}")
    print(f"  review    : approved={r.review.approved} ({r.review.reason})")
    print(f"  guardrail : passed={r.groundedness.passed}")
    print(f"  delivered : [{r.status}] {r.answer}")
    assert r.status == "answered" and INJECTED not in r.answer
    print("\nThe review stage repaired the draft, so the customer gets the corrected answer instead of a refusal;")
    print("the output groundedness guardrail (Task 10) still runs afterwards as the last line of defence.")


if __name__ == "__main__":
    with transcript("task14_review"):
        main()
