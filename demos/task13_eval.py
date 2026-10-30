"""Task 13 - LLM-as-judge evaluation of the full pipeline on 15 queries.

Run:  python -m demos.task13_eval
Also writes eval/results_task13.json.
"""
import json

import config

from demos._transcript import transcript
from eval.judge import JUDGE_PROMPT, judge, mock_judge
from eval.testset import TESTSET
from pipeline import handle
from rag.kb import REQUIRED_TOPICS

METRICS = ("accuracy", "grounding", "completeness", "safety")


def main() -> None:
    covered = {c.topic for c in TESTSET}
    assert set(REQUIRED_TOPICS) <= covered, set(REQUIRED_TOPICS) - covered
    edge = [c for c in TESTSET if c.topic.startswith("edge")]
    print(f"Test set: {len(TESTSET)} queries - all {len(REQUIRED_TOPICS)} required KB topics + "
          f"{len(edge)} out-of-scope/edge cases | judge: LLM-as-judge prompt, MOCK_LLM={config.MOCK_LLM}\n")
    print("Judge prompt template:")
    print("  " + JUDGE_PROMPT.replace("\n", "\n  ") + "\n")

    rows = []
    for case in TESTSET:
        r = handle(case.query, session_id=f"eval-{case.id}")  # fresh session per query
        v = judge(case, r)
        rows.append({"id": case.id, "topic": case.topic, "query": case.query, "status": r.status,
                     "intent": r.response.intent if r.response else None,
                     "sources": r.response.sources if r.response else [], "answer": r.answer,
                     **v.model_dump()})

    print(f"{'ID':4s} {'topic':38s} {'Acc':>5s} {'Grnd':>5s} {'Comp':>5s} {'Safe':>5s}")
    print("-" * 66)
    for row in rows:
        print(f"{row['id']:4s} {row['topic'][:38]:38s} " + " ".join(f"{row[m]:5.2f}" for m in METRICS))
    print("-" * 66)
    avgs = {m: sum(row[m] for row in rows) / len(rows) for m in METRICS}
    print(f"{'AVG':4s} {'(15 queries)':38s} " + " ".join(f"{avgs[m]:5.2f}" for m in METRICS))
    print(f"\nJudge calls: {mock_judge.calls}")

    print("\nPer-query details:")
    for row in rows:
        print(f"\n[{row['id']}] {row['query']}")
        print(f"  status={row['status']} intent={row['intent']} sources={row['sources']}")
        print(f"  answer: {row['answer']}")
        print(f"  facts found={row['facts_found']} missing={row['facts_missing']}")
        print(f"  judge: {row['rationale']}")

    out = config.ROOT / "eval" / "results_task13.json"
    out.write_text(json.dumps({"averages": avgs, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    with transcript("task13_eval"):
        main()
