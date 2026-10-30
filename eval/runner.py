"""Shared evaluation loop: run a test set through pipeline.handle(), judge each answer,
print per-query scores + averages, and optionally save them as JSON."""
from __future__ import annotations

import json
from pathlib import Path

from eval.judge import judge
from eval.testset import EvalCase
from pipeline import handle

METRICS = ("accuracy", "grounding", "completeness", "safety")


def evaluate(testset: list[EvalCase], session_prefix: str = "eval") -> dict:
    rows = []
    for case in testset:
        r = handle(case.query, session_id=f"{session_prefix}-{case.id}")  # fresh session per query
        v = judge(case, r)
        rows.append({"id": case.id, "topic": case.topic, "query": case.query, "status": r.status,
                     "intent": r.response.intent if r.response else None,
                     "sources": r.response.sources if r.response else [], "answer": r.answer,
                     **v.model_dump()})
    averages = {m: sum(row[m] for row in rows) / len(rows) for m in METRICS}
    return {"averages": averages, "rows": rows}


def print_table(results: dict) -> None:
    rows = results["rows"]
    print(f"{'ID':4s} {'topic':38s} {'Acc':>5s} {'Grnd':>5s} {'Comp':>5s} {'Safe':>5s}")
    print("-" * 66)
    for row in rows:
        print(f"{row['id']:4s} {row['topic'][:38]:38s} " + " ".join(f"{row[m]:5.2f}" for m in METRICS))
    print("-" * 66)
    avgs = results["averages"]
    print(f"{'AVG':4s} {f'({len(rows)} queries)':38s} " + " ".join(f"{avgs[m]:5.2f}" for m in METRICS))


def print_details(results: dict) -> None:
    for row in results["rows"]:
        print(f"\n[{row['id']}] {row['query']}")
        print(f"  status={row['status']} intent={row['intent']} sources={row['sources']}")
        print(f"  answer: {row['answer']}")
        print(f"  facts found={row['facts_found']} missing={row['facts_missing']}")
        print(f"  judge: {row['rationale']}")


def save(results: dict, path: Path) -> None:
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
