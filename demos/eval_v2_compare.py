"""Before/after on the held-out set v2.

"Before" is read from eval/results_v2_baseline.json, which was committed (with the
v2 queries) before the answer-selection changes. "After" is the current system,
run live. The Task 13 set served as the development set for these changes, so
v2 is the honest measurement of whether they generalise.

Run:  python -m demos.eval_v2_compare     (writes eval/results_v2_after.json)
"""
import json

import config

from demos._transcript import transcript
from eval.runner import METRICS, evaluate, print_details, save
from eval.testset import TESTSET_V2

CHANGES = [
    "small-to-big retrieval: answer from the whole top-ranked document, not only the matched sentence",
    "always quote the 2 best sentences of that document, in document order (was: 2nd only within 0.05)",
    "order-ID hint narrowed to questions about one specific purchase ('return it', 'where is my order')",
]


def main() -> None:
    before = json.loads((config.ROOT / "eval" / "results_v2_baseline.json").read_text(encoding="utf-8"))
    print("Changes evaluated (designed on the Task 13 development set only):")
    for c in CHANGES:
        print(f"  - {c}")
    print(f"\nHeld-out set v2: {len(TESTSET_V2)} queries | before = frozen baseline, after = live run\n")
    after = evaluate(TESTSET_V2, session_prefix="v2after")

    print(f"{'ID':4s} {'topic':32s} " + " ".join(f"{m[:4].title():>11s}" for m in METRICS))
    print(f"{'':37s}" + " ".join(f"{'bef -> aft':>11s}" for _ in METRICS))
    print("-" * 85)
    for b, a in zip(before["rows"], after["rows"]):
        cells = []
        for m in METRICS:
            mark = "+" if a[m] > b[m] else "-" if a[m] < b[m] else " "
            cells.append(f"{b[m]:4.2f}->{a[m]:4.2f}{mark}")
        print(f"{a['id']:4s} {a['topic'][:32]:32s} " + " ".join(f"{c:>11s}" for c in cells))
    print("-" * 85)
    print(f"{'AVG':37s}" + " ".join(f"{before['averages'][m]:4.2f}->{after['averages'][m]:4.2f} " for m in METRICS))

    regressions = [(a["id"], m) for b, a in zip(before["rows"], after["rows"]) for m in METRICS if a[m] < b[m]]
    print(f"\nRegressions on any metric: {regressions or 'none'}")
    print("\nPer-query details (after):")
    print_details(after)
    out = config.ROOT / "eval" / "results_v2_after.json"
    save(after, out)
    print(f"\nSaved {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    with transcript("eval_v2_compare"):
        main()
