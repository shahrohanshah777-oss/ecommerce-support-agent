"""Baseline score of the system on the held-out set v2, recorded BEFORE the
answer-selection improvements. Committed on its own so git history shows the
held-out queries and this baseline existed before the code changed.

Run:  python -m demos.eval_v2_baseline
"""
import config

from demos._transcript import transcript
from eval.runner import evaluate, print_details, print_table, save
from eval.testset import TESTSET_V2
from rag.kb import REQUIRED_TOPICS


def main() -> None:
    assert set(REQUIRED_TOPICS) <= {c.topic for c in TESTSET_V2}
    print(f"Held-out set v2: {len(TESTSET_V2)} queries (12 KB topics + 3 edge cases), never used for tuning.")
    print("System state: before the answer-selection improvements.\n")
    results = evaluate(TESTSET_V2, session_prefix="v2base")
    print_table(results)
    print("\nPer-query details:")
    print_details(results)
    out = config.ROOT / "eval" / "results_v2_baseline.json"
    save(results, out)
    print(f"\nSaved {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    import sys

    frozen = config.ROOT / "eval" / "results_v2_baseline.json"
    if frozen.exists() and "--force" not in sys.argv:
        sys.exit(f"{frozen.relative_to(config.ROOT)} is the frozen pre-improvement baseline; refusing to "
                 "overwrite it with the current system's scores (pass --force to override). "
                 "Use `python -m demos.eval_v2_compare` for before/after.")
    with transcript("eval_v2_baseline"):
        main()
