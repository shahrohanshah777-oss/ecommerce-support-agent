"""LLM-as-judge evaluation of the full pipeline on the 15-query test set.

Run:  python -m demos.llm_judge_eval
Also writes eval/results_testset.json.
"""
import config

from demos._transcript import transcript
from eval.judge import JUDGE_PROMPT, mock_judge
from eval.runner import evaluate, print_details, print_table, save
from eval.testset import TESTSET
from rag.kb import REQUIRED_TOPICS


def main() -> None:
    covered = {c.topic for c in TESTSET}
    assert set(REQUIRED_TOPICS) <= covered, set(REQUIRED_TOPICS) - covered
    edge = [c for c in TESTSET if c.topic.startswith("edge")]
    print(f"Test set: {len(TESTSET)} queries - all {len(REQUIRED_TOPICS)} required KB topics + "
          f"{len(edge)} out-of-scope/edge cases | judge: LLM-as-judge prompt, MOCK_LLM={config.MOCK_LLM}\n")
    print("Judge prompt template:")
    print("  " + JUDGE_PROMPT.replace("\n", "\n  ") + "\n")

    results = evaluate(TESTSET)
    print_table(results)
    print(f"\nJudge calls: {mock_judge.calls}")
    print("\nPer-query details:")
    print_details(results)

    out = config.ROOT / "eval" / "results_testset.json"
    save(results, out)
    print(f"\nSaved {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    with transcript("llm_judge_eval"):
        main()
