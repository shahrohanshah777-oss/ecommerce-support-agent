"""Regenerate every transcript in transcripts/ (MOCK_LLM, no API keys, no network).

Each demo runs in its own fresh Python process so no state (session memory,
response cache, counters) leaks between demos.

Run:  python run_all_demos.py
"""
import subprocess
import sys
import time

DEMOS = [
    "demos.mock_llm_spike",
    "demos.order_dataset",
    "demos.knowledge_base",
    "demos.chunk_indexing",
    "demos.idk_calibration",
    "demos.grounded_answers",
    "demos.retrieval_eval",
    "demos.order_status_tool",
    "demos.crew_kickoff",
    "demos.session_memory",
    "demos.structured_output",
    "demos.guardrail_checks",
    "demos.api_and_logging",
    "demos.llm_judge_eval",
    "demos.eval_v2_compare",  # eval_v2_baseline is frozen evidence and deliberately not re-run
    "demos.autogen_review",
    "demos.governance_controls",
    "demos.response_cache",
]


def main() -> int:
    failed = []
    for module in DEMOS:
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, "-m", module], capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        ok = proc.returncode == 0
        print(f"{'OK  ' if ok else 'FAIL'} {module:34s} {time.perf_counter() - t0:6.1f}s")
        if not ok:
            failed.append(module)
            print(proc.stdout[-1500:], proc.stderr[-1500:], sep="\n")
    print(f"\n{len(DEMOS) - len(failed)}/{len(DEMOS)} demos passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
