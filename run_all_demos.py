"""Regenerate every transcript in transcripts/ (MOCK_LLM, no API keys, no network).

Each demo runs in its own fresh Python process so no state (session memory,
response cache, counters) leaks between tasks.

Run:  python run_all_demos.py
"""
import subprocess
import sys
import time

DEMOS = [
    "demos.phase1_mock_llm_spike",
    "demos.task01_dataset",
    "demos.task02_kb",
    "demos.task03_indexing",
    "demos.task04_calibrate",
    "demos.task04_grounded",
    "demos.task05_retrieval_eval",
    "demos.task06_order_status",
    "demos.task07_crew",
    "demos.task08_memory",
    "demos.task09_schema",
    "demos.task10_guardrails",
    "demos.task11_12_api",
    "demos.task13_eval",
    "demos.eval_v2_compare",  # eval_v2_baseline is frozen evidence and deliberately not re-run
    "demos.task14_review",
    "demos.task15_governance",
    "demos.task16_cache",
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
