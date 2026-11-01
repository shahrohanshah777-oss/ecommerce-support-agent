"""check_order_status() lookups and the escalation-threshold justification.

Run:  python -m demos.order_status_tool
"""
import config  # noqa: F401

import numpy as np

import dataset
from demos._transcript import transcript
from tools.order_status import (
    ESCALATION_THRESHOLD, W_AGE, W_DELAY, check_order_status, escalation_score,
)


def main() -> None:
    print(f"escalation_score = {W_DELAY} * delayed_shipment + {W_AGE} * (days_since_created / 30)")
    print(f"escalate when escalation_score >= {ESCALATION_THRESHOLD}\n")

    print("Sample lookups:")
    samples = ["NYK-1001", "NYK-1003", "nyk-1002"]
    samples += [next(o["record_id"] for o in dataset.ORDERS if o["delayed_shipment"] and o["days_since_created"] >= 12)]
    samples += ["NYK-9999"]
    for rid in samples:
        print(f"  check_order_status({rid!r}) -> {check_order_status(rid)}")

    scores = np.array([escalation_score(o["delayed_shipment"], o["days_since_created"]) for o in dataset.ORDERS])
    days = np.array([o["days_since_created"] for o in dataset.ORDERS])
    delayed_days = sorted(o["days_since_created"] for o in dataset.ORDERS if o["delayed_shipment"])
    flagged = [o for o, s in zip(dataset.ORDERS, scores) if s >= ESCALATION_THRESHOLD]

    print("\nDistribution in the generated data (60 orders):")
    print(f"  days_since_created percentiles  p50={np.percentile(days, 50):.0f}  p75={np.percentile(days, 75):.0f}  "
          f"p80={np.percentile(days, 80):.0f}  p90={np.percentile(days, 90):.1f}")
    print(f"  escalation_score   percentiles  p50={np.percentile(scores, 50):.3f}  p75={np.percentile(scores, 75):.3f}  "
          f"p80={np.percentile(scores, 80):.3f}  p90={np.percentile(scores, 90):.3f}")
    print(f"  non-delayed orders: max score {scores[[not o['delayed_shipment'] for o in dataset.ORDERS]].max():.3f} "
          f"(can never reach {ESCALATION_THRESHOLD})")
    print(f"  ages of the {len(delayed_days)} delayed orders: {delayed_days}")
    gap_lo = max(d for d in delayed_days if d < 8)
    gap_hi = min(d for d in delayed_days if d >= 8)
    print(f"    -> two clusters, {min(delayed_days)}-{gap_lo} days and {gap_hi}-{max(delayed_days)} days; "
          f"scores {escalation_score(True, gap_lo):.3f} vs {escalation_score(True, gap_hi):.3f}")

    pct_rank = 100.0 * (scores < ESCALATION_THRESHOLD).mean()
    print(f"\nThreshold {ESCALATION_THRESHOLD}:")
    print(f"  = delayed AND days_since_created >= 8 (past the KB's 7-day upper delivery SLA)")
    print(f"  = the {pct_rank:.1f}th percentile of escalation_score in this data")
    print(f"  flags {len(flagged)}/60 orders ({100 * len(flagged) / 60:.1f}%): "
          f"{[o['record_id'] for o in flagged]}")
    assert all(o["delayed_shipment"] and o["days_since_created"] >= 8 for o in flagged)
    assert gap_lo < 8 <= gap_hi


if __name__ == "__main__":
    with transcript("order_status_tool"):
        main()
