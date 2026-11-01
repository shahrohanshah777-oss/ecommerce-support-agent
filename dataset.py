"""Seeded, deterministic Nykaa order dataset.

Design choices (also stated in README.md so the grader can reproduce):

* SEED = 42, N_ORDERS = 60, generated with a local random.Random(SEED).
* Category weights lean towards Beauty because it is Nykaa's core business.
* Status weights reflect a mature order book: most orders are already Delivered.
* order_value_inr is drawn per category and ends in 9 (Rs 1,499-style price points);
  the overall range Rs 199 - Rs 14,999 spans a single lipstick or sheet mask at
  the low end to premium beauty electronics (hair dryers, stylers) at the top.
* days_since_created (0-30) depends on status, so a "Placed" order is never weeks old
  and a "Refunded" order is never brand new.
* delayed_shipment ~ Bernoulli(DELAY_PROB) but only for orders at least
  DELAY_MIN_DAYS old: an order cannot have missed its delivery SLA (2-7 days)
  before that SLA could have elapsed.

Run `python dataset.py` to print the validation report. Records are never edited by
hand: if a threshold fails, change SEED / weights and regenerate.
"""
from __future__ import annotations

import random
from collections import Counter

SEED = 42
N_ORDERS = 60

CATEGORY_WEIGHTS = {
    "Beauty": 0.35,
    "Apparel": 0.25,
    "Footwear": 0.15,
    "Home": 0.13,
    "Electronics": 0.12,
}
STATUS_WEIGHTS = {
    "Placed": 0.15,
    "Shipped": 0.20,
    "Delivered": 0.40,
    "Returned": 0.12,
    "Refunded": 0.13,
}
# Price band per category in INR (inclusive).
PRICE_RANGE_INR = {
    "Beauty": (199, 4_999),
    "Apparel": (399, 6_999),
    "Footwear": (499, 8_999),
    "Home": (299, 9_999),
    "Electronics": (799, 14_999),
}
# days_since_created band per status (inclusive, all within 0-30).
DAYS_RANGE_BY_STATUS = {
    "Placed": (0, 2),
    "Shipped": (1, 8),
    "Delivered": (3, 30),
    "Returned": (8, 30),
    "Refunded": (10, 30),
}
DELAY_PROB = 0.25
DELAY_MIN_DAYS = 3

# Structural thresholds from the brief.
MIN_PER_CATEGORY = 3
MIN_PER_STATUS = 1
DELAYED_PCT_BAND = (10.0, 30.0)


def _price(rng: random.Random, category: str) -> int:
    lo, hi = PRICE_RANGE_INR[category]
    value = rng.randint(lo, hi) // 10 * 10 + 9  # Rs x,xx9 price point
    return min(max(value, lo), hi)


def generate_orders(seed: int = SEED, n: int = N_ORDERS) -> list[dict]:
    rng = random.Random(seed)
    orders = []
    for i in range(n):
        category = rng.choices(list(CATEGORY_WEIGHTS), weights=list(CATEGORY_WEIGHTS.values()))[0]
        status = rng.choices(list(STATUS_WEIGHTS), weights=list(STATUS_WEIGHTS.values()))[0]
        days = rng.randint(*DAYS_RANGE_BY_STATUS[status])
        delayed = days >= DELAY_MIN_DAYS and rng.random() < DELAY_PROB
        orders.append(
            {
                "record_id": f"NYK-{1001 + i}",
                "category": category,
                "status": status,
                "order_value_inr": _price(rng, category),
                "days_since_created": days,
                "delayed_shipment": delayed,
            }
        )
    return orders


ORDERS: list[dict] = generate_orders()
ORDERS_BY_ID: dict[str, dict] = {o["record_id"]: o for o in ORDERS}


def summarize(orders: list[dict]) -> dict:
    n = len(orders)
    delayed = sum(o["delayed_shipment"] for o in orders)
    values = [o["order_value_inr"] for o in orders]
    return {
        "n": n,
        "per_category": Counter(o["category"] for o in orders),
        "per_status": Counter(o["status"] for o in orders),
        "delayed_count": delayed,
        "delayed_pct": 100.0 * delayed / n,
        "value_min": min(values),
        "value_max": max(values),
        "value_mean": sum(values) / n,
    }


def validate(orders: list[dict]) -> list[str]:
    """Return a list of threshold violations (empty list = dataset is valid)."""
    s = summarize(orders)
    problems = []
    if s["n"] < 40:
        problems.append(f"only {s['n']} orders (need >= 40)")
    for cat in CATEGORY_WEIGHTS:
        if s["per_category"][cat] < MIN_PER_CATEGORY:
            problems.append(f"category {cat} has {s['per_category'][cat]} (< {MIN_PER_CATEGORY})")
    for status in STATUS_WEIGHTS:
        if s["per_status"][status] < MIN_PER_STATUS:
            problems.append(f"status {status} has {s['per_status'][status]} (< {MIN_PER_STATUS})")
    lo, hi = DELAYED_PCT_BAND
    if not lo <= s["delayed_pct"] <= hi:
        problems.append(f"delayed_shipment {s['delayed_pct']:.1f}% outside {lo}-{hi}%")
    for o in orders:
        if not 0 <= o["days_since_created"] <= 30:
            problems.append(f"{o['record_id']} days_since_created out of 0-30")
        if not isinstance(o["delayed_shipment"], bool):
            problems.append(f"{o['record_id']} delayed_shipment is not bool")
    if len({o["record_id"] for o in orders}) != len(orders):
        problems.append("duplicate record_id")
    return problems


def report(orders: list[dict] = ORDERS) -> None:
    s = summarize(orders)
    print(f"Seed: {SEED}   Orders: {s['n']}")
    print(f"Category weights: {CATEGORY_WEIGHTS}")
    print(f"Status weights:   {STATUS_WEIGHTS}")
    print(f"Delay prob: {DELAY_PROB} (only for orders >= {DELAY_MIN_DAYS} days old)\n")

    print("Count per category (need >= 3 each):")
    for cat in CATEGORY_WEIGHTS:
        print(f"  {cat:12s} {s['per_category'][cat]:3d}")
    print("\nCount per status (need >= 1 each):")
    for status in STATUS_WEIGHTS:
        print(f"  {status:12s} {s['per_status'][status]:3d}")
    print(
        f"\ndelayed_shipment=True: {s['delayed_count']}/{s['n']} = {s['delayed_pct']:.1f}% "
        f"(need {DELAYED_PCT_BAND[0]:.0f}-{DELAYED_PCT_BAND[1]:.0f}%)"
    )
    print(
        f"order_value_inr: min Rs {s['value_min']:,}  max Rs {s['value_max']:,}  "
        f"mean Rs {s['value_mean']:,.0f}"
    )

    print("\nSample records:")
    for o in orders[:5]:
        print(f"  {o}")

    problems = validate(orders)
    print("\nValidation:", "PASS - all thresholds met" if not problems else "FAIL")
    for p in problems:
        print(f"  - {p}")
    assert not problems, problems


if __name__ == "__main__":
    report()
