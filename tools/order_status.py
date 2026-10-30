"""Task 6 - order lookup with a designed escalation score.

    escalation_score = W_DELAY * delayed_shipment + W_AGE * (days_since_created / 30)
                     = 0.6 * delayed + 0.4 * age        (age in [0, 1])

* A delayed shipment is the dominant signal (0.6): a non-delayed order tops out
  at 0.4 and can never be escalated on age alone.
* Age breaks ties among delayed orders: the longer a delayed order has been
  open, the more urgent it is.
* ESCALATION_THRESHOLD = 0.70  <=>  delayed AND days_since_created >= 8, i.e. the
  order is late *and* older than the KB's 7-day upper delivery SLA. In the
  generated data the delayed orders' ages form two clusters (3-6 days and 12-24
  days) and 0.70 falls in the gap; see demos/task06_order_status.py for the
  percentile justification.
"""
from __future__ import annotations

import dataset

W_DELAY = 0.6
W_AGE = 0.4
MAX_DAYS = 30
ESCALATION_THRESHOLD = 0.70


def escalation_score(delayed_shipment: bool, days_since_created: int) -> float:
    age = min(max(days_since_created, 0), MAX_DAYS) / MAX_DAYS
    return round(W_DELAY * float(delayed_shipment) + W_AGE * age, 3)


def check_order_status(record_id: str) -> dict:
    """Look up one order. Returns status, order_value_inr and escalation_score."""
    rid = record_id.strip().upper()
    order = dataset.ORDERS_BY_ID.get(rid)
    if order is None:
        return {"found": False, "record_id": rid, "error": f"No order found with ID {rid}."}
    score = escalation_score(order["delayed_shipment"], order["days_since_created"])
    return {
        "found": True,
        "record_id": rid,
        "category": order["category"],
        "status": order["status"],
        "order_value_inr": order["order_value_inr"],
        "days_since_created": order["days_since_created"],
        "delayed_shipment": order["delayed_shipment"],
        "escalation_score": score,
        "escalate": score >= ESCALATION_THRESHOLD,
    }
