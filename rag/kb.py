"""Knowledge-base loading. Each kb/*.md file is one document:
a `# Title` line followed by a 2-5 sentence policy paragraph."""
from __future__ import annotations

import re

import config

# Required topic (from the brief) -> document that covers it.
REQUIRED_TOPICS = {
    "return window by product category": "01_return_window",
    "COD refund timelines": "02_cod_refund_timelines",
    "delivery SLAs": "03_delivery_sla",
    "reverse-pickup eligibility": "04_reverse_pickup",
    "warranty terms by category": "05_warranty_terms",
    "order-cancellation policy": "06_order_cancellation",
    "loyalty-points redemption policy": "07_loyalty_points",
    "payment-failure/retry policy": "08_payment_failure_retry",
    "size-exchange policy": "09_size_exchange",
    "damaged-item claim process": "10_damaged_item_claims",
    "international shipping restrictions": "11_international_shipping",
    "customer-support escalation matrix": "12_escalation_matrix",
}

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences. The KB avoids abbreviations with
    periods (e.g. 'Rs' not 'Rs.'), so a punctuation + capital rule is exact."""
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]


def answer_sentences(answer: str) -> list[str]:
    """The quoted pieces of an extractive answer. Fixed-window fragments are joined
    with '...' and may start lower-case, so split on '...' first, then on sentences."""
    text = answer.replace("Relevant policy: ", "")
    return [s.strip() for piece in text.split("...") for s in split_sentences(piece) if s.strip()]


def load_documents(kb_dir=config.KB_DIR) -> list[dict]:
    docs = []
    for path in sorted(kb_dir.glob("*.md")):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        title = lines[0].lstrip("#").strip()
        body = " ".join(line.strip() for line in lines[1:] if line.strip())
        docs.append({"doc_id": path.stem, "title": title, "text": body})
    return docs
