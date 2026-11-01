"""Document-level precision/recall for each chunking strategy.

Top-k chunks are mapped to their parent doc_id and de-duplicated before scoring:
    precision = |retrieved docs ∩ relevant docs| / |retrieved docs|
    recall    = |retrieved docs ∩ relevant docs| / |relevant docs|
"""
from __future__ import annotations

import config
from rag.index import search

# The same 5 in-scope queries demonstrated in demos/grounded_answers.py, with hand-labelled relevant docs.
EVAL_QUERIES = {
    "What is the return window for electronics?": {"01_return_window"},
    "How long does a COD refund take to reach my bank account?": {"02_cod_refund_timelines"},
    "Can I still cancel my order if the shipment is delayed?": {"06_order_cancellation", "03_delivery_sla"},
    "How many reward points can I use on an order?": {"07_loyalty_points"},
    "I received a damaged hair straightener, what should I do?": {"10_damaged_item_claims"},
}


def score_query(query: str, relevant: set[str], collection: str, k: int = config.TOP_K) -> dict:
    hits = search(query, collection, k)
    chunk_docs = [h["doc_id"] for h in hits]
    retrieved = list(dict.fromkeys(chunk_docs))  # dedup, keep rank order
    hit_docs = [d for d in retrieved if d in relevant]
    return {
        "query": query,
        "chunk_docs": chunk_docs,
        "retrieved": retrieved,
        "relevant": sorted(relevant),
        "tp": len(hit_docs),
        "precision": len(hit_docs) / len(retrieved),
        "recall": len(hit_docs) / len(relevant),
    }


def evaluate(collection: str, k: int = config.TOP_K) -> dict:
    rows = [score_query(q, rel, collection, k) for q, rel in EVAL_QUERIES.items()]
    return {
        "collection": collection,
        "rows": rows,
        "mean_precision": sum(r["precision"] for r in rows) / len(rows),
        "mean_recall": sum(r["recall"] for r in rows) / len(rows),
    }
