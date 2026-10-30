"""Task 4 (part 1) - calibrate the "I don't know" threshold empirically.

Measures top-1 cosine similarity for in-scope and out-of-scope queries on both
collections and checks that config.IDK_THRESHOLD lies in the observed gap.

Run:  python -m demos.task04_calibrate
"""
import config

from demos._transcript import transcript
from rag.index import search

IN_SCOPE = [
    "How many days do I have to return a pair of shoes?",
    "When will I get my money back for a cash on delivery order?",
    "How long does standard delivery take to Bangalore?",
    "Is there a warranty on a hair dryer?",
    "Can I cancel my order after it has been shipped?",
    "How do I redeem my reward points?",
    "My payment failed but money was deducted",
    "Can I exchange a dress for a bigger size?",
    "The lipstick I received was broken",
    "Do you ship to Dubai?",
    "How do I escalate my complaint?",
    "Is free pickup available for my return?",
]
OUT_OF_SCOPE = [
    "What is the capital of France?",
    "Write a Python function to reverse a linked list",
    "Who won the cricket world cup in 2011?",
    "What will the weather be like in Mumbai tomorrow?",
    "Explain quantum entanglement in simple terms",
    "Give me a recipe for butter chicken",
    "What is the stock price of Reliance today?",
    "Recommend a good gaming laptop under 80000",
    "How do I file my income tax return?",
    "Translate good morning into French",
]


def measure(collection: str) -> tuple[list[float], list[float]]:
    print(f"\n=== Collection: {collection}")
    groups = {}
    for label, queries in (("IN ", IN_SCOPE), ("OUT", OUT_OF_SCOPE)):
        sims = []
        for q in queries:
            hit = search(q, collection, k=1)[0]
            sims.append(hit["similarity"])
            print(f"  {label} top1={hit['similarity']:.3f}  -> {hit['doc_id']:26s} | {q}")
        groups[label] = sims
    lo_in, hi_out = min(groups["IN "]), max(groups["OUT"])
    print(f"  in-scope  : min {lo_in:.3f}  max {max(groups['IN ']):.3f}  mean {sum(groups['IN '])/len(IN_SCOPE):.3f}")
    print(f"  out-scope : min {min(groups['OUT']):.3f}  max {hi_out:.3f}  mean {sum(groups['OUT'])/len(OUT_OF_SCOPE):.3f}")
    print(f"  gap       : ({hi_out:.3f}, {lo_in:.3f})  width {lo_in - hi_out:.3f}  midpoint {(lo_in + hi_out) / 2:.4f}")
    return groups["IN "], groups["OUT"]


def main() -> None:
    print(f"Embedding model: {config.EMBED_MODEL} | chunks embedded as '<doc title>: <chunk>'")
    for collection in (config.COLLECTION_FIXED, config.COLLECTION_SENTENCE):
        ins, outs = measure(collection)
        assert max(outs) < config.IDK_THRESHOLD < min(ins), "threshold not inside observed gap"
        for preset in (0.5, 0.6, 0.7):
            wrongly_refused = sum(s < preset for s in ins)
            print(f"  tutorial preset {preset}: would wrongly refuse {wrongly_refused}/{len(ins)} in-scope queries")
    print(f"\nChosen IDK_THRESHOLD = {config.IDK_THRESHOLD} - inside the observed gap for BOTH collections.")
    print("Caveat: the gap is narrow (~0.02); see README for discussion.")


if __name__ == "__main__":
    with transcript("task04_calibrate"):
        main()
