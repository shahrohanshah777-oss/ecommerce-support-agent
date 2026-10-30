"""Task 4 (part 2) - grounded generation on 5 in-scope queries + 1 out-of-scope query,
for both collections.

Run:  python -m demos.task04_grounded
"""
import config

from demos._transcript import transcript
from rag.eval_retrieval import EVAL_QUERIES
from rag.generate import answer
from rag.kb import answer_sentences

OUT_OF_SCOPE = "Can you book me a train ticket to Delhi for tomorrow?"


def main() -> None:
    print(f"IDK_THRESHOLD = {config.IDK_THRESHOLD} (calibrated in task04_calibrate) | top_k = {config.TOP_K}\n")
    queries = list(EVAL_QUERIES) + [OUT_OF_SCOPE]
    for collection in (config.COLLECTION_SENTENCE, config.COLLECTION_FIXED):
        print("=" * 100)
        print(f"Collection: {collection}")
        print("=" * 100)
        for q in queries:
            r = answer(q, collection)
            tag = "IN-SCOPE " if q in EVAL_QUERIES else "OUT-SCOPE"
            print(f"\n[{tag}] Q: {q}")
            print("  retrieved: " + ", ".join(f"{c.doc_id}({c.similarity:.3f})" for c in r.context))
            print(f"  top1={r.top_similarity:.3f} vs threshold {r.threshold} -> grounded={r.grounded}")
            print(f"  A: {r.answer}")
            if r.sources:
                print(f"  sources: {r.sources}")
            if q in EVAL_QUERIES:
                assert r.grounded, f"in-scope query fell below threshold: {q}"
                # Grounding check: every quoted sentence is verbatim retrieved text.
                ctx = " ".join(r.evidence)
                for s in answer_sentences(r.answer):
                    assert s in ctx, f"answer text not in retrieved context: {s!r}"
            else:
                assert not r.grounded and r.answer.startswith("I don't know"), "fallback did not fire"
        print()
    print("All in-scope queries answered from retrieved context; out-of-scope query triggered the fallback.")


if __name__ == "__main__":
    with transcript("task04_grounded"):
        main()
