"""Task 5 - document-level precision & recall for BOTH collections on the Task 4 queries.

Run:  python -m demos.task05_retrieval_eval
"""
import config

from demos._transcript import transcript
from rag.eval_retrieval import evaluate


def main() -> None:
    print(f"top_k = {config.TOP_K} chunks -> mapped to parent doc_id -> de-duplicated -> scored\n")
    results = {}
    for collection in (config.COLLECTION_FIXED, config.COLLECTION_SENTENCE):
        res = evaluate(collection)
        results[collection] = res
        print("=" * 100)
        print(f"Collection: {collection}")
        print("=" * 100)
        for i, r in enumerate(res["rows"], 1):
            print(f"Q{i}: {r['query']}")
            print(f"    top-{config.TOP_K} chunk docs : {r['chunk_docs']}")
            print(f"    unique retrieved: {r['retrieved']}")
            print(f"    relevant        : {r['relevant']}")
            print(f"    precision = {r['tp']}/{len(r['retrieved'])} = {r['precision']:.2f}   "
                  f"recall = {r['tp']}/{len(r['relevant'])} = {r['recall']:.2f}")
        p = [f"{r['precision']:.2f}" for r in res["rows"]]
        rc = [f"{r['recall']:.2f}" for r in res["rows"]]
        print(f"\nMean precision = ({' + '.join(p)}) / {len(p)} = {res['mean_precision']:.3f}")
        print(f"Mean recall    = ({' + '.join(rc)}) / {len(rc)} = {res['mean_recall']:.3f}\n")

    print("Summary")
    print(f"  {'collection':12s} {'mean P':>7s} {'mean R':>7s}")
    for name, res in results.items():
        print(f"  {name:12s} {res['mean_precision']:7.3f} {res['mean_recall']:7.3f}")


if __name__ == "__main__":
    with transcript("task05_retrieval_eval"):
        main()
