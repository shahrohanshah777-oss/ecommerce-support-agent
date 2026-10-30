"""Task 2 - knowledge-base inventory: documents, sentence counts, topic coverage.

Run:  python -m demos.task02_kb
"""
import config  # noqa: F401

from demos._transcript import transcript
from rag.kb import REQUIRED_TOPICS, load_documents, split_sentences


def main() -> None:
    docs = load_documents()
    topic_of = {doc_id: topic for topic, doc_id in REQUIRED_TOPICS.items()}
    print(f"Knowledge base: {len(docs)} documents in kb/ (need >= 12)\n")
    print(f"{'doc_id':28s} {'sent':>4s}  required topic covered")
    print("-" * 80)
    for d in docs:
        n = len(split_sentences(d["text"]))
        print(f"{d['doc_id']:28s} {n:4d}  {topic_of.get(d['doc_id'], '(extra) ' + d['title'])}")
        assert 2 <= n <= 5, f"{d['doc_id']} has {n} sentences (need 2-5)"

    ids = {d["doc_id"] for d in docs}
    missing = [t for t, doc_id in REQUIRED_TOPICS.items() if doc_id not in ids]
    assert len(docs) >= 12 and not missing, missing
    print(f"\nAll {len(REQUIRED_TOPICS)} required topics covered; every document has 2-5 sentences.")

    sample = docs[0]
    print(f"\nSample document - {sample['doc_id']} ({sample['title']}):")
    for i, s in enumerate(split_sentences(sample["text"]), 1):
        print(f"  [{i}] {s}")


if __name__ == "__main__":
    with transcript("task02_kb"):
        main()
