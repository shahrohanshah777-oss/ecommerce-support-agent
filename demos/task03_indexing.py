"""Task 3 - two chunking strategies, embedded locally, indexed in two ChromaDB collections.

Run:  python -m demos.task03_indexing
"""
import config

from demos._transcript import transcript
from rag.chunking import chunk_corpus
from rag.index import build_index, get_collection, search

SAMPLE_QUERY = "How long do I have to return shoes?"


def main() -> None:
    print(f"Embedding model : {config.EMBED_MODEL} (local SentenceTransformers, normalised vectors)")
    print(f"Fixed strategy  : {config.FIXED_CHUNK_WORDS}-word windows, {config.FIXED_CHUNK_OVERLAP}-word overlap")
    print("Sentence strategy: one chunk per sentence")
    print("Embedded text   : '<doc title>: <chunk>' (contextual header); stored text is the raw chunk\n")

    counts = build_index()  # collection.upsert() into two separate collections
    for name, n in counts.items():
        meta = get_collection(name).metadata
        print(f"Collection {name:12s}: {n} chunks  (distance: {meta['hnsw:space']})")

    print("\nExample - document 01_return_window chunked both ways:")
    for name in (config.COLLECTION_FIXED, config.COLLECTION_SENTENCE):
        print(f"  [{name}]")
        for c in chunk_corpus(name):
            if c["metadata"]["doc_id"] == "01_return_window":
                print(f"    {c['id']}: {c['text']}")

    print(f"\nSample query on both collections: {SAMPLE_QUERY!r}")
    for name in (config.COLLECTION_FIXED, config.COLLECTION_SENTENCE):
        print(f"  [{name}]")
        hits = search(SAMPLE_QUERY, name)
        for h in hits:
            print(f"    sim={h['similarity']:.3f}  {h['doc_id']:22s} {h['text'][:90]}")
        assert hits[0]["doc_id"] in {"01_return_window", "09_size_exchange"}


if __name__ == "__main__":
    with transcript("task03_indexing"):
        main()
