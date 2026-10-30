"""Task 3 - two chunking strategies over the knowledge base.

* fixed:    sliding windows of FIXED_CHUNK_WORDS words, overlapping by
            FIXED_CHUNK_OVERLAP words (windows ignore sentence boundaries).
* sentence: one chunk per sentence.

Every chunk keeps its parent `doc_id` so retrieval can be scored per document.
"""
from __future__ import annotations

import config
from rag.kb import load_documents, split_sentences


def fixed_chunks(text: str, size: int = config.FIXED_CHUNK_WORDS,
                 overlap: int = config.FIXED_CHUNK_OVERLAP) -> list[str]:
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def sentence_chunks(text: str) -> list[str]:
    return split_sentences(text)


STRATEGIES = {
    config.COLLECTION_FIXED: fixed_chunks,
    config.COLLECTION_SENTENCE: sentence_chunks,
}


def chunk_document(doc: dict, collection: str) -> list[dict]:
    pieces = STRATEGIES[collection](doc["text"])
    return [
        {
            "id": f"{doc['doc_id']}::{collection}::{i}",
            "text": piece,
            "metadata": {"doc_id": doc["doc_id"], "title": doc["title"], "chunk_index": i},
        }
        for i, piece in enumerate(pieces)
    ]


def chunk_corpus(collection: str, docs: list[dict] | None = None) -> list[dict]:
    docs = docs if docs is not None else load_documents()
    return [c for d in docs for c in chunk_document(d, collection)]
