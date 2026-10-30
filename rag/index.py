"""Task 3 - embed chunks locally and index each strategy in its own ChromaDB collection.

Run:  python -m rag.index      (rebuilds both collections)
"""
from __future__ import annotations

from functools import lru_cache

import config
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from rag.cache import answer_cache
from rag.chunking import STRATEGIES, chunk_corpus, chunk_document


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load the embedding model from the local cache; download it only on first use."""
    try:
        return SentenceTransformer(config.EMBED_MODEL, local_files_only=True)
    except OSError:
        print(f"[index] {config.EMBED_MODEL} not cached yet - downloading once (~90 MB)...")
        return SentenceTransformer(config.EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


@lru_cache(maxsize=1)
def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
    )


def get_collection(name: str):
    return get_client().get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def _embedding_text(chunk: dict) -> str:
    """Contextual chunk header: the parent document's title is embedded with the
    chunk (stored text stays the raw chunk). Short policy sentences rarely repeat
    their topic, e.g. 'Nykaa currently delivers only to addresses within India'
    never says 'shipping' - its title does."""
    return f"{chunk['metadata']['title']}: {chunk['text']}"


def _upsert(collection, chunks: list[dict]) -> None:
    collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embed([_embedding_text(c) for c in chunks]),
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


def build_index() -> dict[str, int]:
    """(Re)index the whole KB into both collections. Returns chunk counts."""
    counts = {}
    for name in STRATEGIES:
        collection = get_collection(name)
        chunks = chunk_corpus(name)
        stale = set(collection.get(include=[])["ids"]) - {c["id"] for c in chunks}
        if stale:
            collection.delete(ids=sorted(stale))
        _upsert(collection, chunks)
        counts[name] = collection.count()
    answer_cache.clear()
    return counts


def upsert_document(doc: dict) -> dict[str, int]:
    """Add or update a single document in both collections (used by /add-document)."""
    added = {}
    for name in STRATEGIES:
        collection = get_collection(name)
        old = collection.get(where={"doc_id": doc["doc_id"]}, include=[])["ids"]
        if old:
            collection.delete(ids=old)
        chunks = chunk_document(doc, name)
        _upsert(collection, chunks)
        added[name] = len(chunks)
    answer_cache.clear()  # cached answers (incl. "I don't know") may now be stale
    return added


def ensure_index() -> None:
    if any(get_collection(name).count() == 0 for name in STRATEGIES):
        build_index()


def search(query: str, collection: str, k: int = config.TOP_K) -> list[dict]:
    """Top-k chunks with cosine similarity (1 - cosine distance), best first."""
    ensure_index()
    res = get_collection(collection).query(
        query_embeddings=embed([query]), n_results=k, include=["documents", "metadatas", "distances"]
    )
    return [
        {
            "id": cid,
            "text": text,
            "doc_id": meta["doc_id"],
            "title": meta["title"],
            "similarity": round(1.0 - dist, 4),
        }
        for cid, text, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


if __name__ == "__main__":
    print(build_index())
