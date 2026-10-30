"""Task 4 - grounded generation over the retrieved context only.

Under MOCK_LLM the generator is extractive: it may only copy sentences that are
present in the retrieved chunks, ranked by embedding similarity to the query.
That makes every answer grounded by construction; the only judgement call is
whether retrieval found anything relevant, which is the calibrated
IDK_THRESHOLD on top-1 cosine similarity.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

import config
from rag.index import embed, search
from rag.kb import split_sentences
from rag.lexical import keyword_overlap

IDK_ANSWER = (
    "I don't know - that question isn't covered by the Nykaa support knowledge base. "
    "Please contact Nykaa customer support for help."
)

PROMPT_TEMPLATE = """You are a Nykaa customer-support assistant.
Answer the question using ONLY the context below. If the context does not contain
the answer, reply exactly: "I don't know".

Context:
{context}

Question: {query}
Answer:"""

LEXICAL_WEIGHT = 0.4  # sentence score = dense cosine + 0.4 * IDF keyword overlap
MAX_ANSWER_SENTENCES = 2
SECOND_SENTENCE_MARGIN = 0.05  # include a 2nd sentence only if nearly as relevant
MIN_FRAGMENT_WORDS = 6  # drop window-edge scraps like "Reward points..."
MAX_WORD_OVERLAP = 0.5  # a 2nd sentence must add new content, not repeat the 1st


class RetrievedChunk(BaseModel):
    id: str
    doc_id: str
    title: str
    text: str
    similarity: float


class GroundedAnswer(BaseModel):
    query: str
    collection: str
    answer: str
    grounded: bool
    top_similarity: float
    threshold: float
    sources: list[str]
    context: list[RetrievedChunk]


class MockGenerator:
    """Extractive stand-in for an LLM. Counts calls (used by the Task 16 cache demo)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt: str, query: str, chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
        self.calls += 1
        candidates = []  # (sentence, doc_id, title)
        for c in chunks:
            for s in split_sentences(c.text):
                if len(s.split()) >= MIN_FRAGMENT_WORDS and all(s != x[0] for x in candidates):
                    candidates.append((s, c.doc_id, c.title))
        if not candidates:  # every piece was a scrap: fall back to the best chunk itself
            candidates = [(chunks[0].text, chunks[0].doc_id, chunks[0].title)]
        # Dense score uses the same "<title>: <text>" header as indexing; the
        # keyword term separates sentences that share a document title.
        vecs = np.array(embed([query] + [f"{t}: {s}" for s, _, t in candidates]))
        scores = vecs[1:] @ vecs[0] + LEXICAL_WEIGHT * np.array(
            [keyword_overlap(query, s) for s, _, _ in candidates]
        )
        order = [int(i) for i in np.argsort(-scores)]
        picked = [order[0]]
        for i in order[1:]:
            if len(picked) == MAX_ANSWER_SENTENCES or scores[i] < scores[order[0]] - SECOND_SENTENCE_MARGIN:
                break
            if _word_overlap(candidates[i][0], candidates[picked[0]][0]) <= MAX_WORD_OVERLAP:
                picked.append(i)
        sentences = [_tidy(candidates[i][0]) for i in picked]
        sources = list(dict.fromkeys(candidates[i][1] for i in picked))
        return " ".join(sentences), sources


def _word_overlap(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    return len(wa & wb) / max(1, min(len(wa), len(wb)))


def _tidy(fragment: str) -> str:
    """Fixed-size windows can start/end mid-sentence; mark that with ellipses."""
    out = fragment.strip()
    if out and out[0].islower():
        out = "..." + out
    if out and out[-1] not in ".!?":
        out += "..."
    return out


generator = MockGenerator()


def answer(query: str, collection: str = config.RECOMMENDED_COLLECTION, k: int = config.TOP_K) -> GroundedAnswer:
    hits = search(query, collection, k)
    chunks = [RetrievedChunk(**{key: h[key] for key in ("id", "doc_id", "title", "text", "similarity")}) for h in hits]
    top = chunks[0].similarity if chunks else 0.0
    base = dict(query=query, collection=collection, top_similarity=top,
                threshold=config.IDK_THRESHOLD, context=chunks)

    if top < config.IDK_THRESHOLD:
        return GroundedAnswer(answer=IDK_ANSWER, grounded=False, sources=[], **base)

    # Only chunks that clear the threshold are allowed into the context.
    usable = [c for c in chunks if c.similarity >= config.IDK_THRESHOLD]
    prompt = PROMPT_TEMPLATE.format(context="\n".join(f"- {c.text}" for c in usable), query=query)
    text, sources = generator(prompt, query, usable)
    return GroundedAnswer(answer=text, grounded=True, sources=sources, **base)
