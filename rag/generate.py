"""Task 4 - grounded generation over the retrieved context only.

Retrieval uses small chunks (one sentence each) because they match queries
precisely; generation uses "small-to-big" expansion: the generator sees every
chunk of the top-ranked document, not just the one sentence that matched.
Sentence chunks reliably find the right *document* but often miss the sentence
that holds the answer (diagnosed on the development set, see README).

Under MOCK_LLM the generator is extractive: it quotes the 2 best-scoring
sentences of that document verbatim, in document order. Every answer is
therefore grounded by construction; whether retrieval found anything relevant
at all is decided by the calibrated IDK_THRESHOLD on top-1 cosine similarity.
With MOCK_LLM=false the configured LLM answers PROMPT_TEMPLATE over the same
evidence, behind the same threshold.
"""
from __future__ import annotations

import re

import numpy as np
from pydantic import BaseModel

import config
from rag.cache import answer_cache, normalize
from rag.index import embed, parent_chunks, search
from rag.kb import split_sentences
from rag.lexical import keyword_overlap

IDK_ANSWER = (
    "I don't know - that question isn't covered by the Nykaa support knowledge base. "
    "Please contact Nykaa customer support for help."
)

PROMPT_TEMPLATE = """You are a Nykaa customer-support assistant.
Answer the question using ONLY the context below, in 1-3 complete sentences that
make sense on their own. If the context does not contain the answer, reply
exactly: "I don't know".

Context:
{context}

Question: {query}
Answer:"""

LEXICAL_WEIGHT = 0.4  # sentence score = dense cosine + 0.4 * IDF keyword overlap
MAX_ANSWER_SENTENCES = 2  # concise replies; quoting more would only game completeness
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
    context: list[RetrievedChunk]  # what retrieval returned (top-k)
    evidence: list[str] = []  # everything the generator was allowed to quote


class MockGenerator:
    """Extractive stand-in for an LLM. Counts calls (used by the Task 16 cache demo)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt: str, query: str, doc_chunks: list[str], doc_id: str, title: str) -> str:
        """Quote the best MAX_ANSWER_SENTENCES sentences of one document, in document order."""
        self.calls += 1
        candidates: list[str] = []
        for chunk in doc_chunks:
            for s in split_sentences(chunk):
                if len(s.split()) >= MIN_FRAGMENT_WORDS and s not in candidates:
                    candidates.append(s)
        if not candidates:  # every piece was a scrap: fall back to the first chunk itself
            candidates = [doc_chunks[0]]
        # Both scores see the same "<title>: <text>" form used at indexing time;
        # the keyword term separates sentences that share a document title.
        headed = [f"{title}: {s}" for s in candidates]
        vecs = np.array(embed([query] + headed))
        scores = vecs[1:] @ vecs[0] + LEXICAL_WEIGHT * np.array([keyword_overlap(query, h) for h in headed])
        picked: list[int] = []
        for i in (int(i) for i in np.argsort(-scores)):
            if len(picked) == MAX_ANSWER_SENTENCES:
                break
            if all(_word_overlap(candidates[i], candidates[j]) <= MAX_WORD_OVERLAP for j in picked):
                picked.append(i)
        return " ".join(_tidy(candidates[i]) for i in sorted(picked))


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


class LLMGenerator:
    """Real-LLM mode: the model answers PROMPT_TEMPLATE over the same evidence.
    Returns None when the model says it doesn't know."""

    def __init__(self) -> None:
        self.calls = 0
        self.tokens_used = 0

    def __call__(self, prompt: str) -> str | None:
        from llm.providers import complete

        self.calls += 1
        out = complete(prompt)
        self.tokens_used += out.tokens
        text = out.text.strip().strip('"').strip()
        return None if not text or text.lower().startswith("i don't know") else text


llm_generator = LLMGenerator()


def policy_query(query: str) -> str:
    """Order IDs never occur in the KB; they only add noise to retrieval and keyword scoring."""
    return " ".join(re.sub(config.ORDER_ID_PATTERN, " ", query, flags=re.IGNORECASE).split())


retrieval_calls = 0  # how many times answer() actually hit the vector index


def answer(query: str, collection: str = config.RECOMMENDED_COLLECTION, k: int = config.TOP_K,
           use_cache: bool = True) -> GroundedAnswer:
    original, query = query, policy_query(query)
    key = (normalize(query), collection, k)
    if use_cache and (cached := answer_cache.get(key)) is not None:
        return cached.model_copy(update={"query": original}, deep=True)
    result = _answer_uncached(original, query, collection, k)
    if use_cache:
        answer_cache.put(key, result.model_copy(deep=True))
    return result


def _answer_uncached(original: str, query: str, collection: str, k: int) -> GroundedAnswer:
    global retrieval_calls
    retrieval_calls += 1
    hits = search(query, collection, k)
    chunks = [RetrievedChunk(**{key: h[key] for key in ("id", "doc_id", "title", "text", "similarity")}) for h in hits]
    top = chunks[0].similarity if chunks else 0.0
    base = dict(query=original, collection=collection, top_similarity=top,
                threshold=config.IDK_THRESHOLD, context=chunks)

    if top < config.IDK_THRESHOLD:
        return GroundedAnswer(answer=IDK_ANSWER, grounded=False, sources=[], **base)

    # Small-to-big: answer from the whole top-ranked document; other retrieved
    # chunks that clear the threshold stay in the evidence (and the prompt).
    top_doc = chunks[0]
    doc_chunks = parent_chunks(top_doc.doc_id, collection)
    evidence = doc_chunks + [c.text for c in chunks
                             if c.similarity >= config.IDK_THRESHOLD and c.text not in doc_chunks]
    prompt = PROMPT_TEMPLATE.format(context="\n".join(f"- {e}" for e in evidence), query=query)
    if config.MOCK_LLM:
        text = generator(prompt, query, doc_chunks, top_doc.doc_id, top_doc.title)
    elif (text := llm_generator(prompt)) is None:  # the model itself found no answer in the context
        return GroundedAnswer(answer=IDK_ANSWER, grounded=False, sources=[], evidence=evidence, **base)
    return GroundedAnswer(answer=text, grounded=True, sources=[top_doc.doc_id], evidence=evidence, **base)
