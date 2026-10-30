"""IDF-weighted keyword overlap, used by the mock generator to choose which
retrieved sentences to quote. Rare, specific terms ("electronics", "COD")
count for more than terms that appear all over the KB ("order", "return")."""
from __future__ import annotations

import math
import re
from functools import lru_cache

from rag.kb import load_documents, split_sentences

STOPWORDS = set(
    """a an the is are was were be been do does did can could i my me we you your it its
    of to for in on at by with and or what how when where who why which will would should
    much many long take get this that there any if after before still""".split()
)


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def terms(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


@lru_cache(maxsize=1)
def _document_frequency() -> tuple[dict[str, int], int]:
    sentences = [s for d in load_documents() for s in split_sentences(d["text"])]
    df: dict[str, int] = {}
    for s in sentences:
        for t in terms(s):
            df[t] = df.get(t, 0) + 1
    return df, len(sentences)


def idf(term: str) -> float:
    df, n = _document_frequency()
    return math.log((n + 1) / (df.get(term, 0) + 1)) + 1


def keyword_overlap(query: str, text: str) -> float:
    """Share of the query's IDF mass whose terms appear in `text` (0..1)."""
    q = terms(query)
    total = sum(idf(t) for t in q)
    return sum(idf(t) for t in q & terms(text)) / total if total else 0.0
