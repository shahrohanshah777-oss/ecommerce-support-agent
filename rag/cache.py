"""Task 16 - in-memory LRU cache for the grounded-generation step, keyed by
normalised query text. Cleared whenever the index changes (build / add-document),
so a cached "I don't know" can never hide newly added knowledge."""
from __future__ import annotations

import re
import threading
from collections import OrderedDict
from typing import Any


def normalize(query: str) -> str:
    """Case-, whitespace- and trailing-punctuation-insensitive cache key."""
    return re.sub(r"\s+", " ", query.lower()).strip().rstrip("?!. ")


class AnswerCache:
    def __init__(self, max_size: int = 512) -> None:
        self.max_size = max_size
        self._store: OrderedDict[tuple, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple) -> Any | None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self.hits += 1
                return self._store[key]
            self.misses += 1
            return None

    def put(self, key: tuple, value: Any) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> dict:
        return {"entries": len(self._store), "hits": self.hits, "misses": self.misses}


answer_cache = AnswerCache()
