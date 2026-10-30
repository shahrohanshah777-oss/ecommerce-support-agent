"""Task 16 - in-memory response cache for the grounded-generation step, keyed by
normalised query text. Evidence: generator call counter, retrieval counter, timing.

Run:  python -m demos.task16_cache
"""
import time

import config  # noqa: F401

import rag.generate as gen
from demos._transcript import transcript
from pipeline import handle
from rag.cache import answer_cache, normalize
from rag.index import build_index

QUERY = "What is the return window for footwear?"


def counters() -> tuple[int, int]:
    return gen.generator.calls, gen.retrieval_calls


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000


def main() -> None:
    gen.answer("warm up the embedding model", use_cache=False)  # keep model load out of the timings
    answer_cache.clear()
    print(f"Cache key = normalize(query) -> {normalize(QUERY)!r}\n")

    print("1) Grounded-generation step called directly")
    g0, r0 = counters()
    first, t1 = timed(lambda: gen.answer(QUERY))
    g1, r1 = counters()
    print(f"  call 1 (miss): generator calls {g0} -> {g1}, retrievals {r0} -> {r1}, {t1:7.1f} ms")
    second, t2 = timed(lambda: gen.answer(QUERY))
    g2, r2 = counters()
    print(f"  call 2 (hit) : generator calls {g1} -> {g2}, retrievals {r1} -> {r2}, {t2:7.1f} ms")
    variant = "   what is the RETURN window for footwear  "
    third, t3 = timed(lambda: gen.answer(variant))
    g3, r3 = counters()
    print(f"  call 3 (hit) : {variant!r} -> generator calls {g2} -> {g3}, {t3:7.1f} ms (normalised key)")
    print(f"  same answer every time: {first.answer == second.answer == third.answer}")
    print(f"  cache stats: {answer_cache.stats()}")
    assert (g1, r1) == (g0 + 1, r0 + 1) and (g2, r2) == (g1, r1) and (g3, r3) == (g2, r2)
    assert first.answer == second.answer == third.answer

    print("\n2) Through the full pipeline (crew + review): the RAG tool still runs, but its")
    print("   generation step is served from the cache on the repeat")
    answer_cache.clear()
    q = "Do my Nykaa reward points ever expire?"
    g0, _ = counters()
    a, ta = timed(lambda: handle(q, "cache-user-1"))
    g1, _ = counters()
    b, tb = timed(lambda: handle(q, "cache-user-2"))
    g2, _ = counters()
    print(f"  request 1: generator calls {g0} -> {g1}  ({ta:7.1f} ms)  rag_lookup calls={len(a.tool_calls['rag_lookup'])}")
    print(f"  request 2: generator calls {g1} -> {g2}  ({tb:7.1f} ms)  rag_lookup calls={len(b.tool_calls['rag_lookup'])}")
    print(f"  identical answers: {a.answer == b.answer}")
    assert g1 == g0 + 1 and g2 == g1 and a.answer == b.answer

    print("\n3) Invalidation: re-indexing (also done by POST /add-document) clears the cache")
    print(f"  before build_index(): {answer_cache.stats()}")
    build_index()
    print(f"  after  build_index(): {answer_cache.stats()}")
    g0, _ = counters()
    gen.answer(QUERY)
    print(f"  next identical query is a miss again: generator calls {g0} -> {counters()[0]}")
    assert counters()[0] == g0 + 1


if __name__ == "__main__":
    with transcript("task16_cache"):
        main()
