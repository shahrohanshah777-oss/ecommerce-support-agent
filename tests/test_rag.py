"""Tests for Tasks 3-5: chunking, grounded generation / fallback, P/R arithmetic."""
import config

import rag.eval_retrieval as ev
from rag.chunking import chunk_corpus, fixed_chunks, sentence_chunks
from rag.generate import answer
from rag.kb import answer_sentences, load_documents


def test_fixed_chunks_window_and_overlap():
    words = [f"w{i}" for i in range(100)]
    chunks = fixed_chunks(" ".join(words), size=40, overlap=10)
    assert [len(c.split()) for c in chunks] == [40, 40, 40]  # starts 0, 30, 60; 60+40 reaches the end
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]  # 10-word overlap
    assert chunks[-1].split()[-1] == "w99"  # nothing dropped


def test_sentence_chunks_are_sentences():
    text = "First rule applies. Second rule applies too! Is there a third?"
    assert sentence_chunks(text) == ["First rule applies.", "Second rule applies too!", "Is there a third?"]


def test_every_chunk_maps_to_a_parent_doc():
    ids = {d["doc_id"] for d in load_documents()}
    for name in (config.COLLECTION_FIXED, config.COLLECTION_SENTENCE):
        assert {c["metadata"]["doc_id"] for c in chunk_corpus(name)} == ids


def test_out_of_scope_triggers_fallback():
    r = answer("What is the capital of France?")
    assert not r.grounded and r.answer.startswith("I don't know") and r.sources == []
    assert r.top_similarity < config.IDK_THRESHOLD


def test_in_scope_answer_quotes_only_retrieved_text():
    r = answer("How many reward points can I use on an order?")
    assert r.grounded and r.sources == ["07_loyalty_points"]
    evidence = " ".join(r.evidence)
    assert all(s in evidence for s in answer_sentences(r.answer))


def test_answer_sentences_splits_lowercase_window_fragments():
    a = "An order can be cancelled while Placed. ...be cancelled, unless delayed, so refuse it."
    assert answer_sentences(a) == ["An order can be cancelled while Placed.", "be cancelled, unless delayed, so refuse it."]


def test_small_to_big_expands_to_the_whole_top_document():
    r = answer("What is the return window for electronics?", use_cache=False)
    assert r.context[0].doc_id == "01_return_window"
    assert "Electronics within 7 days" not in " ".join(c.text for c in r.context)  # not retrieved...
    assert "Electronics within 7 days" in r.answer  # ...but answered via parent-document expansion


def test_precision_recall_dedups_chunks_to_docs(monkeypatch):
    fake_hits = [{"doc_id": "A"}, {"doc_id": "A"}, {"doc_id": "B"}]
    monkeypatch.setattr(ev, "search", lambda q, c, k: fake_hits)
    r = ev.score_query("q", {"A", "C"}, "any")
    assert r["retrieved"] == ["A", "B"]
    assert (r["precision"], r["recall"]) == (0.5, 0.5)
