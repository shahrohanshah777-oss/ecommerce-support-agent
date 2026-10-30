from rag.lexical import _stem, keyword_overlap, terms


def test_stemmer_undoubles_final_consonant():
    assert _stem("cancelled") == _stem("cancel") == "cancel"
    assert _stem("shipped") == _stem("ship") == "ship"
    assert _stem("passed") == "pass"  # keep "ss"
    assert _stem("called") == "call"  # short "ll" stem kept


def test_stopwords_removed_and_overlap_bounded():
    assert terms("What is the return window?") == {"return", "window"}
    assert keyword_overlap("cancel my order", "An order can be cancelled free of charge") == 1.0
    assert keyword_overlap("zzz", "anything") == 0.0
