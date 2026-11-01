"""Tests for the dataset and knowledge-base thresholds."""
import re

import config
import dataset
from rag.kb import REQUIRED_TOPICS, load_documents, split_sentences


def test_dataset_meets_all_thresholds():
    assert dataset.validate(dataset.ORDERS) == []
    assert len(dataset.ORDERS) >= 40


def test_dataset_is_deterministic():
    assert dataset.generate_orders() == dataset.ORDERS


def test_record_ids_match_shared_pattern():
    assert all(re.fullmatch(config.ORDER_ID_PATTERN, o["record_id"]) for o in dataset.ORDERS)


def test_validate_catches_bad_delay_band():
    no_delays = [{**o, "delayed_shipment": False} for o in dataset.ORDERS]
    assert any("delayed_shipment" in p for p in dataset.validate(no_delays))


def test_kb_has_12_docs_covering_every_topic():
    ids = {d["doc_id"] for d in load_documents()}
    assert len(ids) >= 12
    assert set(REQUIRED_TOPICS.values()) <= ids


def test_every_doc_has_2_to_5_sentences():
    for d in load_documents():
        assert 2 <= len(split_sentences(d["text"])) <= 5, d["doc_id"]
