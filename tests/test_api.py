"""Tests for Task 11 (FastAPI + WebSocket) and Task 12 (JSON-Lines logging)."""
import json

import config
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def log_file(tmp_path, monkeypatch):
    path = tmp_path / "requests.jsonl"
    monkeypatch.setattr(config, "REQUEST_LOG", path)
    return path


@pytest.fixture
def client():
    return TestClient(app)


def entries(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_ask_returns_pydantic_response_and_trace_header(client, log_file):
    r = client.post("/ask", json={"query": "Where is my order NYK-1009?", "session_id": "t-api"})
    body = r.json()
    assert r.status_code == 200 and body["order"]["record_id"] == "NYK-1009"
    assert r.headers["x-trace-id"] == body["trace_id"]
    (entry,) = entries(log_file)
    assert entry["trace_id"] == body["trace_id"] and entry["latency_ms"] > 0
    assert entry["event"] == "http_request" and entry["path"] == "/ask"


def test_validation_errors_are_422_and_still_logged(client, log_file):
    assert client.post("/ask", json={"query": ""}).status_code == 422
    assert client.post("/add-document", json={"doc_id": "BAD!", "title": "t", "text": "x"}).status_code == 422
    assert [e["status_code"] for e in entries(log_file)] == [422, 422]


def test_raw_pii_never_written_to_log(client, log_file):
    client.post("/ask", json={"query": "Call 9876543210, card ending 4242, about NYK-1009"})
    raw = log_file.read_text(encoding="utf-8")
    assert "9876543210" not in raw and "4242" not in raw
    assert "[PHONE]" in raw and "[CARD_LAST4]" in raw


def test_websocket_multi_turn_and_disconnect(client, log_file):
    with client.websocket_connect("/ws/chat/t-ws") as ws:
        ws.send_text(json.dumps({"query": "Where is my order NYK-1009?"}))
        assert ws.receive_json()["order"]["record_id"] == "NYK-1009"
        ws.send_text("not json")
        assert "error" in ws.receive_json()
        ws.send_text(json.dumps({"query": "Is it going to be escalated?"}))
        assert ws.receive_json()["order"]["record_id"] == "NYK-1009"  # memory across turns
    # client closed; the server logged the disconnect and still serves new requests
    assert any(e["event"] == "ws_disconnect" for e in entries(log_file))
    assert client.get("/health").status_code == 200
