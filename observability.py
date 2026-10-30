"""Task 12 - ELK-style structured request logging (one JSON object per line).

Every entry carries a trace_id and latency_ms. All string values are passed
through the same mask_pii() used on model input before they are written, so a
fixed-format PII field (phone number, card digits) can never reach disk in the
clear, even if a caller forgot to mask.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import config
from guardrails import mask_pii

_LOCK = threading.Lock()


def new_trace_id() -> str:
    return uuid.uuid4().hex


def _scrub(value: Any) -> Any:
    if isinstance(value, str):
        return mask_pii(value).text
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    return value


def log_event(event: str, trace_id: str, latency_ms: float | None = None, **fields: Any) -> dict:
    entry = {
        "@timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "service": "nykaa-support-agent",
        "event": event,
        "trace_id": trace_id,
        "latency_ms": None if latency_ms is None else round(latency_ms, 1),
        **_scrub(fields),
    }
    line = json.dumps(entry, ensure_ascii=False, default=str)
    config.REQUEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, open(config.REQUEST_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return entry


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.ms = (time.perf_counter() - self.start) * 1000

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000
