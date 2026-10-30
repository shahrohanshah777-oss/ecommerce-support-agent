"""Task 11 - FastAPI deployment of the support crew (+ Task 12 structured logging).

    GET  /health                    liveness + index sizes
    POST /ask                       one question -> SupportResponse-backed answer
    POST /add-document              add/replace a KB document in both collections
    WS   /ws/chat/{session_id}      real-time multi-turn chat (shares session memory)

Run:  uvicorn api.main:app --port 8000
"""
from __future__ import annotations

import json

import config
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from crew.schemas import OrderInfo
from guardrails import mask_pii
from observability import Timer, log_event, new_trace_id
from pipeline import PipelineResult, handle
from rag.chunking import STRATEGIES
from rag.index import get_collection, upsert_document
from rag.kb import split_sentences

app = FastAPI(title="Nykaa Support Agent", version="1.0",
              description="CrewAI support crew behind FastAPI. Runs fully offline under MOCK_LLM.")


# --- Pydantic request / response models ------------------------------------------
class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, examples=["Where is my order NYK-1009?"])
    session_id: str = Field(default="default", pattern=r"^[\w-]{1,64}$")


class AskResponse(BaseModel):
    trace_id: str
    session_id: str
    status: str
    answer: str
    intent: str | None = None
    grounded: bool = False
    sources: list[str] = []
    order: OrderInfo | None = None
    escalate: bool = False
    pii_masked: dict[str, int] = {}
    injection_matched: list[str] = []
    latency_ms: float


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(pattern=r"^[a-z0-9_]{3,64}$", examples=["14_gift_wrapping"])
    title: str = Field(min_length=3, max_length=120)
    text: str = Field(min_length=20, max_length=3000)


class AddDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    sentences: int
    chunks_indexed: dict[str, int]
    pii_masked: dict[str, int] = {}


def to_response(r: PipelineResult, trace_id: str, latency_ms: float) -> AskResponse:
    resp = r.response
    answered = r.status == "answered" and resp is not None
    return AskResponse(
        trace_id=trace_id, session_id=r.session_id, status=r.status, answer=r.answer,
        intent=resp.intent if resp else None,
        grounded=answered and resp.grounded,
        sources=resp.sources if answered else [],
        order=resp.order if answered else None,
        escalate=answered and resp.escalate,
        pii_masked=r.pii_masked, injection_matched=r.injection_matched,
        latency_ms=round(latency_ms, 1),
    )


def pipeline_log_fields(r: PipelineResult) -> dict:
    """What is logged about a pipeline run: masked text and outcome only."""
    return {
        "session_id": r.session_id,
        "query": r.masked_query,  # masked before the model saw it; scrubbed again by log_event
        "pii_masked": r.pii_masked,
        "injection_matched": r.injection_matched,
        "pipeline_status": r.status,
        "intent": r.response.intent if r.response else None,
        "tools": {k: len(v) for k, v in r.tool_calls.items()},
        "grounded_check": r.groundedness.passed if r.groundedness else None,
    }


# --- HTTP middleware: one JSON-Lines entry per request ----------------------------
@app.middleware("http")
async def structured_logging(request: Request, call_next):
    trace_id = new_trace_id()
    request.state.trace_id = trace_id
    request.state.log_fields = {}
    timer = Timer().__enter__()
    try:
        response = await call_next(request)
    except Exception as exc:  # never log the raw body; record type only
        log_event("http_request", trace_id, timer.elapsed_ms, method=request.method,
                  path=request.url.path, status_code=500, error=type(exc).__name__,
                  **request.state.log_fields)
        return JSONResponse({"detail": "internal error", "trace_id": trace_id}, status_code=500)
    response.headers["X-Trace-ID"] = trace_id
    log_event("http_request", trace_id, timer.elapsed_ms, method=request.method,
              path=request.url.path, status_code=response.status_code, **request.state.log_fields)
    return response


# --- Endpoints -------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mock_llm": config.MOCK_LLM,
            "collections": {name: get_collection(name).count() for name in STRATEGIES}}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request) -> AskResponse:
    with Timer() as t:
        result = handle(req.query, req.session_id)
    request.state.log_fields = pipeline_log_fields(result)
    return to_response(result, request.state.trace_id, t.ms)


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(req: AddDocumentRequest, request: Request) -> AddDocumentResponse:
    title, text = mask_pii(req.title), mask_pii(req.text)  # never index PII either
    found = {k: title.findings.get(k, 0) + text.findings.get(k, 0)
             for k in {**title.findings, **text.findings}}
    chunks = upsert_document({"doc_id": req.doc_id, "title": title.text, "text": text.text})
    request.state.log_fields = {"doc_id": req.doc_id, "chars": len(req.text), "chunks": chunks,
                                "pii_masked": found}
    return AddDocumentResponse(trace_id=request.state.trace_id, doc_id=req.doc_id,
                               sentences=len(split_sentences(text.text)), chunks_indexed=chunks,
                               pii_masked=found)


class WsMessage(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


@app.websocket("/ws/chat/{session_id}")
async def ws_chat(websocket: WebSocket, session_id: str) -> None:
    """Multi-turn chat. Each message is one logged request. A client that vanishes
    mid-conversation ends only its own loop; the server keeps serving others."""
    await websocket.accept()
    conn_id, turns = new_trace_id(), 0
    log_event("ws_connect", conn_id, session_id=session_id)
    try:
        while True:
            raw = await websocket.receive_text()
            trace_id = new_trace_id()
            with Timer() as t:
                try:
                    msg = WsMessage.model_validate(json.loads(raw))
                except (json.JSONDecodeError, ValidationError) as exc:
                    log_event("ws_message", trace_id, t.elapsed_ms, session_id=session_id,
                              status_code=422, error=type(exc).__name__)
                    await websocket.send_json({"trace_id": trace_id, "error": 'send {"query": "..."}'})
                    continue
                # Crew runs are blocking; keep the event loop free for other clients.
                result = await run_in_threadpool(handle, msg.query, session_id)
            turns += 1
            log_event("ws_message", trace_id, t.ms, connection_id=conn_id, turn=turns,
                      status_code=200, **pipeline_log_fields(result))
            await websocket.send_json(to_response(result, trace_id, t.ms).model_dump())
    except WebSocketDisconnect as exc:
        log_event("ws_disconnect", conn_id, session_id=session_id, turns=turns,
                  code=exc.code, reason="client disconnected")
    except RuntimeError as exc:  # send after the client already dropped
        log_event("ws_disconnect", conn_id, session_id=session_id, turns=turns,
                  reason=f"client gone during send ({type(exc).__name__})")
