# Nykaa Domain Support Agent (CrewAI)

**Track completed: Nykaa (E-commerce & Retail).**

> Work in progress. Sections below are filled in as each phase is completed.

## Runs fully offline under `MOCK_LLM`

- Every graded transcript in `transcripts/` was generated with `MOCK_LLM=true` (the default), with **zero API keys and zero network access**.
- **Telemetry is disabled:** `config.py` sets `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true` (and ChromaDB's `ANONYMIZED_TELEMETRY=False`) at import time, before `crewai` is loaded. Every entry point imports `config` first.
- **CrewAI execution tracing is disabled too:** `CREWAI_TRACING_ENABLED=false`, `CREWAI_TESTING=true`, and every crew is built with `tracing=False`. In crewai 1.15, `CREWAI_TRACING_ENABLED=false` alone does *not* stop the first-run trace collection or the 20-second "Share this execution trace with CrewAI?" prompt; `CREWAI_TESTING=true` does, and crewai reads that variable only in its tracing code.

## Setup

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

`requirements.txt` pins `protobuf==5.29.6`. Without it, pip resolves crewai's OpenTelemetry dependencies to protobuf 7.x and silently falls back to the legacy `autogen-agentchat 0.2.x`, which has no `RoundRobinGroupChat`.

Run demos from the repo root as modules, e.g. `python -m demos.task01_dataset`.

## The mock LLM (`llm/mock_llm.py`)

`MockLLM` extends `crewai.llms.base_llm.BaseLLM` and drives CrewAI's text ReAct loop deterministically (`supports_function_calling()` returns `False`). On each call it either emits one `Action` / `Action Input` tool step or a `Final Answer`. Per-agent `responder` hooks write the final-answer text.

How it avoids the two silent pitfalls in the brief:

1. **Template `Observation:` text.** CrewAI's system prompt contains `Observation: the result of the action` on every call. After a tool runs, CrewAI appends `\nObservation: <result>` to the *assistant* message that holds the mock's own `Action:` line. `latest_observation()` reads only those assistant messages and never searches system messages. So the first call (before any tool has run) correctly sees no observation.
2. **Dispatch by schema, not name.** `choose_tool()` reads each tool's declared `args_schema` and fills its *required argument names*: `record_id` comes from the order-ID pattern `NYK-\d{4,6}`, and `query` comes from the customer's question. A tool is only called if all its required arguments can be filled. Tool names are never inspected, so a tool named `rag_lookup` receives `query`.

Evidence: `tests/test_mock_llm.py` (unit tests for both pitfalls, `python -m pytest`) and `transcripts/phase1_mock_llm_spike.txt` (real `crew.kickoff()` runs). The spike was also rerun with all outbound sockets blocked, and it made zero network attempts.

## Repository layout

| Path | Purpose |
|---|---|
| `config.py` | Flags, paths, thresholds, telemetry switches |
| `dataset.py` | Task 1: seeded order generator |
| `kb/` | Task 2: knowledge-base documents |
| `rag/` | Tasks 3–5, 16: chunking, indexing, grounded generation, retrieval eval, cache |
| `llm/` | Mock LLM (`crewai.llms.base_llm.BaseLLM` subclass) |
| `tools/` | RAG tool and `check_order_status` |
| `crew/` | Tasks 7, 9: CrewAI crew and Pydantic response schema |
| `review/` | Task 14: Autogen review stage |
| `api/` | Tasks 11–12: FastAPI app and JSON-Lines logging |
| `eval/` | Task 13: LLM-as-judge evaluation |
| `demos/` | One runnable script per task |
| `transcripts/` | Saved output of every demo (graded evidence) |
| `tests/` | Unit tests (`python -m pytest`) |
