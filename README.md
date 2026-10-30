# Nykaa Domain Support Agent (CrewAI)

**Track completed: Nykaa (E-commerce & Retail).**

> Work in progress. Sections below are filled in as each phase is completed.

## Runs fully offline under `MOCK_LLM`

- Every graded transcript in `transcripts/` was generated with `MOCK_LLM=true` (the default), with **zero API keys and zero network access**.
- **Telemetry is disabled:** `config.py` sets `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true` (and ChromaDB's `ANONYMIZED_TELEMETRY=False`) at import time, before `crewai` is loaded. Every entry point imports `config` first.

## Setup

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

`requirements.txt` pins `protobuf==5.29.6`. Without it, pip resolves crewai's OpenTelemetry dependencies to protobuf 7.x and silently falls back to the legacy `autogen-agentchat 0.2.x`, which has no `RoundRobinGroupChat`.

Run demos from the repo root as modules, e.g. `python -m demos.task01_dataset`.

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
