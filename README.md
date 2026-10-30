# Nykaa Domain Support Agent (CrewAI)

**Track completed: Nykaa (E-commerce & Retail).**

> Work in progress. Sections below are filled in as each phase is completed.

## Dataset design (Task 1): reproduce with `python dataset.py`

| Choice | Value |
|---|---|
| Seed | `SEED = 42` (local `random.Random(42)`). This is the first seed tried, and it passed every threshold, so no reseeding was needed and no record was hand-edited |
| Size | `N_ORDERS = 60`, IDs `NYK-1001` … `NYK-1060` |
| Category weights | Beauty 0.35, Apparel 0.25, Footwear 0.15, Home 0.13, Electronics 0.12 (Beauty-heavy, since beauty is Nykaa's core business) |
| Status weights | Placed 0.15, Shipped 0.20, Delivered 0.40, Returned 0.12, Refunded 0.13 |
| Amount range | **Rs 199 – Rs 14,999** overall, drawn per category (Beauty 199–4,999 · Apparel 399–6,999 · Footwear 499–8,999 · Home 299–9,999 · Electronics 799–14,999) and rounded to Rs x,xx9 price points |
| `days_since_created` | Integer 0–30, banded by status (Placed 0–2 · Shipped 1–8 · Delivered 3–30 · Returned 8–30 · Refunded 10–30) |
| `delayed_shipment` | Bernoulli(0.25), only for orders ≥ 3 days old (an order can't miss its 2–7-day delivery SLA before that) |

**Price-range reasoning:** Rs 199 – Rs 14,999 covers everything from a single lipstick or sheet mask at the low end to premium beauty electronics (hair dryers, stylers) at the top, which matches Nykaa's actual price spread.

**Generated result** (`transcripts/task01_dataset.txt`):

| Category | Count | | Status | Count |
|---|---|---|---|---|
| Beauty | 24 | | Placed | 10 |
| Apparel | 11 | | Shipped | 13 |
| Footwear | 11 | | Delivered | 24 |
| Home | 5 | | Returned | 5 |
| Electronics | 9 | | Refunded | 8 |

`delayed_shipment=True`: **14 / 60 = 23.3%** (required 10–30%). Every category has ≥ 3 records and every status has ≥ 1. `dataset.validate()` asserts all of these thresholds.

## Knowledge base (Task 2)

`kb/` contains 13 documents of 3–5 sentences each: the 12 required topics plus an order-status glossary. The topic-to-document mapping is in `rag/kb.py::REQUIRED_TOPICS`, and the inventory is in `transcripts/task02_kb.txt`. All policies were written for this project and are **fictional**. They are internally consistent (e.g. the delivery SLA defines `delayed_shipment`, and the escalation matrix uses a Rs 5,000 threshold), but they are not Nykaa's real policies.

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
