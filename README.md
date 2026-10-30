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

## RAG core (Tasks 3–5)

### Task 3: two chunking strategies, two collections
| Strategy | Collection | Chunks | Rule |
|---|---|---|---|
| Fixed-size with overlap | `kb_fixed` | 42 | 40-word windows, 10-word overlap (ignores sentence boundaries) |
| Sentence-based | `kb_sentence` | 52 | one sentence per chunk |

Both strategies are embedded with **`sentence-transformers/all-MiniLM-L6-v2`** (local, normalised vectors) and `collection.upsert()`-ed into **separate** persistent ChromaDB collections that use cosine distance. Each chunk is embedded as `"<document title>: <chunk>"` (a *contextual chunk header*), while the stored text stays the raw chunk. Without the header, short policy sentences lose their topic: "Nykaa currently delivers only to addresses within India" never says "shipping". In the first calibration run without headers, the in-scope and out-of-scope groups **overlapped**, and "Do you ship to Dubai?" retrieved the wrong document. Evidence: `transcripts/task03_indexing.txt`.

The model is downloaded from Hugging Face once on first run (~90 MB) and loads from the local cache (`local_files_only=True`) after that. All RAG demos were rerun with every outbound socket blocked and made zero network attempts.

### Task 4: "I don't know" threshold, calibrated empirically
Top-1 cosine similarity measured on 12 in-scope and 10 out-of-scope queries (`transcripts/task04_calibrate.txt`):

| Collection | In-scope min | In-scope mean | Out-of-scope max | Out-of-scope mean | Gap midpoint |
|---|---|---|---|---|---|
| `kb_fixed` | 0.350 | 0.566 | 0.329 | 0.156 | 0.3396 |
| `kb_sentence` | 0.349 | 0.586 | 0.330 | 0.165 | 0.3392 |

**Chosen threshold: `IDK_THRESHOLD = 0.34`**, which sits inside the observed gap for both collections. The common tutorial presets would be wrong here: 0.5 wrongly refuses 2/12 in-scope queries, 0.6 refuses 6–7/12, and 0.7 refuses 11–12/12. **Caveat:** the gap is only about 0.02 wide. The closest cases are "Do you ship to Dubai?" (in-scope, 0.35) and "Write a Python function to reverse a linked list" (out-of-scope, 0.33, which matches *reverse pickup* on one word). In production I'd add a cross-encoder re-ranker or an LLM relevance check rather than rely on a single cosine cut-off.

**Grounded generation** (`rag/generate.py`, `transcripts/task04_grounded.txt`): retrieve top-3 chunks. If top-1 < 0.34, return the fallback. Otherwise, under `MOCK_LLM`, an **extractive** generator quotes the 1–2 retrieved sentences that best match the query (dense cosine + 0.4 × IDF-weighted keyword overlap, both computed on `"<title>: <sentence>"`). Every quoted sentence is verbatim retrieved text, and the demo asserts this. The demo answers 5 in-scope queries and refuses 1 out-of-scope query ("book me a train ticket", top-1 0.295 / 0.331) on both collections.

*Honest limitations:* the selector's keyword weight was set while looking at these 5 demo queries, so they aren't a held-out test (the 15 Task 13 queries are). Two answers are grounded but incomplete. For "What is the return window for electronics?" this is a real **retrieval** miss: the right document is retrieved top-1, but the sentence that holds the answer ("…Electronics within 7 days") isn't in the top-3 chunks. For the COD-refund question, the selector quotes the bank-details retry sentence ("…restarts the 5 to 7 business day timeline") rather than the main timeline sentence. Both show the limits of a keyword-plus-embedding extractive mock, which a real LLM generator would not have.

### Task 5: precision / recall per collection (`transcripts/task05_retrieval_eval.txt`)
Top-3 chunks → parent `doc_id` → de-duplicated → scored against hand-labelled relevant documents.

| Query | Relevant | `kb_fixed` P | `kb_fixed` R | `kb_sentence` P | `kb_sentence` R |
|---|---|---|---|---|---|
| Return window for electronics | 01 | 1/3 = 0.33 | 1/1 = 1.00 | 1/3 = 0.33 | 1/1 = 1.00 |
| COD refund to bank | 02 | 1/1 = 1.00 | 1/1 = 1.00 | 1/1 = 1.00 | 1/1 = 1.00 |
| Cancel a delayed shipment | 06, 03 | 1/1 = 1.00 | **1/2 = 0.50** | 2/2 = 1.00 | **2/2 = 1.00** |
| Reward points per order | 07 | 1/1 = 1.00 | 1/1 = 1.00 | 1/1 = 1.00 | 1/1 = 1.00 |
| Damaged hair straightener | 10 | 1/2 = 0.50 | 1/1 = 1.00 | 1/2 = 0.50 | 1/1 = 1.00 |
| **Mean** | | **0.767** | **0.900** | **0.767** | **1.000** |

**Recommendation: deploy sentence-based chunking (`kb_sentence`).** Both strategies have the same mean document-level precision (0.767), but sentence chunks reach a mean recall of 1.000 versus 0.900. On the two-document question about cancelling a delayed shipment, the overlapping fixed windows filled all three top-3 slots with copies of the cancellation policy, while sentence chunks left room for the delivery-SLA document. Sentence chunks also give cleaner answers: the extractive generator quotes complete sentences instead of window fragments like "…within 7". Since the recall gap rests on one query out of five, I'd re-check it on a larger set before production. The crew (Part 2) uses `config.RECOMMENDED_COLLECTION = "kb_sentence"`.

## CrewAI orchestration (Tasks 6, 7, 9)

### Task 6: `check_order_status` and the escalation score (`tools/order_status.py`)
`check_order_status(record_id) -> dict` returns the order's `status`, `order_value_inr`, category, age, delay flag, `escalation_score` and an `escalate` boolean. Unknown IDs return `{"found": false, ...}`.

```
escalation_score = 0.6 × delayed_shipment + 0.4 × (days_since_created / 30)
escalate  ⇔  escalation_score ≥ 0.70
```

* **Design:** a delayed shipment is the dominant signal. A non-delayed order tops out at 0.40, so it can never escalate on age alone. Among delayed orders, age breaks ties: the longer a late order has been open, the more urgent it is.
* **Threshold 0.70** means *delayed and `days_since_created` ≥ 8*, i.e. late and past the KB's 7-day upper delivery SLA. In the generated data, the 14 delayed orders' ages form two clusters (**3–6 days** and **12–24 days**), and 0.70 falls exactly in the gap (scores 0.680 vs 0.760). The threshold is the **86.7th percentile** of `escalation_score` in my data and flags **8/60 orders (13.3%)**, a manageable Level-2 queue (`transcripts/task06_order_status.txt`).
* *Possible extension:* the score ignores status, so a delayed order that has already been Refunded still scores high. A production version would down-weight terminal statuses.

### Task 7: the crew (`crew/crew.py`, `transcripts/task07_crew.txt`)
Three agents run sequentially via `crew.kickoff()`. Each has its own `MockLLM` and only the tools it needs:

| Agent | Tool | Output |
|---|---|---|
| Retrieval Agent | `rag_lookup` (Task 4 grounded generation on `kb_sentence`) | JSON: answer, grounded, sources, context |
| Lookup Agent | `check_order_status` | JSON order record, or `found: false` |
| Response Composer | none | `SupportResponse` draft (Task 9) |

The Retrieval and Lookup tasks use `context=[]`, so an order ID can only come from the customer's own query, never from another agent's output. The demo shows the **RAG tool alone** on a policy question, the **lookup tool** on order questions, **both** on "Can I cancel my order NYK-1004?", a not-found order, and an out-of-scope fallback. Each run prints the agents' ReAct decisions and the tools' own invocation logs.

Two refinements that came out of the demo, and are documented in the transcript:
* **Order IDs are stripped from the retrieval query** (`rag.generate.policy_query`). They never occur in the KB and only added noise. Before this fix, "Can I cancel my order NYK-1004?" quoted the out-of-stock sentence.
* **Policy is attached to an order answer only when retrieval is strong** (`ORDER_POLICY_MIN_SIM = 0.49`), calibrated the same way as the IDK threshold. Pure status questions ("Where is my order NYK-1009?") top out at 0.480, while order + policy questions start at 0.508. Without this, status answers got an irrelevant policy sentence appended.

### Task 9: structured output (`crew/schemas.py`, `transcripts/task09_schema.txt`)
`SupportResponse` (`answer`, `intent ∈ {policy, order_status, out_of_scope}`, `grounded`, `sources`, `order: OrderInfo | None`, `escalate`, `retrieval_top_similarity`; `extra="forbid"`, bounded numbers, an ID pattern) is enforced at three points:
1. The Composer task declares `output_pydantic=SupportResponse`, and CrewAI passes it to the LLM as `response_model` (the response-format contract).
2. `MockLLM` validates its own final answer against `response_model` before returning it, as a real structured-output API would.
3. `run_crew()` re-validates the result in code with `SupportResponse.model_validate(...)`.

The demo validates 5 different crew responses and shows the schema **rejecting** an unknown intent, `escalation_score = 1.5`, an empty answer and a hallucinated extra field.

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
| `rag/` | Tasks 3–5, 16: KB loader, chunking, indexing, grounded generation, keyword scoring, retrieval eval, cache |
| `llm/` | Mock LLM (`crewai.llms.base_llm.BaseLLM` subclass) |
| `tools/` | RAG tool and `check_order_status` |
| `crew/` | Tasks 7, 9: CrewAI crew and Pydantic response schema |
| `review/` | Task 14: Autogen review stage |
| `api/` | Tasks 11–12: FastAPI app and JSON-Lines logging |
| `eval/` | Task 13: LLM-as-judge evaluation |
| `demos/` | One runnable script per task |
| `transcripts/` | Saved output of every demo (graded evidence) |
| `tests/` | Unit tests (`python -m pytest`) |
