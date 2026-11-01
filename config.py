"""Central configuration. Import this module FIRST in every entry point.

Importing it disables CrewAI / OpenTelemetry telemetry before crewai is loaded,
so `crew.kickoff()` never attempts an outbound network call.
"""
import os
from pathlib import Path

# --- Zero-network guarantees (must run before `import crewai`) ---------------
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # ChromaDB telemetry
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")  # Hugging Face hub
# CrewAI execution tracing. CREWAI_TRACING_ENABLED=false alone does NOT stop the
# first-run trace collection + 20s "Share this execution trace?" prompt;
# CREWAI_TESTING=true does (it is only read by crewai's tracing code).
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CREWAI_TESTING", "true")

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
KB_DIR = ROOT / "kb"
# Overridable so demos that add documents can use a scratch index.
CHROMA_DIR = Path(os.environ.get("NYKAA_CHROMA_DIR", ROOT / "chroma_db"))
TRANSCRIPTS_DIR = ROOT / "transcripts"
LOGS_DIR = ROOT / "logs"
REQUEST_LOG = Path(os.environ.get("NYKAA_REQUEST_LOG", LOGS_DIR / "requests.jsonl"))

# --- LLM mode ----------------------------------------------------------------
# Graded transcripts run with MOCK_LLM=true (default): deterministic, no API keys.
MOCK_LLM = os.environ.get("MOCK_LLM", "true").lower() == "true"
# Optional real-LLM mode (MOCK_LLM=false): all four LLM call sites (grounded
# generation, CrewAI crew, Autogen review, judge) use one OpenAI-compatible
# chat-completions endpoint. Default: Groq's free tier, which needs GROQ_API_KEY.
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY_ENV = os.environ.get("LLM_API_KEY_ENV", "GROQ_API_KEY")  # name of the variable, never the key
LLM_TEMPERATURE = 0.0
# Groq's free tier allows 8,000 tokens/minute and one request uses ~5-7k, so 429s
# are expected; the OpenAI SDK waits for the server's retry-after between attempts.
LLM_MAX_RETRIES = 8

# --- Orders ------------------------------------------------------------------
# Shared by dataset.py (ID generation) and the mock LLM (ID extraction).
ORDER_ID_PATTERN = r"\bNYK-\d{4,6}\b"

# --- RAG ---------------------------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_FIXED = "kb_fixed"
COLLECTION_SENTENCE = "kb_sentence"
FIXED_CHUNK_WORDS = 40
FIXED_CHUNK_OVERLAP = 10
# Recommended by the retrieval evaluation (equal mean precision 0.767; recall 1.000 vs 0.900).
RECOMMENDED_COLLECTION = COLLECTION_SENTENCE
TOP_K = 3
# "I don't know" threshold on top-1 cosine similarity, calibrated empirically in
# demos/idk_calibration.py (12 in-scope vs 10 out-of-scope queries):
#   kb_fixed:    in-scope min 0.350, out-of-scope max 0.329 -> midpoint 0.3395
#   kb_sentence: in-scope min 0.349, out-of-scope max 0.330 -> midpoint 0.3395
IDK_THRESHOLD = 0.34
# For order-status queries the Composer appends retrieved policy only above this
# similarity. Calibrated in demos/crew_kickoff.py (order IDs stripped from the
# retrieval query): pure status questions ("Where is my order NYK-1009?") top
# out at 0.480; order+policy questions ("Can I cancel my order NYK-1004?")
# start at 0.508 -> midpoint 0.494.
ORDER_POLICY_MIN_SIM = 0.49
# Real-LLM mode only: a paraphrased answer sentence counts as grounded when a local
# NLI model judges it entailed by the evidence (and its numbers occur in the
# evidence). Calibrated in demos/real_llm_grounding_calibrate.py on 16 labelled
# paraphrases vs 15 invented claims + 3 contradictions: paraphrases >= 0.992,
# the rest <= 0.242 -> midpoint 0.617. (Embedding similarity was tried first and
# rejected: the two groups overlapped and it accepted contradictions.)
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
GROUNDING_MIN_ENTAILMENT = 0.62

# --- Governance --------------------------------------------------------------
# Per-request token budget (runtime governance layer), calibrated from metered
# usage of the full pipeline (crew + review, small-to-big generation) on the 15
# eval queries and padded variants: short queries used 3,245-5,253 tokens; each
# extra query token added ~4.5 tokens (the query is repeated across model calls).
#   estimate = BUDGET_OVERHEAD_TOKENS + BUDGET_QUERY_REPEATS * query_tokens
# Both constants are rounded up, so every measured request used fewer tokens than
# estimated. 8,500 admits queries up to ~500 tokens (~2,000 chars, the API limit).
# Re-measure whenever prompts or tool outputs change (small-to-big raised usage).
TOKEN_BUDGET = 8_500
BUDGET_OVERHEAD_TOKENS = 5_500
BUDGET_QUERY_REPEATS = 6
COST_PER_1K_TOKENS_INR = 0.25  # illustrative rate for cost reporting; MOCK_LLM itself costs nothing
