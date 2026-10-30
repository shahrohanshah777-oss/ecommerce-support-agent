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

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
KB_DIR = ROOT / "kb"
CHROMA_DIR = ROOT / "chroma_db"
TRANSCRIPTS_DIR = ROOT / "transcripts"
LOGS_DIR = ROOT / "logs"
REQUEST_LOG = LOGS_DIR / "requests.jsonl"

# --- LLM mode ----------------------------------------------------------------
# Graded transcripts run with MOCK_LLM=true (default): deterministic, no API keys.
MOCK_LLM = os.environ.get("MOCK_LLM", "true").lower() == "true"

# --- RAG ---------------------------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_FIXED = "kb_fixed"
COLLECTION_SENTENCE = "kb_sentence"
TOP_K = 3
# Placeholder: replaced in Phase 3 with an empirically calibrated value.
IDK_THRESHOLD = None

# --- Governance --------------------------------------------------------------
TOKEN_BUDGET = 2000  # max estimated tokens per request
