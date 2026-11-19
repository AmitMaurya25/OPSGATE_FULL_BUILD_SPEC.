"""Central settings loaded from .env."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

def _f(name, default): return float(os.getenv(name, default))
def _i(name, default): return int(os.getenv(name, default))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "llama-3.1-8b-instant")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SIM_THRESHOLD = _f("SIM_THRESHOLD", 0.45)
SUMMARY_TRIGGER_TOKENS = _i("SUMMARY_TRIGGER_TOKENS", 1200)
SUMMARY_KEEP_LAST_N = _i("SUMMARY_KEEP_LAST_N", 2)
DEFER_MIN_TOOLS = _i("DEFER_MIN_TOOLS", 8)
PAGE_SIZE = _i("PAGE_SIZE", 50)
SUMMARY_EVERY_N_TURNS = _i("SUMMARY_EVERY_N_TURNS", 0)       # 0 = token trigger only
CONTEXT_BUDGET_TOKENS = _i("CONTEXT_BUDGET_TOKENS", 8000)    # for the UI budget bar
AUTO_PRELOAD_TOOLS = _i("AUTO_PRELOAD_TOOLS", 1)             # 1 = host picks top-2 tools up front (saves one LLM round trip)
CACHE_BLOCK_TOKENS = _i("CACHE_BLOCK_TOKENS", 64)            # KV-cache block size used by the hit-rate estimator
MAX_TOOL_OUTPUT_TOKENS = _i("MAX_TOOL_OUTPUT_TOKENS", 3000)  # hard cap on any tool result entering the prompt (engineered mode)
PERSISTENT_MCP = _i("PERSISTENT_MCP", 1)                     # 1 = keep MCP server processes alive between calls

DATA = ROOT / "data"
DB_PATH = DATA / "mock_ops.db"
RUNBOOK_DIR = DATA / "runbooks"
CHROMA_DIR = DATA / "chroma"
MEM0_DIR = DATA / "mem0"
GIT_JSON = DATA / "mock_git.json"
SERVICES = ["payment-worker", "order-api", "inventory-sync"]
