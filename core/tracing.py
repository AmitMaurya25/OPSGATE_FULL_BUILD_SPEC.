"""Optional LangSmith tracing. OFF unless LANGSMITH_API_KEY is in .env, so with no key the app behaves exactly as before.

Privacy: by default (LANGSMITH_REDACT=1) prompts, tool inputs and outputs are NOT sent to LangSmith, only the run structure
(which node/tool ran, in what order, how long, errors). Set LANGSMITH_REDACT=0 only on non-sensitive demo data.
Tracing never touches the prompt, so the KV-cache-friendly prompt layout is unchanged.
"""
import hashlib, os

def setup_tracing() -> dict:
    if not os.getenv("LANGSMITH_API_KEY"):
        return {"enabled": False, "redacted": True, "project": ""}
    redact = os.getenv("LANGSMITH_REDACT", "1") != "0"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "opsgate-context-copilot")
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    os.environ["LANGSMITH_HIDE_INPUTS"] = "true" if redact else "false"
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true" if redact else "false"
    return {"enabled": True, "redacted": redact, "project": os.environ["LANGSMITH_PROJECT"]}

def trace_config(mode: str, user_id: str, thread_id: str) -> dict:
    """Run name, tags and metadata for graph.invoke. User ids are hashed so no identity leaves the machine."""
    return {"run_name": f"opsgate.{mode}", "tags": [mode],
            "metadata": {"mode": mode, "user": hashlib.md5(user_id.encode()).hexdigest()[:8], "thread": thread_id}}
