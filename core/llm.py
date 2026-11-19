"""LLM + embedding factories. Falls back to offline fakes when no GROQ_API_KEY (used by tests)."""
import hashlib, re
import numpy as np
from langchain_core.messages import AIMessage
from core import config

IGNORANCE = "I cannot answer this based on the provided runbooks."

def have_llm() -> bool:
    return bool(config.GROQ_API_KEY)

def get_llm(model: str | None = None, temperature: float = 0.0):
    if have_llm():
        from langchain_groq import ChatGroq
        return ChatGroq(model=model or config.GROQ_MODEL, temperature=temperature, api_key=config.GROQ_API_KEY)
    return FakeLLM()

class FakeLLM:
    """Deterministic offline stand-in so the pipeline can be tested without an API key."""
    def __init__(self, tools=None): self._tools = tools or []
    def bind_tools(self, tools): return FakeLLM(tools)
    def invoke(self, messages, *a, **k):
        msgs = messages if isinstance(messages, list) else [messages]
        last_user = next((m.content for m in reversed(msgs) if getattr(m, "type", "") == "human"), "")
        last_tool = [m for m in msgs if getattr(m, "type", "") == "tool"]
        names = [getattr(t, "name", None) or t.get("name") if not isinstance(t, dict) else t["name"] for t in self._tools]
        if self._tools:
            if not last_tool and "search_tools" in names and len(names) == 1:
                return AIMessage(content="", tool_calls=[{"name": "search_tools", "args": {"query": last_user}, "id": "c1"}])
            if not last_tool or last_tool[-1].name == "search_tools":
                pick = [n for n in names if n != "search_tools"]
                if pick:
                    return AIMessage(content="", tool_calls=[{"name": pick[0], "args": {}, "id": "c2"}])
        if last_tool:
            return AIMessage(content="Result: " + last_tool[-1].content[:300])
        text = " ".join(getattr(m, "content", "") for m in msgs)
        if "Summarize" in text or "summary" in text.lower():
            return AIMessage(content="Decisions: none. Facts: see history. Preferences: none. Pending: none. Goal: ongoing.")
        return AIMessage(content=f"(offline) {last_user[:200]}")

class HashEmbedder:
    """Offline bag-of-words hashing embedder. Weak quality; for tests only."""
    dim = 256
    def embed(self, texts):
        out = []
        for t in texts:
            v = np.zeros(self.dim)
            for w in re.findall(r"[a-z0-9_]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
            n = np.linalg.norm(v); out.append((v / n if n else v).tolist())
        return out

class LocalEmbedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(config.EMBEDDING_MODEL)
    def embed(self, texts):
        return self.m.encode(texts, normalize_embeddings=True).tolist()

_emb = None
def get_embedder():
    global _emb
    if _emb is None:
        _emb = HashEmbedder() if config.EMBEDDING_PROVIDER == "hash" else LocalEmbedder()
    return _emb

def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0
