"""Memory backends. Mem0Store = real Mem0 (needs GROQ_API_KEY); LocalStore = offline JSON store with rule-based tagging (tests/demo)."""
import json, re, time, uuid
from core import config
from core.llm import get_embedder, cosine, have_llm

CUSTOM_INSTRUCTIONS = """Extract durable facts about the USER only. Ignore chit-chat and general-knowledge questions.
Rewrite first person as third person ("The user ..."). Start every memory with exactly one or two tags:
[Semantic] stable facts (identity, location, diet, allergies, role, goals)
[Procedural] interaction rules/preferences (language, tone, step-by-step, format)
[Episodic] events and decisions (what happened, what was tried, what was built)
A fact may carry two tags. Example: "[Semantic][Episodic] The user moved from Delhi to Bore." """

TAG_RE = re.compile(r"\[(Semantic|Episodic|Procedural)\]", re.I)
def tags_of(text): return [t.capitalize() for t in TAG_RE.findall(text)]
def strip_tags(text): return TAG_RE.sub("", text).strip()

class LocalStore:
    """Offline store. Rule-based tagger stands in for Mem0's LLM extraction."""
    def __init__(self, path=None):
        self.path = path or (config.MEM0_DIR / "local_memories.json"); config.MEM0_DIR.mkdir(parents=True, exist_ok=True)
        self.items = json.loads(self.path.read_text()) if self.path.exists() else []
    def _save(self): self.path.write_text(json.dumps(self.items, indent=1))
    @staticmethod
    def _tag(line):
        l = line.lower()
        if re.search(r"\b(prefer|always|please use|step by step|step-by-step|don't use|do not use)\b", l): return "[Procedural]"
        if re.search(r"\b(i am|i'm|my name|i live|i work|allergic|vegetarian|vegan|i moved)\b", l): return "[Semantic]"
        if re.search(r"\b(i built|we discussed|i tried|i decided|yesterday|last week|project)\b", l): return "[Episodic]"
        return None
    def add(self, messages, user_id):
        out = []
        for m in messages:
            role, text = (m["role"], m["content"]) if isinstance(m, dict) else (m.type, m.content)
            if role not in ("user", "human"): continue
            for sent in re.split(r"(?<=[.!?])\s+", text):
                tag = self._tag(sent)
                if tag:
                    rec = {"id": uuid.uuid4().hex[:8], "user_id": user_id, "memory": f"{tag} The user said: {sent.strip()}", "ts": time.time()}
                    self.items.append(rec); out.append(rec)
        self._save(); return out
    def add_raw(self, text, user_id, tag="[Semantic]"):
        """Store an already-tagged fact verbatim (used by update(), so a replacement is never lost to the tagger)."""
        rec = {"id": uuid.uuid4().hex[:8], "user_id": user_id, "memory": f"{tag} {text.strip()}", "ts": time.time()}
        self.items.append(rec); self._save(); return [rec]
    def get_all(self, user_id): return [i for i in self.items if i["user_id"] == user_id]
    def search(self, query, user_id, limit=5):
        mine = self.get_all(user_id)
        if not mine: return []
        emb = get_embedder(); q = emb.embed([query])[0]; vecs = emb.embed([m["memory"] for m in mine])
        res = [{**m, "score": round(cosine(q, v), 4)} for m, v in zip(mine, vecs)]
        return sorted(res, key=lambda r: -r["score"])[:limit]
    def delete(self, mid): self.items = [i for i in self.items if i["id"] != mid]; self._save()

class Mem0Store:
    """Real Mem0 with Groq LLM + local HF embedder + local Chroma. UNVERIFIED without an API key: check against the installed mem0ai version."""
    def __init__(self):
        from mem0 import Memory
        cfg = {"llm": {"provider": "groq", "config": {"model": config.GROQ_MODEL, "api_key": config.GROQ_API_KEY}},
               "embedder": {"provider": "huggingface", "config": {"model": config.EMBEDDING_MODEL}},
               "vector_store": {"provider": "chroma", "config": {"collection_name": "opsgate_mem", "path": str(config.MEM0_DIR / "mem0_chroma")}},
               "custom_instructions": CUSTOM_INSTRUCTIONS}
        self.m = Memory.from_config(cfg)
    def add(self, messages, user_id):
        msgs = [m if isinstance(m, dict) else {"role": "user" if m.type == "human" else "assistant", "content": m.content} for m in messages]
        r = self.m.add(msgs, user_id=user_id); return r.get("results", []) if isinstance(r, dict) else r
    def add_raw(self, text, user_id, tag="[Semantic]"):
        try: r = self.m.add([{"role": "user", "content": f"{tag} {text}"}], user_id=user_id, infer=False)   # verbatim, no LLM rewrite (verify on your mem0ai version)
        except TypeError: r = self.m.add([{"role": "user", "content": f"{tag} {text}"}], user_id=user_id)
        return r.get("results", []) if isinstance(r, dict) else r
    def get_all(self, user_id):
        try: r = self.m.get_all(filters={"user_id": user_id})
        except TypeError: r = self.m.get_all(user_id=user_id)
        return r.get("results", r) if isinstance(r, dict) else r
    def search(self, query, user_id, limit=5):
        try: r = self.m.search(query, filters={"user_id": user_id}, limit=limit)
        except TypeError: r = self.m.search(query, user_id=user_id, limit=limit)
        return r.get("results", r) if isinstance(r, dict) else r
    def delete(self, mid): self.m.delete(mid)

def get_store():
    return Mem0Store() if have_llm() else LocalStore()
