"""Memory lifecycle: extract -> store -> retrieve -> inject.
Semantic + Procedural: cold-loaded into the static system prompt (no retrieval).
Episodic: retrieved per query (threshold 0.45) and injected into the user section."""
import hashlib, json
from core import config
from memory.mem0_store import get_store, tags_of, strip_tags
from memory.dedup import dedupe

class MemoryManager:
    def __init__(self, user_id: str, store=None):
        self.user_id, self.store = user_id, store or get_store()
        self._dedupe_cache: dict = {}
    def _by_tag(self, tag):
        return [r for r in self.store.get_all(self.user_id) if tag in tags_of(r["memory"])]
    def _dedupe_once(self, records):
        """Same memories => same merged text, with no repeat LLM call. A byte-stable system prompt keeps the KV cache warm across sessions."""
        key = hashlib.md5(json.dumps([(r.get("id"), r["memory"]) for r in records], sort_keys=True).encode()).hexdigest()
        if key not in self._dedupe_cache: self._dedupe_cache[key] = sorted(dedupe(records))
        return self._dedupe_cache[key]
    def bootstrap(self) -> tuple[list[str], list[str]]:
        """Cold-load: de-duplicated semantic and procedural lists for the system prompt."""
        return self._dedupe_once(self._by_tag("Semantic")), self._dedupe_once(self._by_tag("Procedural"))
    def episodic(self, query: str, k: int = 2, threshold: float | None = None) -> list[str]:
        th = config.SIM_THRESHOLD if threshold is None else threshold
        hits = [r for r in self.store.search(query, self.user_id, limit=10) if "Episodic" in tags_of(r["memory"]) and r.get("score", 0) >= th]
        return [strip_tags(r["memory"]) for r in hits[:k]]
    def end_session(self, messages):
        """Post-session extraction pipeline. The system prompt and summary notes are not conversation, so they are skipped."""
        from core.summarizer import NOTE
        convo = [m for m in messages if getattr(m, "type", "human") != "system" and not str(getattr(m, "content", m.get("content", "") if isinstance(m, dict) else "")).startswith(NOTE)]
        return self.store.add(convo, self.user_id) if convo else []
    def forget(self, needle: str) -> int:
        gone = [r for r in self.store.get_all(self.user_id) if needle.lower() in r["memory"].lower()]
        for r in gone: self.store.delete(r["id"])
        return len(gone)
    def update(self, old_needle: str, new_text: str, tag: str = "[Semantic]"):
        """Mem0 appends instead of replacing; this wrapper forgets the obsolete fact, then stores the new one."""
        self.forget(old_needle)
        return self.store.add_raw(new_text, self.user_id, tag)
