"""Threshold-gated retrieval. Returns ORIGINAL chunks only (never the enrichment summary)."""
from functools import lru_cache
from core import config
from core.llm import get_embedder
from ingestion.contextual_indexer import get_collection

@lru_cache(maxsize=256)
def _embed_query(query: str) -> tuple:  # repeated / retried questions skip the embedding model
    return tuple(get_embedder().embed([query])[0])

def retrieve(query: str, k: int = 5, threshold: float | None = None, use_threshold: bool = True) -> list[dict]:
    th = config.SIM_THRESHOLD if threshold is None else threshold
    col = get_collection()
    if col.count() == 0: return []
    q = list(_embed_query(query))                        # SAME model as indexing
    res = col.query(query_embeddings=[q], n_results=k)
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = 1 - dist                                    # cosine distance -> similarity
        if use_threshold and sim < th: continue
        out.append({"text": doc, "score": round(sim, 4), "section": meta.get("section") or meta.get("chapter"), "source": meta.get("source"), "kind": meta.get("kind")})
    return out
