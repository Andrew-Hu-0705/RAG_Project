from functools import lru_cache

import chromadb
from sentence_transformers import CrossEncoder

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    FETCH_K,
    RERANK_MODEL,
    RERANK_SCORE_THRESHOLD,
    TOP_K,
)
from src.embeddings import embed_query


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=chromadb.Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(COLLECTION_NAME)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)


def retrieve(query: str, k: int = TOP_K, rerank: bool = True, fetch_k: int = FETCH_K):
    """Return (top_chunks, has_good_match).

    Embedding search alone (bi-encoder) is fast but imprecise -- it scores the
    query and each chunk independently, so it's easy for a topically-similar
    but wrong-paper chunk to outrank the right one in this corpus (heavy
    vocabulary overlap by design, see config.py). Reranking widens the initial
    net to fetch_k candidates, then scores each (query, chunk) pair jointly
    with a cross-encoder, which is slower per-pair but far more accurate at
    judging actual relevance -- and gives a real "no good match" signal that
    bi-encoder cosine distance doesn't reliably provide.
    """
    collection = get_collection()
    query_embedding = embed_query(query)
    n_results = fetch_k if rerank else k

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    candidates = [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]

    if not candidates:
        return [], False

    if rerank:
        reranker = get_reranker()
        pairs = [(query, c["text"]) for c in candidates]
        scores = reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        has_good_match = candidates[0]["rerank_score"] >= RERANK_SCORE_THRESHOLD
    else:
        has_good_match = True

    return candidates[:k], has_good_match
