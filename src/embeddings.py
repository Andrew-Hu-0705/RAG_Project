from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, QUERY_INSTRUCTION


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed document chunks for storage. bge-small-en-v1.5 needs no
    instruction prefix on the passage side -- only queries get one."""
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


def embed_query(text: str) -> list[float]:
    model = get_model()
    return model.encode(
        QUERY_INSTRUCTION + text, normalize_embeddings=True, show_progress_bar=False
    ).tolist()
