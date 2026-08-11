import anthropic
from dotenv import load_dotenv

from src.config import GENERATION_MODEL
from src.retrieval import retrieve

load_dotenv()
client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a research assistant answering questions using only the numbered excerpts provided below, drawn from a corpus of papers on RAG, LLM agents, and evaluation.

Rules:
- Answer only from the excerpts. Do not use outside knowledge.
- Cite the excerpt number(s) that support each claim, like [1] or [2, 4].
- If the excerpts don't actually contain enough information to answer the question, say so explicitly instead of guessing."""

NO_MATCH_MESSAGE = (
    "I don't have relevant information in the corpus to answer this question confidently."
)


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] ({c['metadata']['title']})\n{c['text']}")
    return "\n\n".join(parts)


def generate_answer(query: str, k: int = 5, rerank: bool = True) -> dict:
    chunks, has_good_match = retrieve(query, k=k, rerank=rerank)

    if not has_good_match:
        return {
            "answer": NO_MATCH_MESSAGE,
            "sources": [],
            "chunks": chunks,
            "grounded": False,
        }

    context = format_context(chunks)
    response = client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=4096,
        output_config={"effort": "medium"},
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Excerpts:\n\n{context}\n\nQuestion: {query}"}
        ],
    )
    answer = next((b.text for b in response.content if b.type == "text"), "")

    return {
        "answer": answer,
        "sources": [
            {
                "index": i + 1,
                "title": c["metadata"]["title"],
                "paper_id": c["metadata"]["paper_id"],
                "url": c["metadata"]["url"],
            }
            for i, c in enumerate(chunks)
        ],
        "chunks": chunks,
        "grounded": True,
    }


def generate_baseline(query: str) -> str:
    """No retrieval -- same model answering from parametric knowledge only.
    Used in stage 4 to measure whether retrieval actually helps."""
    response = client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=4096,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": query}],
    )
    return next((b.text for b in response.content if b.type == "text"), "")


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "How does ReAct combine reasoning and acting?"
    result = generate_answer(query)
    print(f"Q: {query}\n")
    print(f"A: {result['answer']}\n")
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  [{s['index']}] {s['title']}")
