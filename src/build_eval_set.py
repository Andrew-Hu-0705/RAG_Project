"""Draft the eval set: one question per paper, generated from a specific
sampled chunk so that chunk is automatically the retrieval ground truth,
plus a handful of hand-written unanswerable questions.

This is a DRAFT. Read data/eval_set_draft.json before trusting any number
computed from it -- rename it to eval_set.json (or edit it first) once
you've reviewed it.
"""

import json

import anthropic
from dotenv import load_dotenv

from src.config import EVAL_SET_DRAFT_PATH, GENERATION_MODEL
from src.retrieval import get_collection

load_dotenv()
client = anthropic.Anthropic()

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "reference_answer": {"type": "string"},
    },
    "required": ["question", "reference_answer"],
    "additionalProperties": False,
}

DRAFT_SYSTEM = """You are building a QA eval set for a RAG system. Given one excerpt from a paper, write ONE question that:
- can be answered using ONLY the information in this excerpt, not general knowledge
- is specific enough that a vague or generic answer would not satisfy it
- reads like something a researcher skimming this topic area would actually ask

Then write a concise reference answer (2-4 sentences) using only the excerpt's content."""

# Hand-written, not model-drafted: plausible-sounding but genuinely outside
# the corpus, to test whether the system abstains instead of hallucinating.
UNANSWERABLE_QUESTIONS = [
    "What chunk overlap percentage does the RAG survey paper recommend for legal contracts?",
    "How does the Segment Anything Model (SAM) perform on medical image segmentation benchmarks?",
    "What was the reported carbon footprint of training GPT-3, according to the papers in this corpus?",
    "What programming language is Toolformer's tool-calling API implemented in?",
    "What GPU cluster configuration was used to train the underlying model behind Reflexion?",
]


def sample_chunks_per_paper():
    """One representative chunk per paper. Skips the first chunk (usually
    title/authors/abstract boilerplate) and picks roughly a third of the way
    through, aiming for method/results content specific enough to support a
    non-generic question."""
    collection = get_collection()
    all_data = collection.get(include=["documents", "metadatas"])

    by_paper = {}
    for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
        by_paper.setdefault(meta["paper_id"], []).append((doc, meta))

    chosen = []
    for paper_id, chunks in sorted(by_paper.items()):
        chunks.sort(key=lambda x: x[1]["chunk_index"])
        candidates = chunks[1:] if len(chunks) > 1 else chunks
        idx = len(candidates) // 3
        chosen.append(candidates[idx])
    return chosen


def draft_question(doc_text: str, title: str) -> dict:
    response = client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=1024,
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": QUESTION_SCHEMA},
        },
        system=DRAFT_SYSTEM,
        messages=[{"role": "user", "content": f"Paper: {title}\n\nExcerpt:\n{doc_text}"}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def build_draft():
    chosen = sample_chunks_per_paper()
    eval_set = []

    for i, (doc_text, meta) in enumerate(chosen, start=1):
        print(f"[{i}/{len(chosen)}] drafting question from: {meta['title'][:60]}...")
        drafted = draft_question(doc_text, meta["title"])
        eval_set.append(
            {
                "id": f"q{i:03d}",
                "question": drafted["question"],
                "reference_answer": drafted["reference_answer"],
                "gold_chunk_ids": [f"{meta['paper_id']}_{meta['chunk_index']}"],
                "source_paper": meta["title"],
                "answerable": True,
            }
        )

    for i, question in enumerate(UNANSWERABLE_QUESTIONS, start=1):
        eval_set.append(
            {
                "id": f"u{i:03d}",
                "question": question,
                "reference_answer": None,
                "gold_chunk_ids": [],
                "source_paper": None,
                "answerable": False,
            }
        )

    EVAL_SET_DRAFT_PATH.write_text(json.dumps(eval_set, indent=2))
    print(f"\nWrote {len(eval_set)} draft questions to {EVAL_SET_DRAFT_PATH}")
    print(f"  {len(chosen)} answerable, {len(UNANSWERABLE_QUESTIONS)} unanswerable")
    print("\nReview this file before trusting any metric computed from it.")


if __name__ == "__main__":
    build_draft()
