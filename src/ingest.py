"""Download arXiv papers, chunk them, embed the chunks, and load them into
Chroma. Re-running after adding new IDs to config.PAPERS only processes the
new ones (see data/papers/processed.json). Pass --force to redo everything.
"""

import argparse
import json
import re

import arxiv
import chromadb
from pypdf import PdfReader
from tqdm import tqdm
import tiktoken

from src.config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR, PAPERS, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS
from src.embeddings import embed_passages

PROCESSED_PATH = DATA_DIR / "processed.json"
METADATA_PATH = DATA_DIR / "metadata.json"

_ENC = tiktoken.get_encoding("cl100k_base")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _load_json(path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default


def _save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2))


def download_papers(paper_ids: list[str]) -> dict:
    """Fetch metadata + PDF for each paper id not already on disk."""
    client = arxiv.Client()
    metadata = _load_json(METADATA_PATH, {})

    for paper_id in tqdm(paper_ids, desc="Downloading"):
        pdf_path = DATA_DIR / f"{paper_id}.pdf"
        if pdf_path.exists() and paper_id in metadata:
            continue
        result = next(client.results(arxiv.Search(id_list=[paper_id])))
        result.download_pdf(dirpath=str(DATA_DIR), filename=f"{paper_id}.pdf")
        metadata[paper_id] = {
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "url": result.entry_id,
        }

    _save_json(METADATA_PATH, metadata)
    return metadata


def extract_text(pdf_path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """Greedily pack sentences into ~chunk_size-token windows, carrying the
    last ~overlap tokens of each chunk into the next so an idea that straddles
    a boundary isn't lost to just one side."""
    sentences = split_sentences(text)
    chunks, current, current_tokens = [], [], 0

    for sentence in sentences:
        sent_tokens = len(_ENC.encode(sentence))
        if current_tokens + sent_tokens > chunk_size and current:
            chunks.append(" ".join(current))
            carry, carry_tokens = [], 0
            for s in reversed(current):
                t = len(_ENC.encode(s))
                if carry_tokens + t > overlap:
                    break
                carry.insert(0, s)
                carry_tokens += t
            current, current_tokens = carry, carry_tokens
        current.append(sentence)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


def ingest(force: bool = False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paper_ids = list(PAPERS.keys())

    processed = set(_load_json(PROCESSED_PATH, []))
    if force:
        processed = set()
    todo = [pid for pid in paper_ids if pid not in processed]

    if not todo:
        print("Nothing new to ingest. Pass --force to reprocess everything.")
        return

    metadata = download_papers(todo)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=chromadb.Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    chunk_counts = {}
    for paper_id in tqdm(todo, desc="Chunking + embedding"):
        pdf_path = DATA_DIR / f"{paper_id}.pdf"
        text = extract_text(pdf_path)
        chunks = chunk_text(text)
        if not chunks:
            print(f"WARNING: no extractable text for {paper_id}, skipping")
            continue
        chunk_counts[paper_id] = len(chunks)

        embeddings = embed_passages(chunks)
        ids = [f"{paper_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "paper_id": paper_id,
                "title": metadata[paper_id]["title"],
                "url": metadata[paper_id]["url"],
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]
        collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        processed.add(paper_id)
        _save_json(PROCESSED_PATH, sorted(processed))

    total_chunks = collection.count()
    print("\n--- Ingestion summary ---")
    print(f"Papers processed this run: {len(chunk_counts)}")
    print(f"Total papers in corpus:    {len(processed)}")
    print(f"Total chunks in Chroma:    {total_chunks}")
    if chunk_counts:
        avg = sum(chunk_counts.values()) / len(chunk_counts)
        print(f"Avg chunks/paper (this run): {avg:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="reprocess all papers from scratch")
    args = parser.parse_args()
    ingest(force=args.force)
