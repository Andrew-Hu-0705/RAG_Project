# RAG over Agent & RAG-Eval Papers

A retrieval-augmented generation system over 28 arXiv papers on LLM agents, RAG methods, and
RAG evaluation — built with a real, labeled evaluation harness rather than a demo-only pipeline.

## Why this corpus

A tight topical cluster (RAG methods, agent/tool-use architectures, RAG evaluation methodology)
rather than a broad mix of papers. The vocabulary overlaps heavily across papers, which makes
retrieval a real discrimination problem instead of a trivial keyword match — a query like "how
does tool calling work" could plausibly match a dozen papers in this set, so getting the *right*
one requires actual semantic precision, not just topic matching.

## Architecture

```
arXiv API --> ingest.py --> chunk (sentence-packed, ~400 tokens, ~15% overlap)
                         --> embed (BAAI/bge-small-en-v1.5, local) --> Chroma (persisted)

query --> embed_query --> Chroma top-20 --> cross-encoder rerank --> top-5
                                                                    --> Claude Opus 5 (grounded, cited)
                                                                    --> Streamlit UI
```

- **Chunking**: sentence-boundary packing (not paragraph-splitting — pypdf's PDF extraction
  doesn't reliably preserve paragraph breaks), ~400 tokens with ~15% overlap.
- **Embeddings**: `BAAI/bge-small-en-v1.5`, local/open-source. Chosen over an API embedding
  model for zero marginal cost, no external dependency for anyone cloning the repo, and because
  it's specifically trained for retrieval (asymmetric query/passage encoding).
- **Vector store**: Chroma, persisted locally. No account, no API key, instant at this scale
  (~1,600 vectors).
- **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks the top 20 embedding-search
  candidates down to the top 5. Measured to matter (see Results) — not assumed.
- **Generation**: Claude Opus 5, prompted to answer only from numbered excerpts, cite them, and
  explicitly decline when the excerpts don't support an answer.

## Running it

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

python -m src.ingest              # download papers, chunk, embed, index (~30-40 min, one-time)
python -m src.build_eval_set      # draft the labeled eval set (review before trusting it)
python -m src.evaluate            # run the eval harness -> results/eval_report.json
streamlit run app.py              # demo UI
```

## Evaluation methodology

30-50 question eval sets built by hand don't scale for a solo project, so this one uses a
hybrid approach, disclosed rather than hidden:

- **28 answerable questions**, each drafted by Claude from one specific, sampled chunk per
  paper — so the source chunk is the retrieval ground truth by construction, not inferred after
  the fact. Reviewed by hand afterward: 3 of the 28 auto-drafted questions turned out to be
  sourced from bibliography/acknowledgements chunks (not something a real user would ask) and
  were redrafted from actual method/results content before any metric was computed.
- **5 unanswerable questions**, hand-written (not model-drafted) — plausible-sounding but
  genuinely outside the corpus, to test whether the system declines rather than fabricates.
- **Retrieval scoring**: precision@k / recall@k against the labeled gold chunk, computed with
  and without reranking to isolate its actual effect.
- **Answer scoring**: LLM-as-judge (Claude Opus 5) grading each answer against the reference
  answer on correct/partial/incorrect. **Known limitation**: the judge is the same model family
  as the generator, a self-preference risk. Partially mitigated by grading against a fixed
  reference answer rather than open-ended quality, but not eliminated.
- **Abstention**: measured at two levels — the pre-generation retrieval-score gate, and whether
  the system's actual final answer text declines. These turned out to diverge significantly (see
  Results) — worth understanding before you trust either number in isolation.

## Results

Full output in [`results/eval_report.json`](results/eval_report.json). Summary:

| Metric | Value |
|---|---|
| Corpus | 28 papers, 1,638 chunks |
| Eval set | 33 questions (28 answerable, 5 unanswerable) |
| Retrieval recall@5 — with reranking | 78.6% |
| Retrieval recall@5 — without reranking | 67.9% |
| Retrieval precision@1 — with reranking | 53.6% |
| Retrieval precision@1 — without reranking | 32.1% |
| RAG answer accuracy (LLM-judge vs. reference) | 89.3% correct, 100% correct-or-partial |
| Baseline (no retrieval) answer accuracy | 67.9% correct, 100% correct-or-partial |
| RAG final-answer abstention on unanswerable questions | 100% |
| Baseline final-answer abstention on unanswerable questions | 20% |
| Retrieval-gate abstention (pre-generation threshold alone) | 20% |

**On that last row**: the pre-generation retrieval-score threshold, by itself, is a weak
safety gate (20% catch rate) — it's the system prompt instruction doing the real work of
getting the model to decline (100% catch rate in the final answer). Initially assumed the
threshold was a structural guarantee; the eval showed it wasn't, which is exactly the kind of
thing a real harness is supposed to catch.

## Limitations

- Judge and generator share a model family (self-preference risk, not eliminated).
- Eval set is 33 questions from 28 papers — enough to be a real signal, not enough to be a
  tight confidence interval. Treat percentages as directional, not precise to the decimal.
- Retrieval ground truth is single-chunk per question; a question answerable from multiple
  chunks (e.g. restated across intro and results) would still only count one as "gold," which
  can understate recall in edge cases.
