from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "papers"
CHROMA_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "arxiv_papers"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 60

# --- Retrieval ---
TOP_K = 5
FETCH_K = 20  # candidates pulled by embedding search before reranking
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Cross-encoder relevance score (raw logit) below which we treat retrieval as
# "no good match" rather than pass weak chunks to the generator. Placeholder --
# calibrate against the labeled eval set in stage 4.
RERANK_SCORE_THRESHOLD = -3.0

# --- Generation ---
GENERATION_MODEL = "claude-opus-5"

# --- Evaluation ---
# Judge reuses the generation model. Known limitation: same model family as
# the generator being judged carries a self-preference risk. Mitigated by
# grading against a fixed reference answer rather than open-ended quality.
JUDGE_MODEL = "claude-opus-5"
EVAL_SET_PATH = ROOT_DIR / "data" / "eval_set.json"
EVAL_SET_DRAFT_PATH = ROOT_DIR / "data" / "eval_set_draft.json"
RESULTS_DIR = ROOT_DIR / "results"
EVAL_K_VALUES = [1, 3, 5]

# arXiv IDs, grouped by sub-topic. Deliberately a tight cluster (RAG methods,
# agent/tool-use architectures, and evaluation methodology) so the vocabulary
# overlaps heavily -- that's what makes retrieval a real test instead of a
# trivial keyword match. Titles are for human sanity-checking only; the
# ingestion script fetches ground-truth metadata from the arXiv API itself.
PAPERS = {
    # --- Core RAG / retrieval ---
    "2005.11401": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "2004.04906": "Dense Passage Retrieval for Open-Domain Question Answering",
    "2002.08909": "REALM: Retrieval-Augmented Language Model Pre-Training",
    "2007.01282": "Leveraging Passage Retrieval with Generative Models for Open Domain QA",
    "2212.10496": "Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)",
    "2310.11511": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
    "2401.15884": "Corrective Retrieval Augmented Generation",
    "2305.06983": "Active Retrieval Augmented Generation (FLARE)",
    "2307.03172": "Lost in the Middle: How Language Models Use Long Contexts",
    "2312.10997": "Retrieval-Augmented Generation for Large Language Models: A Survey",
    # --- Agents / tool use ---
    "2210.03629": "ReAct: Synergizing Reasoning and Acting in Language Models",
    "2302.04761": "Toolformer: Language Models Can Teach Themselves to Use Tools",
    "2303.11366": "Reflexion: Language Agents with Verbal Reinforcement Learning",
    "2205.00445": "MRKL Systems: Modular Reasoning, Knowledge and Language",
    "2210.03350": "Measuring and Narrowing the Compositionality Gap in Language Models (Self-Ask)",
    "2304.03442": "Generative Agents: Interactive Simulacra of Human Behavior",
    "2305.16291": "Voyager: An Open-Ended Embodied Agent with Large Language Models",
    "2303.17580": "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace",
    "2305.15334": "Gorilla: Large Language Model Connected with Massive APIs",
    "2308.11432": "A Survey on Large Language Model based Autonomous Agents",
    # --- Reasoning (supports agent behavior) ---
    "2201.11903": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
    "2305.10601": "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
    "2112.09332": "WebGPT: Browser-assisted Question-Answering with Human Feedback",
    "2305.04091": "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning",
    # --- Evaluation methodology ---
    "2309.15217": "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
    "2311.09476": "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems",
    "2306.05685": "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
    "2305.14283": "Query Rewriting for Retrieval-Augmented Large Language Models",
}
