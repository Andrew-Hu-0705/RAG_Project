"""Evaluation harness: retrieval quality (precision@k/recall@k, with vs
without reranking), answer quality (LLM-judge against reference answers,
RAG vs no-retrieval baseline), and abstention on unanswerable questions.

Usage: python -m src.evaluate
Writes results/eval_report.json and prints a summary.
"""

import json
import statistics

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

from src.config import EVAL_K_VALUES, EVAL_SET_PATH, JUDGE_MODEL, PAPERS, RESULTS_DIR
from src.generate import generate_answer, generate_baseline
from src.retrieval import retrieve

load_dotenv()
client = anthropic.Anthropic()

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "reasoning"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """You are grading whether a candidate answer is factually consistent with a reference answer, for a RAG evaluation harness.

Score:
- "correct": the candidate captures the key facts in the reference answer, with no contradictions or fabricated claims
- "partial": the candidate is broadly on-topic and includes some correct information from the reference, but misses key facts or includes minor inaccuracies
- "incorrect": the candidate contradicts the reference, is substantively wrong, or fails to actually answer the question

Grade only against the reference answer's content -- not writing style or length."""


def judge(question: str, reference_answer: str, candidate_answer: str) -> dict:
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=512,
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": JUDGE_SCHEMA},
        },
        system=JUDGE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Reference answer: {reference_answer}\n\n"
                    f"Candidate answer: {candidate_answer}"
                ),
            }
        ],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return json.loads(text) if text else {"verdict": "incorrect", "reasoning": "empty judge response"}


def chunk_id(chunk: dict) -> str:
    return f"{chunk['metadata']['paper_id']}_{chunk['metadata']['chunk_index']}"


def precision_recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> tuple[float, float]:
    top_k = retrieved_ids[:k]
    hits = len(set(top_k) & set(gold_ids))
    precision = hits / k
    recall = hits / len(gold_ids) if gold_ids else 0.0
    return precision, recall


def evaluate_retrieval(answerable: list[dict]) -> dict:
    max_k = max(EVAL_K_VALUES)
    per_variant = {"rerank": {k: [] for k in EVAL_K_VALUES}, "no_rerank": {k: [] for k in EVAL_K_VALUES}}

    for item in tqdm(answerable, desc="Retrieval eval"):
        for variant, rerank_flag in [("rerank", True), ("no_rerank", False)]:
            chunks, _ = retrieve(item["question"], k=max_k, rerank=rerank_flag)
            retrieved_ids = [chunk_id(c) for c in chunks]
            for k in EVAL_K_VALUES:
                p, r = precision_recall_at_k(retrieved_ids, item["gold_chunk_ids"], k)
                per_variant[variant][k].append({"precision": p, "recall": r})

    summary = {}
    for variant, by_k in per_variant.items():
        summary[variant] = {
            k: {
                "precision": round(statistics.mean(x["precision"] for x in scores), 3),
                "recall": round(statistics.mean(x["recall"] for x in scores), 3),
            }
            for k, scores in by_k.items()
        }
    return summary


def evaluate_answers(answerable: list[dict]) -> dict:
    rag_verdicts, baseline_verdicts = [], []
    examples = []

    for item in tqdm(answerable, desc="Answer quality eval"):
        rag_result = generate_answer(item["question"])
        baseline_answer = generate_baseline(item["question"])

        rag_judged = judge(item["question"], item["reference_answer"], rag_result["answer"])
        baseline_judged = judge(item["question"], item["reference_answer"], baseline_answer)

        rag_verdicts.append(rag_judged["verdict"])
        baseline_verdicts.append(baseline_judged["verdict"])
        examples.append(
            {
                "id": item["id"],
                "question": item["question"],
                "rag_verdict": rag_judged["verdict"],
                "baseline_verdict": baseline_judged["verdict"],
            }
        )

    def accuracy(verdicts: list[str]) -> dict:
        n = len(verdicts)
        return {
            "correct_rate": round(verdicts.count("correct") / n, 3),
            "correct_or_partial_rate": round(
                (verdicts.count("correct") + verdicts.count("partial")) / n, 3
            ),
        }

    return {
        "rag": accuracy(rag_verdicts),
        "baseline": accuracy(baseline_verdicts),
        "examples": examples,
    }


ABSTENTION_SCHEMA = {
    "type": "object",
    "properties": {"attempted_answer": {"type": "boolean"}},
    "required": ["attempted_answer"],
    "additionalProperties": False,
}

ABSTENTION_SYSTEM = """Classify whether the given answer actually attempts to answer the question with
specific claims, or instead declines / says the information isn't available. Hedging
that still asserts specific facts ("SAM generally underperforms on medical images, e.g. X, Y")
counts as attempting an answer. A refusal, "I don't know", or "the provided material
doesn't cover this" counts as not attempting."""


def classify_attempted_answer(question: str, answer: str) -> bool:
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=256,
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": ABSTENTION_SCHEMA},
        },
        system=ABSTENTION_SYSTEM,
        messages=[{"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return json.loads(text)["attempted_answer"] if text else True


def evaluate_abstention(unanswerable: list[dict]) -> dict:
    """Two distinct signals worth keeping separate:
    - retrieval_gate: did the pre-generation rerank-score threshold refuse to
      even call the LLM (a structural guarantee, independent of the model's
      own judgment)
    - final_answer: did the system's actual output text decline to answer
      (what the user actually sees, whichever path produced it)
    """
    retrieval_gate_abstained = 0
    rag_final_abstained, baseline_final_abstained = 0, 0
    examples = []

    for item in tqdm(unanswerable, desc="Abstention eval"):
        rag_result = generate_answer(item["question"])
        baseline_answer = generate_baseline(item["question"])

        gate_abstained = not rag_result["grounded"]
        rag_attempted = classify_attempted_answer(item["question"], rag_result["answer"])
        baseline_attempted = classify_attempted_answer(item["question"], baseline_answer)

        retrieval_gate_abstained += int(gate_abstained)
        rag_final_abstained += int(not rag_attempted)
        baseline_final_abstained += int(not baseline_attempted)

        examples.append(
            {
                "id": item["id"],
                "question": item["question"],
                "retrieval_gate_abstained": gate_abstained,
                "rag_final_answer_abstained": not rag_attempted,
                "baseline_final_answer_abstained": not baseline_attempted,
                "rag_answer": rag_result["answer"],
                "baseline_answer_preview": baseline_answer[:200],
            }
        )

    n = len(unanswerable)
    return {
        "retrieval_gate_abstention_rate": round(retrieval_gate_abstained / n, 3) if n else None,
        "rag_final_answer_abstention_rate": round(rag_final_abstained / n, 3) if n else None,
        "baseline_final_answer_abstention_rate": round(baseline_final_abstained / n, 3) if n else None,
        "examples": examples,
    }


def run():
    eval_set = json.loads(EVAL_SET_PATH.read_text())
    answerable = [q for q in eval_set if q["answerable"]]
    unanswerable = [q for q in eval_set if not q["answerable"]]

    retrieval_summary = evaluate_retrieval(answerable)
    answer_summary = evaluate_answers(answerable)
    abstention_summary = evaluate_abstention(unanswerable)

    report = {
        "corpus_size_papers": len(PAPERS),
        "eval_set_size": len(eval_set),
        "eval_set_answerable": len(answerable),
        "eval_set_unanswerable": len(unanswerable),
        "retrieval": retrieval_summary,
        "answer_quality": answer_summary,
        "abstention": abstention_summary,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "eval_report.json").write_text(json.dumps(report, indent=2))
    print_summary(report)
    return report


def print_summary(report: dict):
    print("\n=== Retrieval quality (precision@k / recall@k) ===")
    for variant in ["rerank", "no_rerank"]:
        print(f"  {variant}:")
        for k, m in report["retrieval"][variant].items():
            print(f"    k={k}: precision={m['precision']:.3f}  recall={m['recall']:.3f}")

    print("\n=== Answer quality (LLM-judge vs reference answers) ===")
    for system in ["rag", "baseline"]:
        m = report["answer_quality"][system]
        print(f"  {system}: correct={m['correct_rate']:.1%}  correct-or-partial={m['correct_or_partial_rate']:.1%}")

    print("\n=== Abstention on unanswerable questions ===")
    a = report["abstention"]
    print(f"  Retrieval-gate abstention (structural, pre-generation): {a['retrieval_gate_abstention_rate']:.1%}")
    print(f"  RAG final-answer abstention rate:      {a['rag_final_answer_abstention_rate']:.1%}")
    print(f"  Baseline final-answer abstention rate: {a['baseline_final_answer_abstention_rate']:.1%}")

    print(f"\nCorpus: {report['corpus_size_papers']} papers. Eval set: {report['eval_set_size']} questions "
          f"({report['eval_set_answerable']} answerable, {report['eval_set_unanswerable']} unanswerable).")


if __name__ == "__main__":
    run()
