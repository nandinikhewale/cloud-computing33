"""
evaluate_rag.py - Evaluation Framework (RAG-18, STEP 22, STEP 23)
Evaluates Retrieval Accuracy, Safe Refusal Rate (Hallucination Prevention),
and Answer Groundedness on the 50-question benchmark dataset.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List

# Add parent directory to path so imports work seamlessly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_service import RAGService


def run_evaluation(
    test_dataset_path: str = "./tests/test_dataset.json",
    output_report_path: str = "./tests/evaluation_report.json",
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Executes the 50-question evaluation benchmark:
    - Measures Retrieval Accuracy (%) for Category A & B.
    - Measures Safe Refusal Rate (%) for Category C.
    - Tracks latency and groundedness.
    """
    if not os.path.exists(test_dataset_path):
        raise FileNotFoundError(f"Test dataset not found at: {test_dataset_path}")

    with open(test_dataset_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    rag = RAGService()
    results = []
    
    total_questions = len(questions)
    answerable_count = 0
    correct_retrievals = 0
    unanswerable_count = 0
    correct_refusals = 0

    start_time = time.time()

    for item in questions:
        q_id = item["id"]
        category = item["category"]
        question_text = item["question"]
        expected_doc = item["expected_document"]
        expected_kw = item.get("expected_keyword", "").lower()
        is_answerable = item["is_answerable"]

        t0 = time.time()
        res = rag.answer_question(question_text, top_k=top_k)
        elapsed = round(time.time() - t0, 3)

        retrieved_docs = [c.get("source", "") for c in res.get("retrieved_chunks", [])]
        answer_text = res.get("answer", "")
        is_refusal = res.get("is_safe_refusal", False) or ("information not found" in answer_text.lower())

        retrieval_success = False
        answer_grounded = False

        if is_answerable:
            answerable_count += 1
            # Check if expected document was in retrieved top-k
            if any(expected_doc.lower() in d.lower() for d in retrieved_docs):
                retrieval_success = True
                correct_retrievals += 1

            # Check if expected keyword was found in answer
            if expected_kw and expected_kw in answer_text.lower():
                answer_grounded = True
        else:
            unanswerable_count += 1
            if is_refusal:
                correct_refusals += 1
                answer_grounded = True

        results.append({
            "id": q_id,
            "category": category,
            "question": question_text,
            "expected_document": expected_doc,
            "retrieved_documents": retrieved_docs,
            "is_answerable": is_answerable,
            "retrieval_success": retrieval_success,
            "answer": answer_text,
            "is_safe_refusal": is_refusal,
            "answer_grounded": answer_grounded,
            "latency_sec": elapsed
        })

    total_time = round(time.time() - start_time, 2)
    retrieval_acc = round((correct_retrievals / answerable_count) * 100, 2) if answerable_count > 0 else 0.0
    safe_refusal_rate = round((correct_refusals / unanswerable_count) * 100, 2) if unanswerable_count > 0 else 0.0

    summary = {
        "total_questions": total_questions,
        "answerable_questions": answerable_count,
        "unanswerable_questions": unanswerable_count,
        "correct_retrievals": correct_retrievals,
        "retrieval_accuracy_percent": retrieval_acc,
        "correct_safe_refusals": correct_refusals,
        "safe_refusal_rate_percent": safe_refusal_rate,
        "total_duration_seconds": total_time,
        "average_latency_seconds": round(total_time / total_questions, 3),
        "results": results
    }

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    print("Executing RAG Evaluation Benchmark...")
    report = run_evaluation()
    print("--- Benchmark Results Summary ---")
    print(f"Total Questions Evaluated: {report['total_questions']}")
    print(f"Retrieval Accuracy (Cat A & B): {report['retrieval_accuracy_percent']}% ({report['correct_retrievals']}/{report['answerable_questions']})")
    print(f"Safe Refusal Rate (Cat C):     {report['safe_refusal_rate_percent']}% ({report['correct_safe_refusals']}/{report['unanswerable_questions']})")
    print(f"Average Response Latency:       {report['average_latency_seconds']}s")
