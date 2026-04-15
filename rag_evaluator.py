from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class EvaluationRecord:
    question_id: str
    question: str
    expected_answer: str
    predicted_answer: str
    expected_paragraph: Optional[str] = None
    retrieved_paragraph: Optional[str] = None
    expected_document_id: Optional[str] = None
    retrieved_document_id: Optional[str] = None


@dataclass(frozen=True)
class QuestionResult:
    question_id: str
    answer_accuracy: float
    answer_usefulness: float
    paragraph_retrieval_score: Optional[float]
    document_retrieval_score: Optional[float]
    overall_score: float


def _normalize_text(value: str) -> str:
    tokens = _TOKEN_RE.findall(value.lower())
    return " ".join(tokens)


def _token_set(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.lower()))


def _extract_number(value: str) -> Optional[float]:
    match = _NUMBER_RE.search(value.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _token_f1(expected: str, predicted: str) -> float:
    expected_tokens = _token_set(expected)
    predicted_tokens = _token_set(predicted)
    if not expected_tokens and not predicted_tokens:
        return 1.0
    if not expected_tokens or not predicted_tokens:
        return 0.0

    common = len(expected_tokens & predicted_tokens)
    if common == 0:
        return 0.0

    precision = common / len(predicted_tokens)
    recall = common / len(expected_tokens)
    return (2 * precision * recall) / (precision + recall)


def score_answer_accuracy(expected: str, predicted: str, numeric_tolerance: float = 1e-6) -> float:
    expected_number = _extract_number(expected)
    predicted_number = _extract_number(predicted)

    if expected_number is not None and predicted_number is not None:
        return 1.0 if math.isclose(expected_number, predicted_number, rel_tol=0.0, abs_tol=numeric_tolerance) else 0.0

    return 1.0 if _normalize_text(expected) == _normalize_text(predicted) else 0.0


def score_answer_usefulness(expected: str, predicted: str) -> float:
    expected_number = _extract_number(expected)
    predicted_number = _extract_number(predicted)

    if expected_number is not None and predicted_number is not None:
        denominator = max(abs(expected_number), abs(predicted_number), 1.0)
        relative_error = abs(expected_number - predicted_number) / denominator
        return max(0.0, 1.0 - min(relative_error, 1.0))

    return _token_f1(expected, predicted)


def score_paragraph_retrieval(expected_paragraph: Optional[str], retrieved_paragraph: Optional[str]) -> Optional[float]:
    if expected_paragraph is None:
        return None
    if not retrieved_paragraph:
        return 0.0
    return _token_f1(expected_paragraph, retrieved_paragraph)


def score_document_retrieval(expected_document_id: Optional[str], retrieved_document_id: Optional[str]) -> Optional[float]:
    if expected_document_id is None:
        return None
    if not retrieved_document_id:
        return 0.0
    return 1.0 if _normalize_text(expected_document_id) == _normalize_text(retrieved_document_id) else 0.0


def evaluate_question(record: EvaluationRecord) -> QuestionResult:
    answer_accuracy = score_answer_accuracy(record.expected_answer, record.predicted_answer)
    answer_usefulness = score_answer_usefulness(record.expected_answer, record.predicted_answer)
    paragraph_score = score_paragraph_retrieval(record.expected_paragraph, record.retrieved_paragraph)
    document_score = score_document_retrieval(record.expected_document_id, record.retrieved_document_id)

    parts = [answer_accuracy, answer_usefulness]
    if paragraph_score is not None:
        parts.append(paragraph_score)
    if document_score is not None:
        parts.append(document_score)

    overall_score = sum(parts) / len(parts)

    return QuestionResult(
        question_id=record.question_id,
        answer_accuracy=answer_accuracy,
        answer_usefulness=answer_usefulness,
        paragraph_retrieval_score=paragraph_score,
        document_retrieval_score=document_score,
        overall_score=overall_score,
    )


def evaluate_batch(records: list[EvaluationRecord]) -> list[QuestionResult]:
    return [evaluate_question(record) for record in records]


def summarize_results(results: list[QuestionResult]) -> dict[str, float]:
    if not results:
        return {
            "questions": 0,
            "avg_answer_accuracy": 0.0,
            "avg_answer_usefulness": 0.0,
            "avg_paragraph_retrieval": 0.0,
            "avg_document_retrieval": 0.0,
            "avg_overall": 0.0,
        }

    paragraph_scores = [r.paragraph_retrieval_score for r in results if r.paragraph_retrieval_score is not None]
    document_scores = [r.document_retrieval_score for r in results if r.document_retrieval_score is not None]

    return {
        "questions": len(results),
        "avg_answer_accuracy": sum(r.answer_accuracy for r in results) / len(results),
        "avg_answer_usefulness": sum(r.answer_usefulness for r in results) / len(results),
        "avg_paragraph_retrieval": (sum(paragraph_scores) / len(paragraph_scores)) if paragraph_scores else 0.0,
        "avg_document_retrieval": (sum(document_scores) / len(document_scores)) if document_scores else 0.0,
        "avg_overall": sum(r.overall_score for r in results) / len(results),
    }


def evaluate_systems(records_by_system: dict[str, list[EvaluationRecord]]) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for system_name, records in records_by_system.items():
        question_results = evaluate_batch(records)
        output[system_name] = {
            "questions": question_results,
            "summary": summarize_results(question_results),
        }
    return output
