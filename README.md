# RAG_evaluator

Code that evaluates accuracy and usefulness of observatory RAG system answers on questions.

## What is evaluated

For each question and each RAG system answer, the evaluator computes:

- **answer_accuracy**: strict correctness (exact number/text after normalization)
- **answer_usefulness**: softer usefulness score (numeric closeness or token overlap)
- **paragraph_retrieval_score**: how well retrieved paragraph matches expected source paragraph
- **document_retrieval_score**: whether retrieved document ID matches expected document ID
- **overall_score**: average of available metrics for that question

## Minimal usage

```python
from rag_evaluator import EvaluationRecord, evaluate_systems

records_by_system = {
    "rag_v1": [
        EvaluationRecord(
            question_id="q1",
            question="How many telescopes are in array A?",
            expected_answer="4",
            predicted_answer="4",
            expected_paragraph="Array A consists of 4 telescopes.",
            retrieved_paragraph="Array A consists of 4 telescopes and support systems.",
            expected_document_id="DOC-A",
            retrieved_document_id="doc-a",
        )
    ]
}

results = evaluate_systems(records_by_system)
print(results["rag_v1"]["summary"])
```
