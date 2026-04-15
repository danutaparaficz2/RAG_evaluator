import unittest

from rag_evaluator import (
    EvaluationRecord,
    evaluate_batch,
    evaluate_question,
    evaluate_systems,
    score_answer_accuracy,
    score_answer_usefulness,
    score_document_retrieval,
    score_paragraph_retrieval,
    summarize_results,
)


class TestRagEvaluator(unittest.TestCase):
    def test_numeric_answer_accuracy_exact(self):
        self.assertEqual(score_answer_accuracy("42", "42.0"), 1.0)
        self.assertEqual(score_answer_accuracy("42", "43"), 0.0)

    def test_numeric_answer_usefulness_partial(self):
        self.assertAlmostEqual(score_answer_usefulness("100", "90"), 0.9)

    def test_text_answer_matching(self):
        self.assertEqual(score_answer_accuracy("Mars", "mars"), 1.0)
        self.assertGreater(score_answer_usefulness("very large telescope", "telescope is large"), 0.6)

    def test_retrieval_scores(self):
        paragraph_score = score_paragraph_retrieval(
            "The observatory is in the Atacama desert.",
            "Located in the Atacama desert, this observatory studies stars.",
        )
        self.assertIsNotNone(paragraph_score)
        self.assertGreater(paragraph_score, 0.5)

        self.assertEqual(score_document_retrieval("Doc-12", "doc 12"), 1.0)
        self.assertEqual(score_document_retrieval("Doc-12", "Doc-99"), 0.0)

    def test_question_and_summary(self):
        record = EvaluationRecord(
            question_id="q1",
            question="How many telescopes are in array A?",
            expected_answer="4",
            predicted_answer="4",
            expected_paragraph="Array A consists of 4 telescopes.",
            retrieved_paragraph="Array A consists of 4 telescopes and support systems.",
            expected_document_id="DOC-A",
            retrieved_document_id="doc a",
        )

        result = evaluate_question(record)
        self.assertEqual(result.answer_accuracy, 1.0)
        self.assertEqual(result.document_retrieval_score, 1.0)
        self.assertGreater(result.paragraph_retrieval_score, 0.5)
        self.assertGreater(result.overall_score, 0.8)

        summary = summarize_results([result])
        self.assertEqual(summary["questions"], 1)
        self.assertGreater(summary["avg_overall"], 0.8)

    def test_evaluate_systems(self):
        systems = {
            "rag_a": [
                EvaluationRecord(
                    question_id="1",
                    question="q",
                    expected_answer="10",
                    predicted_answer="10",
                )
            ],
            "rag_b": [
                EvaluationRecord(
                    question_id="1",
                    question="q",
                    expected_answer="10",
                    predicted_answer="9",
                )
            ],
        }

        results = evaluate_systems(systems)
        rag_a_score = results["rag_a"]["summary"]["avg_answer_accuracy"]
        rag_b_score = results["rag_b"]["summary"]["avg_answer_accuracy"]
        self.assertGreater(rag_a_score, rag_b_score)


if __name__ == "__main__":
    unittest.main()
