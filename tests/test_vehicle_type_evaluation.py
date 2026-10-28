import unittest

from model.reasoning.worker import LLMReasoningWorker
from experiments.vehicle_types.evaluation import (
    EvaluationRecord,
    calculate_metrics,
    synthetic_cases,
)


class ResponseParsingTests(unittest.TestCase):
    def setUp(self):
        self.worker = LLMReasoningWorker()

    def test_parses_fenced_and_named_probabilities(self):
        response = 'result:\n```json\n{"7": {"Yield": 2, "Normal": 1, "Force": 1}}\n```'
        parsed = self.worker.parse_llm_response(response, [7])
        self.assertEqual(parsed[7], [0.5, 0.25, 0.25])

    def test_invalid_probability_is_not_replaced_by_normal(self):
        parsed = self.worker.parse_llm_response('{"7": [-1, 1, 1]}', [7])
        self.assertIsNone(parsed)

    def test_prompt_contains_history_and_no_ground_truth(self):
        case = synthetic_cases(samples_per_class=1, seed=3)[0]
        prompt, target_ids = self.worker._generate_prompt(case["history"])
        self.assertEqual(len(target_ids), 1)
        self.assertIn("Vehicle summaries", prompt)
        self.assertIn("front_gap", prompt)
        self.assertIn("sampled trajectory", prompt)
        self.assertNotIn("ground_truth", prompt)
        self.assertNotIn("behavior_type", prompt)

    def test_temporal_probability_smoothing(self):
        worker = LLMReasoningWorker(temporal_alpha=0.75)
        self.assertEqual(worker.smooth_probabilities({7: [1.0, 0.0, 0.0]})[7], [1.0, 0.0, 0.0])
        fused = worker.smooth_probabilities({7: [0.0, 0.0, 1.0]})[7]
        self.assertEqual(fused, [0.75, 0.0, 0.25])


class MetricsTests(unittest.TestCase):
    def test_unparsed_predictions_count_as_incorrect(self):
        records = [
            EvaluationRecord("1", 1, "Cautious", "Cautious", 0.8, 0.1, 0.1),
            EvaluationRecord("2", 2, "Normal", None, None, None, None, error="bad JSON"),
        ]
        metrics = calculate_metrics(records)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["conditional_accuracy"], 1.0)
        self.assertEqual(metrics["confusion_matrix"]["Normal"]["Unparsed"], 1)


if __name__ == "__main__":
    unittest.main()
