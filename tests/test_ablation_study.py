import contextlib
import io
import unittest

import numpy as np

from model.game.solver import GameSolver
from experiments.ablation import (
    AblationController,
    DETERMINISTIC,
    UNIFORM,
    WITHOUT_FAST_SLOW,
    aggregate_group,
    parse_seed_spec,
)
from model.perception.filter import extend_forward


class SeedParsingTests(unittest.TestCase):
    def test_mixed_seed_spec(self):
        self.assertEqual(parse_seed_spec("0:3,5,7:10"), [0, 1, 2, 5, 7, 8, 9])

    def test_rejects_empty_seed_spec(self):
        with self.assertRaises(ValueError):
            parse_seed_spec("")


class BeliefAblationTests(unittest.TestCase):
    def make_controller(self, variant):
        return AblationController(
            variant,
            history_len=20,
            llm_interval_steps=10,
            model=None,
            base_url=None,
            prototypes=None,
        )

    def test_fixed_belief_variants_cover_every_observed_vehicle(self):
        snapshot = {"others_info": {"ids": [3, 8]}}
        deterministic = self.make_controller(DETERMINISTIC)
        uniform = self.make_controller(UNIFORM)
        self.assertEqual(
            deterministic._beliefs_for_current_vehicles([3, 8]),
            {3: [0.0, 1.0, 0.0], 8: [0.0, 1.0, 0.0]},
        )
        for probabilities in uniform._beliefs_for_current_vehicles([3, 8]).values():
            np.testing.assert_allclose(probabilities, [1 / 3, 1 / 3, 1 / 3])

    def test_game_solver_preserves_injected_probabilities(self):
        solver = GameSolver(roslog=False)
        with contextlib.redirect_stdout(io.StringIO()):
            solver.inject_llm_beliefs({7: [0.2, 0.3, 0.5]})
            solver._update_beliefs(
                {
                    "ids": [7],
                    "states": np.asarray([[0.0, 0.0, 0.0, 12.0, 0.0]]),
                }
            )
        np.testing.assert_allclose(solver.belief_dict[7], [0.2, 0.3, 0.5])

    def test_direct_action_accepts_string_lane_identifiers(self):
        controller = self.make_controller(WITHOUT_FAST_SLOW)
        controller.worker.last_raw_response = '{"_ego_target_lane": "path2"}'
        self.assertEqual(controller._parse_direct_target([0, 1, "path2"]), "path2")


class AggregationTests(unittest.TestCase):
    def test_metrics_use_valid_episodes_and_pooled_step_latency(self):
        episodes = [
            {
                "variant": DETERMINISTIC,
                "scenario": "test",
                "seed": 0,
                "status": "ok",
                "success": 1,
                "min_ttc_s": 3.0,
                "average_speed_ms": 10.0,
                "acceleration_variance": 0.2,
            },
            {
                "variant": DETERMINISTIC,
                "scenario": "test",
                "seed": 1,
                "status": "failed",
                "success": 0,
                "min_ttc_s": None,
                "average_speed_ms": None,
                "acceleration_variance": None,
            },
        ]
        steps = [
            {
                "variant": DETERMINISTIC,
                "scenario": "test",
                "seed": 0,
                "decision_latency_ms": value,
            }
            for value in (1.0, 2.0, 3.0)
        ]
        summary = aggregate_group(episodes, steps)
        self.assertEqual(summary["episodes"], 1)
        self.assertEqual(summary["failed_episodes"], 1)
        self.assertEqual(summary["SR_percent"], 100.0)
        self.assertAlmostEqual(summary["P95_Decision_Latency_ms"], 2.9)


class GeometryRobustnessTests(unittest.TestCase):
    def test_extend_forward_skips_duplicate_start_points(self):
        points = np.asarray([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        extended = extend_forward(points, ego_other_idx=2, min_length=5.0, spacing=1.0)
        self.assertTrue(np.isfinite(extended).all())
        self.assertLess(extended[0, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
