"""Run four core-module ablations with paired simulator seeds.

Probability order throughout this project is:
    [Yield/Cautious, Normal, Force/Aggressive]

Decision latency starts immediately before belief handling and ends after the
game trajectory has been converted into the low-level control command. Thus an
asynchronous LLM request is not charged to the fast path, while a synchronous
request in without_fast_slow is charged to that step.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

import highway_env  # noqa: F401  Register local environments.
from model.game.solver import GameSolver
from model.reasoning.worker import LLMReasoningWorker, runtime_setting
from model.control.stanley import StanleyController
from model.perception.filter import InteractionFilter, set_limit_speed
from model.perception.env_adapter import EnvironmentAdapter, lane_path_map_dict


DETERMINISTIC = "deterministic_type"
UNIFORM = "uniform_belief"
WITHOUT_FAST_SLOW = "without_fast_slow"
FULL = "full_model"
VARIANTS = (DETERMINISTIC, UNIFORM, WITHOUT_FAST_SLOW, FULL)
DISPLAY_NAMES = {
    DETERMINISTIC: "Deterministic Type",
    UNIFORM: "Uniform Belief",
    WITHOUT_FAST_SLOW: "w/o Fast--Slow",
    FULL: "Full Model",
}

DEFAULT_SCENARIOS = ("MergeEnv", "HighwayEnv")
SCENARIO_OBJECTIVES = {
    "HighwayEnv": "survive",
    "MergeEnv": "route",
    "AdversarialMergeEnv": "merge",
    "CutinEnv": "merge",
}
NORMAL_BELIEF = [0.0, 1.0, 0.0]
UNIFORM_BELIEF = [1.0 / 3.0] * 3


@dataclass
class EpisodeResult:
    variant: str
    method: str
    scenario: str
    seed: int
    status: str
    success: int
    collision: int
    objective: str
    objective_completed: int
    steps: int
    duration_s: float
    min_ttc_s: Optional[float]
    average_speed_ms: Optional[float]
    acceleration_variance: Optional[float]
    p95_decision_latency_ms: Optional[float]
    mean_decision_latency_ms: Optional[float]
    llm_updates: int
    llm_failures: int
    termination: str
    error: str = ""


class AblationController:
    """Apply only the belief/coordination change selected by the variant."""

    def __init__(
        self,
        variant: str,
        *,
        history_len: int,
        llm_interval_steps: int,
        model: Optional[str],
        base_url: Optional[str],
        prototypes: Optional[dict],
    ) -> None:
        self.variant = variant
        self.history_len = history_len
        self.llm_interval_steps = max(1, llm_interval_steps)
        self.worker: Optional[LLMReasoningWorker] = None
        self.last_beliefs: Dict[int, List[float]] = {}
        self.last_result_timestamp = 0.0
        self.direct_target_lane = None

        if variant in (WITHOUT_FAST_SLOW, FULL):
            self.worker = LLMReasoningWorker(
                history_len=history_len,
                model=model,
                base_url=base_url,
                behavior_prototypes=prototypes,
            )
            if FULL == variant:
                self.worker.start()

    def close(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.worker.stop()
            self.worker.join(timeout=1.0)

    def _is_update_step(self, step: int) -> bool:
        observed_frames = step + 1
        return (
            observed_frames >= self.history_len
            and (observed_frames - self.history_len) % self.llm_interval_steps == 0
        )

    @staticmethod
    def _current_vehicle_ids(others_info: dict) -> List[int]:
        return [int(vehicle_id) for vehicle_id in list(others_info.get("ids") or [])]

    def _beliefs_for_current_vehicles(self, vehicle_ids: Sequence[int]) -> Dict[int, List[float]]:
        if self.variant == DETERMINISTIC:
            return {vehicle_id: list(NORMAL_BELIEF) for vehicle_id in vehicle_ids}
        if self.variant == UNIFORM:
            return {vehicle_id: list(UNIFORM_BELIEF) for vehicle_id in vehicle_ids}
        return {
            vehicle_id: list(self.last_beliefs.get(vehicle_id, UNIFORM_BELIEF))
            for vehicle_id in vehicle_ids
        }

    @staticmethod
    def _remaining_lane_length(data_process, lane_index) -> float:
        points = data_process.lanes.get(lane_index) or []
        if len(points) < 2:
            return 0.0
        xy = np.asarray([[point.x, point.y] for point in points], dtype=float)
        ego_xy = np.asarray(data_process.ego_state[:2], dtype=float)
        nearest = int(np.argmin(np.linalg.norm(xy - ego_xy, axis=1)))
        return float(np.linalg.norm(np.diff(xy[nearest:], axis=0), axis=1).sum())

    def _direct_action_prompt(self, prompt: str, data_process) -> str:
        candidates = sorted(
            dict.fromkeys(data_process.near_target_idxes), key=lambda value: str(value)
        )
        lengths = {
            index: round(self._remaining_lane_length(data_process, index), 1)
            for index in candidates
        }
        return (
            prompt
            + "\nThis is the direct-action ablation. Also select the ego target lane."
            + f"\nCurrent lane: {data_process.curr_lane_idx}."
            + f"\nLegal candidate lanes: {candidates}."
            + f"\nApproximate remaining path length by lane in metres: {lengths}."
            + "\nBalance collision avoidance and route progress. A short ending lane may require merging."
            + '\nAdd exactly one top-level field named "_ego_target_lane" whose value is one legal lane identifier.'
            + "\nThe other top-level fields must remain the vehicle probability arrays already requested."
        )

    def _parse_direct_target(self, candidates: Sequence):
        if self.worker is None or not self.worker.last_raw_response:
            return None
        data = self.worker._extract_json(self.worker.last_raw_response)
        if not data:
            return None
        value = data.get("_ego_target_lane")
        for candidate in candidates:
            if value == candidate or str(value) == str(candidate):
                return candidate
        return None

    def update(self, step: int, snapshot: dict, data_process) -> Tuple[dict, Optional[int], bool, Optional[float], str]:
        vehicle_ids = self._current_vehicle_ids(snapshot["others_info"])
        if self.variant in (DETERMINISTIC, UNIFORM):
            return self._beliefs_for_current_vehicles(vehicle_ids), None, False, None, ""

        assert self.worker is not None
        due = self._is_update_step(step)
        llm_updated = False
        llm_latency_ms: Optional[float] = None
        error = ""

        if self.variant == FULL:
            self.worker.feed_data(snapshot, request_inference=due)
            beliefs, timestamp = self.worker.fetch_result()
            if beliefs is not None and timestamp > self.last_result_timestamp:
                self.last_beliefs.update(beliefs)
                self.last_result_timestamp = timestamp
                llm_updated = True
                if self.worker.last_latency_s is not None:
                    llm_latency_ms = self.worker.last_latency_s * 1000.0
            if due and self.worker.last_error:
                error = self.worker.last_error
        else:
            self.worker.feed_data(snapshot, request_inference=False)
            if due:
                with self.worker.lock:
                    history = list(self.worker.history_buffer)
                prompt, target_ids = self.worker._generate_prompt(history)
                if prompt:
                    candidates = sorted(
                        dict.fromkeys(data_process.near_target_idxes), key=lambda value: str(value)
                    )
                    direct_prompt = self._direct_action_prompt(prompt, data_process)
                    beliefs = self.worker.get_response(direct_prompt, target_ids)
                    if beliefs:
                        beliefs = self.worker.smooth_probabilities(beliefs)
                        self.last_beliefs.update(beliefs or {})
                        llm_updated = True
                    selected = self._parse_direct_target(candidates)
                    if selected is not None:
                        self.direct_target_lane = selected
                    if self.worker.last_latency_s is not None:
                        llm_latency_ms = self.worker.last_latency_s * 1000.0
                    error = self.worker.last_error or ""

        forced_target = None
        if self.variant == WITHOUT_FAST_SLOW:
            candidates = list(data_process.near_target_idxes)
            if self.direct_target_lane in candidates:
                forced_target = self.direct_target_lane

        beliefs = self._beliefs_for_current_vehicles(vehicle_ids)
        return beliefs, forced_target, llm_updated, llm_latency_ms, error


def parse_seed_spec(spec: str) -> List[int]:
    """Parse comma-separated integers and Python-style start:stop[:step] ranges."""
    seeds: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            values = [int(value) if value else None for value in part.split(":")]
            if len(values) not in (2, 3):
                raise ValueError(f"Invalid seed range: {part}")
            start = 0 if values[0] is None else values[0]
            stop = values[1]
            step = 1 if len(values) == 2 or values[2] is None else values[2]
            if stop is None or step == 0:
                raise ValueError(f"Invalid seed range: {part}")
            seeds.extend(range(start, stop, step))
        else:
            seeds.append(int(part))
    result = list(dict.fromkeys(seeds))
    if not result or any(seed < 0 for seed in result):
        raise ValueError("At least one non-negative seed is required")
    return result


def load_prototypes(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def instantaneous_ttc(env, cap_s: float) -> float:
    """Heading-aligned TTC to vehicles in the ego collision corridor."""
    ego = env.unwrapped.vehicle
    if ego.crashed:
        return 0.0
    heading = float(ego.heading)
    forward = np.asarray([math.cos(heading), math.sin(heading)])
    left = np.asarray([-math.sin(heading), math.cos(heading)])
    ego_velocity = float(ego.speed) * forward
    minimum = cap_s

    for other in env.unwrapped.road.vehicles:
        if other is ego:
            continue
        relative = np.asarray(other.position, dtype=float) - np.asarray(ego.position, dtype=float)
        longitudinal = float(relative @ forward)
        lateral = abs(float(relative @ left))
        corridor = 0.5 * (float(ego.WIDTH) + float(other.WIDTH)) + 0.5
        if lateral > corridor:
            continue
        other_velocity = float(other.speed) * np.asarray(
            [math.cos(float(other.heading)), math.sin(float(other.heading))]
        )
        if longitudinal >= 0:
            closing_speed = float((ego_velocity - other_velocity) @ forward)
        else:
            closing_speed = float((other_velocity - ego_velocity) @ forward)
        clearance = abs(longitudinal) - 0.5 * (float(ego.LENGTH) + float(other.LENGTH))
        if closing_speed > 0.1:
            minimum = min(minimum, max(0.0, clearance) / closing_speed)
    return float(minimum)


def build_action(av_traj, av_acceleration, data_process, planner: StanleyController) -> dict:
    trajectory = np.asarray(av_traj, dtype=float)
    count = trajectory.shape[0]
    formatted = np.zeros((count, 6), dtype=float)
    formatted[:, 0] = trajectory[:, 0]
    formatted[:, 1] = trajectory[:, 1]
    formatted[:, 2] = trajectory[:, 3]
    formatted[:, 3] = trajectory[:, 2]

    accelerations = np.asarray(av_acceleration, dtype=float).reshape(-1)
    if accelerations.size == 0:
        accelerations = np.zeros(1, dtype=float)
    used = min(accelerations.size, count)
    formatted[:used, 4] = accelerations[:used]
    if used < count:
        formatted[used:, 4] = accelerations[used - 1]
    formatted[:, 5] = np.insert(
        np.cumsum(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)),
        0,
        0.0,
    )

    x, y, heading, speed = data_process.ego_state[:4]
    planner.route = formatted[:, :2]
    steering = float(planner.act_stanley([x, y, speed, heading], plan_idx=0))
    controls = np.column_stack((formatted[:, 4], np.full(count, steering)))
    return {"trajectory": formatted, "action": controls.tolist()}


def mapped_lane(scenario: str, raw_lane_index):
    try:
        return lane_path_map_dict[scenario][raw_lane_index]
    except (KeyError, TypeError):
        return None


def run_episode(
    *,
    scenario: str,
    seed: int,
    variant: str,
    max_steps: int,
    style: str,
    history_len: int,
    llm_interval_steps: int,
    model: Optional[str],
    base_url: Optional[str],
    prototypes: Optional[dict],
    ttc_cap_s: float,
    quiet_solver: bool,
) -> Tuple[EpisodeResult, List[dict]]:
    objective = SCENARIO_OBJECTIVES.get(scenario, "survive")
    env = None
    controller = None
    step_rows: List[dict] = []
    collision = False
    objective_completed = False
    termination = "max_steps"
    error = ""

    try:
        env = gym.make(scenario)
        env.reset(seed=seed)
        data_process = EnvironmentAdapter()
        traffic_filter = InteractionFilter()
        game_solver = GameSolver(roslog=False, AV_style=style)
        planner = StanleyController()
        controller = AblationController(
            variant,
            history_len=history_len,
            llm_interval_steps=llm_interval_steps,
            model=model,
            base_url=base_url,
            prototypes=prototypes,
        )

        data_process.update_data(scenario, env)
        initial_lane = data_process.curr_lane_idx
        null_stream = open(os.devnull, "w", encoding="utf-8")
        try:
            for step in range(max_steps):
                data_process.update_data(scenario, env)
                if objective == "merge" and data_process.curr_lane_idx != initial_lane:
                    objective_completed = True

                preprocessing_context = (
                    contextlib.redirect_stdout(null_stream)
                    if quiet_solver
                    else contextlib.nullcontext()
                )
                with preprocessing_context:
                    limit_speed = set_limit_speed(data_process, game_solver)
                    others = traffic_filter.get_close_vehicles_game_lane(data_process, limit_speed)
                snapshot = {
                    "ego_state": copy.deepcopy(data_process.ego_state),
                    "others_info": copy.deepcopy(others),
                    "curr_lane_idx": data_process.curr_lane_idx,
                }

                ttc = instantaneous_ttc(env, ttc_cap_s)
                decision_started = time.perf_counter()
                beliefs, forced_target, llm_updated, llm_latency_ms, llm_error = controller.update(
                    step, snapshot, data_process
                )
                if forced_target is not None:
                    data_process.near_target_idxes = [forced_target]
                    data_process.near_permit_idxes = [forced_target]

                output_context = (
                    contextlib.redirect_stdout(null_stream)
                    if quiet_solver
                    else contextlib.nullcontext()
                )
                with output_context:
                    game_solver.inject_llm_beliefs(beliefs)
                    av_traj, av_a, other_trajectories, target_lane = game_solver._prallel_solve_game(
                        data_process=data_process,
                        others_info=others,
                    )
                    action = build_action(av_traj, av_a, data_process, planner)
                latency_ms = (time.perf_counter() - decision_started) * 1000.0

                data_process.lane_changeing = game_solver.changing
                data_process.game_pred_trajs = other_trajectories
                env.unwrapped.controlled_vehicles[0].game_traj = np.asarray(av_traj)[:, :4]
                _, _, terminated, truncated, _ = env.step(action)
                collision = bool(env.unwrapped.vehicle.crashed)
                if collision:
                    ttc = 0.0

                after_lane = mapped_lane(scenario, env.unwrapped.vehicle.lane_index)
                if objective == "merge" and after_lane is not None and after_lane != initial_lane:
                    objective_completed = True

                step_rows.append(
                    {
                        "variant": variant,
                        "method": DISPLAY_NAMES[variant],
                        "scenario": scenario,
                        "seed": seed,
                        "step": step,
                        "speed_ms": float(data_process.ego_state[3]),
                        "acceleration_ms2": float(data_process.ego_state[4]),
                        "ttc_s": float(ttc),
                        "decision_latency_ms": float(latency_ms),
                        "selected_target_lane": target_lane,
                        "llm_update": int(llm_updated),
                        "llm_latency_ms": llm_latency_ms,
                        "llm_error": llm_error,
                    }
                )

                if terminated or truncated:
                    termination = "collision" if collision else ("terminated" if terminated else "truncated")
                    break
        finally:
            null_stream.close()

        if objective == "survive":
            objective_completed = not collision
        elif objective == "route":
            objective_completed = termination == "terminated" and not collision
        success = int((not collision) and objective_completed)
        status = "ok"
    except Exception as exc:
        status = "failed"
        success = 0
        error = f"{type(exc).__name__}: {exc}"
        termination = "exception"
        if env is not None:
            collision = bool(getattr(env.unwrapped.vehicle, "crashed", False))
    finally:
        if controller is not None:
            controller.close()
        if env is not None:
            env.close()

    speeds = [row["speed_ms"] for row in step_rows]
    accelerations = [row["acceleration_ms2"] for row in step_rows]
    ttcs = [row["ttc_s"] for row in step_rows]
    latencies = [row["decision_latency_ms"] for row in step_rows]
    llm_updates = sum(row["llm_update"] for row in step_rows)
    llm_failures = sum(bool(row["llm_error"]) for row in step_rows)
    result = EpisodeResult(
        variant=variant,
        method=DISPLAY_NAMES[variant],
        scenario=scenario,
        seed=seed,
        status=status,
        success=success,
        collision=int(collision),
        objective=objective,
        objective_completed=int(objective_completed),
        steps=len(step_rows),
        duration_s=len(step_rows) * 0.1,
        min_ttc_s=min(ttcs) if ttcs else None,
        average_speed_ms=float(np.mean(speeds)) if speeds else None,
        acceleration_variance=float(np.var(accelerations)) if accelerations else None,
        p95_decision_latency_ms=float(np.percentile(latencies, 95)) if latencies else None,
        mean_decision_latency_ms=float(np.mean(latencies)) if latencies else None,
        llm_updates=llm_updates,
        llm_failures=llm_failures,
        termination=termination,
        error=error,
    )
    return result, step_rows


def aggregate_group(episodes: Sequence[dict], steps: Sequence[dict]) -> dict:
    valid = [row for row in episodes if row["status"] == "ok"]
    valid_keys = {(row["variant"], row["scenario"], int(row["seed"])) for row in valid}
    valid_steps = [
        row
        for row in steps
        if (row["variant"], row["scenario"], int(row["seed"])) in valid_keys
    ]
    if not valid:
        return {
            "episodes": 0,
            "failed_episodes": len(episodes),
            "SR_percent": None,
            "Min_TTC_s": None,
            "Average_Speed_ms": None,
            "Acceleration_Variance": None,
            "P95_Decision_Latency_ms": None,
        }
    return {
        "episodes": len(valid),
        "failed_episodes": len(episodes) - len(valid),
        "SR_percent": 100.0 * float(np.mean([int(row["success"]) for row in valid])),
        "Min_TTC_s": float(np.mean([float(row["min_ttc_s"]) for row in valid])),
        "Average_Speed_ms": float(np.mean([float(row["average_speed_ms"]) for row in valid])),
        "Acceleration_Variance": float(
            np.mean([float(row["acceleration_variance"]) for row in valid])
        ),
        "P95_Decision_Latency_ms": float(
            np.percentile([float(row["decision_latency_ms"]) for row in valid_steps], 95)
        ),
    }


def summarize(episodes: Sequence[dict], steps: Sequence[dict]) -> Tuple[List[dict], List[dict]]:
    scenario_rows = []
    overall_rows = []
    for variant in VARIANTS:
        selected = [row for row in episodes if row["variant"] == variant]
        if not selected:
            continue
        overall = {"variant": variant, "method": DISPLAY_NAMES[variant]}
        overall.update(aggregate_group(selected, steps))
        overall_rows.append(overall)
        for scenario in sorted({row["scenario"] for row in selected}):
            subset = [row for row in selected if row["scenario"] == scenario]
            step_subset = [
                row for row in steps if row["variant"] == variant and row["scenario"] == scenario
            ]
            summary = {
                "variant": variant,
                "method": DISPLAY_NAMES[variant],
                "scenario": scenario,
            }
            summary.update(aggregate_group(subset, step_subset))
            scenario_rows.append(summary)
    return overall_rows, scenario_rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def latex_value(value: Optional[float], digits: int = 2) -> str:
    return "[TBD]" if value is None else f"{float(value):.{digits}f}"


def write_latex(path: Path, summaries: Sequence[dict]) -> None:
    by_variant = {row["variant"]: row for row in summaries}
    lines = []
    for variant in VARIANTS:
        row = by_variant.get(variant)
        if row is None:
            lines.append(f"{DISPLAY_NAMES[variant]} & [TBD] & [TBD] & [TBD] & [TBD] & [TBD] \\\\")
            continue
        values = [
            latex_value(row["SR_percent"]),
            latex_value(row["Min_TTC_s"]),
            latex_value(row["Average_Speed_ms"]),
            latex_value(row["Acceleration_Variance"], 3),
            latex_value(row["P95_Decision_Latency_ms"], 1),
        ]
        lines.append(f"{DISPLAY_NAMES[variant]} & " + " & ".join(values) + " \\\\")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_args(args, variants: Sequence[str], scenarios: Sequence[str]) -> None:
    missing = [scenario for scenario in scenarios if scenario not in lane_path_map_dict]
    if missing:
        raise ValueError(f"Scenarios have no lane/path mapping: {missing}")
    if any(variant in (WITHOUT_FAST_SLOW, FULL) for variant in variants):
        if not runtime_setting("DASHSCOPE_API_KEY"):
            raise RuntimeError(
                "DASHSCOPE_API_KEY is required for without_fast_slow/full_model. "
                "Set it in the environment; do not place API keys in source code."
            )
    if args.max_steps <= 0 or args.history_len <= 0:
        raise ValueError("max_steps and history_len must be positive")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS) + ["all"])
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--seeds", default="0:10", help="For example: 0:30 or 0,3,9")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--style", default="Normal", choices=["Aggressive", "Normal", "Conservative"])
    parser.add_argument("--history-len", type=int, default=20)
    parser.add_argument("--llm-interval-steps", type=int, default=10)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--prototype-file", default=os.getenv("LLM_VEHICLE_PROTOTYPES"))
    parser.add_argument("--ttc-cap-s", type=float, default=50.0)
    parser.add_argument("--output-dir", default="simulation_results/ablation")
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the untimed one-step per-scenario JIT warm-up (not recommended for latency evaluation)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    variants = list(VARIANTS) if "all" in args.variants else list(dict.fromkeys(args.variants))
    scenarios = list(dict.fromkeys(args.scenarios))
    seeds = parse_seed_spec(args.seeds)
    validate_args(args, variants, scenarios)
    prototypes = load_prototypes(args.prototype_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = vars(args).copy()
    manifest.update(
        {
            "variants_resolved": variants,
            "scenarios_resolved": scenarios,
            "seeds_resolved": seeds,
            "metric_definition": {
                "SR": "collision-free objective completion; route means normal road-end termination; survive means reaching the evaluation horizon without collision",
                "Min_TTC": "mean episode-minimum heading-aligned corridor TTC, capped at ttc_cap_s",
                "Average_Speed": "mean of episode-average ego speeds",
                "Acceleration_Variance": "mean of within-episode population variances",
                "P95_Decision_Latency": "pooled per-step P95 from belief handling through low-level action",
            },
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    episode_rows: List[dict] = []
    step_rows: List[dict] = []
    if not args.skip_warmup:
        print("Warming up numerical kernels (excluded from all metrics)...", flush=True)
        with open(os.devnull, "w", encoding="utf-8") as null_stream:
            for scenario in scenarios:
                with contextlib.redirect_stdout(null_stream):
                    warmup, _ = run_episode(
                        scenario=scenario,
                        seed=seeds[0],
                        variant=DETERMINISTIC,
                        max_steps=1,
                        style=args.style,
                        history_len=args.history_len,
                        llm_interval_steps=args.llm_interval_steps,
                        model=None,
                        base_url=None,
                        prototypes=None,
                        ttc_cap_s=args.ttc_cap_s,
                        quiet_solver=True,
                    )
                if warmup.status != "ok":
                    raise RuntimeError(
                        f"Warm-up failed for {scenario}: {warmup.error or warmup.termination}"
                    )

    total = len(scenarios) * len(seeds) * len(variants)
    completed = 0
    for scenario in scenarios:
        for seed in seeds:
            for variant in variants:
                completed += 1
                print(f"[{completed}/{total}] {scenario} seed={seed} variant={variant}", flush=True)
                episode, episode_steps = run_episode(
                    scenario=scenario,
                    seed=seed,
                    variant=variant,
                    max_steps=args.max_steps,
                    style=args.style,
                    history_len=args.history_len,
                    llm_interval_steps=args.llm_interval_steps,
                    model=args.model,
                    base_url=args.base_url,
                    prototypes=prototypes,
                    ttc_cap_s=args.ttc_cap_s,
                    quiet_solver=not args.verbose_solver,
                )
                episode_rows.append(asdict(episode))
                step_rows.extend(episode_steps)
                write_csv(output_dir / "episodes.csv", episode_rows)
                write_csv(output_dir / "step_metrics.csv", step_rows)
                print(
                    f"  status={episode.status}, success={episode.success}, "
                    f"min_ttc={episode.min_ttc_s}, p95={episode.p95_decision_latency_ms}",
                    flush=True,
                )

    overall, by_scenario = summarize(episode_rows, step_rows)
    write_csv(output_dir / "ablation_summary.csv", overall)
    write_csv(output_dir / "scenario_summary.csv", by_scenario)
    write_latex(output_dir / "ablation_table_rows.tex", overall)
    failed = sum(row["status"] != "ok" for row in episode_rows)
    print(f"Finished: {len(episode_rows) - failed} valid, {failed} failed. Results: {output_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
