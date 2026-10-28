"""Asynchronous LLM inference for surrounding-vehicle behaviour types.

The probability order used by the game solver is always:
``[Yield/Cautious, Normal, Force/Aggressive]``.
"""

from __future__ import annotations

import copy
import json
import math
import os
import queue
import re
import statistics
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

try:  # Optional: the simulator can still start when online inference is disabled.
    from openai import OpenAI
except ImportError:  # pragma: no cover - depends on the runtime environment
    OpenAI = None

try:  # Compatibility with the package used by the original project.
    from zhipuai import ZhipuAI
except ImportError:  # pragma: no cover - depends on the runtime environment
    ZhipuAI = None


INTENT_LABELS: Tuple[str, str, str] = ("Yield", "Normal", "Force")


def runtime_setting(name: str, default=None):
    """Read process settings, then Windows user settings (never log secrets)."""
    value = os.getenv(name)
    if value:
        return value
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                return winreg.QueryValueEx(key, name)[0]
        except OSError:
            pass
    return default


class LLMReasoningWorker(threading.Thread):
    """Keep a history window and asynchronously classify the nearest vehicles."""

    def __init__(
        self,
        history_len: int = 20,
        dt: float = 0.1,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        behavior_prototypes: Optional[dict] = None,
        temporal_alpha: float = 0.7,
        timeout_s: float = 10.0,
        max_retries: int = 0,
        enable_thinking: bool = False,
        prompt_version: str = "legacy",
    ) -> None:
        super().__init__(daemon=True)
        if prompt_version not in ("legacy", "relative_evidence_v1", "relative_evidence_v2",
                                  "relative_evidence_v5", "relative_evidence_v6",
                                  "relative_evidence_v6b"):
            raise ValueError(f"Unknown prompt version: {prompt_version}")
        self.prompt_version = prompt_version
        self.history_len = history_len
        self.dt = dt
        self.model = model or runtime_setting("LLM_VEHICLE_MODEL", "qwen-turbo")
        self.api_key = api_key or runtime_setting("DASHSCOPE_API_KEY")
        self.base_url = base_url or runtime_setting(
            "LLM_VEHICLE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking
        self._client = None
        self.last_usage = None
        self.last_response_model = None
        self.last_error: Optional[str] = None
        prototype_path = os.getenv("LLM_VEHICLE_PROTOTYPES")
        if behavior_prototypes is None and prototype_path:
            try:
                with open(prototype_path, "r", encoding="utf-8") as handle:
                    behavior_prototypes = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                self.last_error = f"Could not load behaviour prototypes: {exc}"
        self.behavior_prototypes = behavior_prototypes
        self.temporal_alpha = min(max(temporal_alpha, 0.0), 1.0)
        self.smoothed_beliefs: Dict[int, List[float]] = {}

        self.history_buffer = deque(maxlen=history_len)
        self.trigger_queue: queue.Queue[bool] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.latest_beliefs: Optional[Dict[int, List[float]]] = None
        self.belief_timestamp = 0.0
        self.lock = threading.Lock()

        # Exposed for evaluation and diagnostics; no API key is stored here.
        self.last_raw_response: Optional[str] = None
        self.last_latency_s: Optional[float] = None

    @staticmethod
    def _frame_vehicle(frame: dict, vehicle_id: int) -> Optional[dict]:
        others = frame.get("others_info") or {}
        ids = list(others.get("ids") or [])
        if vehicle_id not in ids:
            return None
        idx = ids.index(vehicle_id)
        states = others.get("states")
        coords = others.get("frenet_relative_coords")
        laneids = others.get("laneids")
        if states is None or idx >= len(states):
            return None
        state = states[idx]
        if len(state) < 4:
            return None
        ds, dl = (coords[idx] if coords is not None and idx < len(coords) else (0.0, 0.0))
        lane_id = laneids[idx] if laneids is not None and idx < len(laneids) else None
        return {
            "speed": float(state[3]),
            "heading": float(state[2]),
            "reported_acceleration": float(state[4]) if len(state) > 4 else None,
            "ds": float(ds),
            "dl": float(dl),
            "lane_id": lane_id,
        }

    def _front_context(self, frame: dict, vehicle_id: int) -> Optional[dict]:
        """Find the target's nearest observed leader in the same lane."""
        target = self._frame_vehicle(frame, vehicle_id)
        if target is None or target["lane_id"] is None:
            return None
        candidates = []
        ego_state = frame.get("ego_state")
        if (
            ego_state is not None
            and len(ego_state) >= 4
            and frame.get("curr_lane_idx") == target["lane_id"]
            and -target["ds"] > 0
        ):
            candidates.append((-target["ds"], float(ego_state[3])))

        others = frame.get("others_info") or {}
        for other_id in list(others.get("ids") or []):
            if other_id == vehicle_id:
                continue
            other = self._frame_vehicle(frame, other_id)
            if other is None or other["lane_id"] != target["lane_id"]:
                continue
            gap = other["ds"] - target["ds"]
            if gap > 0:
                candidates.append((gap, other["speed"]))
        if not candidates:
            return None
        gap, leader_speed = min(candidates, key=lambda item: item[0])
        closing_speed = target["speed"] - leader_speed
        return {
            "gap": gap,
            "leader_relative_speed": leader_speed - target["speed"],
            "time_headway": gap / max(target["speed"], 0.1),
            "ttc": gap / closing_speed if closing_speed > 0.1 else None,
        }

    def _summarize_vehicle_history(self, history: Sequence[dict], vehicle_id: int) -> dict:
        samples = []
        for frame_index, frame in enumerate(history):
            sample = self._frame_vehicle(frame, vehicle_id)
            if sample is None:
                continue
            sample = dict(sample)
            sample["frame_index"] = frame_index
            sample["front"] = self._front_context(frame, vehicle_id)
            samples.append(sample)
        if not samples:
            return {}

        accelerations = []
        for previous, current in zip(samples, samples[1:]):
            elapsed = max((current["frame_index"] - previous["frame_index"]) * self.dt, self.dt)
            accelerations.append((current["speed"] - previous["speed"]) / elapsed)
        duration = max((samples[-1]["frame_index"] - samples[0]["frame_index"]) * self.dt, self.dt)
        lane_changes = sum(a["lane_id"] != b["lane_id"] for a, b in zip(samples, samples[1:]))
        front_samples = [sample["front"] for sample in samples if sample["front"] is not None]
        finite_ttcs = [front["ttc"] for front in front_samples if front["ttc"] is not None]
        headways = [front["time_headway"] for front in front_samples]
        gaps = [front["gap"] for front in front_samples]

        sample_indices = sorted({0, len(samples) // 4, len(samples) // 2, 3 * len(samples) // 4, len(samples) - 1})
        trajectory = []
        for idx in sample_indices:
            sample = samples[idx]
            trajectory.append(
                f"t={sample['frame_index'] * self.dt:.1f}:v={sample['speed']:.1f},"
                f"dx={sample['ds']:.1f},dy={sample['dl']:.1f},lane={sample['lane_id']}"
            )
        return {
            "observed_frames": len(samples),
            "duration": duration,
            "speed_start": samples[0]["speed"],
            "speed_end": samples[-1]["speed"],
            "speed_mean": statistics.fmean(sample["speed"] for sample in samples),
            "speed_max": max(sample["speed"] for sample in samples),
            "speed_slope": (samples[-1]["speed"] - samples[0]["speed"]) / duration,
            "acceleration_mean": statistics.fmean(accelerations) if accelerations else 0.0,
            "acceleration_min": min(accelerations) if accelerations else 0.0,
            "acceleration_max": max(accelerations) if accelerations else 0.0,
            "lane_changes": lane_changes,
            "front_observation_ratio": len(front_samples) / len(samples),
            "front_gap_mean": statistics.fmean(gaps) if gaps else None,
            "front_gap_min": min(gaps) if gaps else None,
            "headway_mean": statistics.fmean(headways) if headways else None,
            "headway_min": min(headways) if headways else None,
            "ttc_min": min(finite_ttcs) if finite_ttcs else None,
            "trajectory": trajectory,
        }

    @staticmethod
    def _format_optional(value: Optional[float], suffix: str = "") -> str:
        return "NA" if value is None else f"{value:.2f}{suffix}"

    def _generate_prompt(self, history: Sequence[dict]) -> Tuple[Optional[str], List[int]]:
        """Build a readable, deterministic prompt from a trajectory history."""
        if self.prompt_version == "relative_evidence_v1":
            return self._generate_prompt_relative(history)
        if self.prompt_version == "relative_evidence_v2":
            return self._generate_prompt_relative(history, decisive_gap_rate=True)
        if self.prompt_version == "relative_evidence_v5":
            return self._generate_prompt_relative(history, decisive_gap_rate=True,
                                                  binding_rule=True,
                                                  operational_definitions=True)
        if self.prompt_version == "relative_evidence_v6":
            return self._generate_prompt_v6(history)
        if self.prompt_version == "relative_evidence_v6b":
            return self._generate_prompt_v6(history, speed_offset_is_evidence=True)
        return self._generate_prompt_legacy(history)

    def _generate_prompt_legacy(self, history: Sequence[dict]) -> Tuple[Optional[str], List[int]]:
        """Original absolute-prototype prompt (kept for reproducible old runs)."""
        if not history:
            return None, []

        latest = history[-1]
        ego_state = latest.get("ego_state")
        others = latest.get("others_info") or {}
        ids = list(others.get("ids") or [])
        if ego_state is None or len(ego_state) < 4 or not ids:
            return None, []

        current = []
        for vehicle_id in ids:
            item = self._frame_vehicle(latest, vehicle_id)
            if item is not None:
                item["id"] = vehicle_id
                item["distance"] = math.hypot(item["ds"], item["dl"])
                current.append(item)
        current.sort(key=lambda item: item["distance"])
        targets = current[:3]
        target_ids = [item["id"] for item in targets]
        if not target_ids:
            return None, []

        ego_speed = float(ego_state[3])
        curr_lane_idx = latest.get("curr_lane_idx")
        lines = [
            "You classify persistent driving behaviour from trajectory evidence.",
            "For every vehicle, estimate probabilities in this exact order:",
            "[Yield, Normal, Force].",
            "Simulator behaviour prototypes (soft evidence, not hard thresholds):",
            "- Yield/Cautious: desired speed about 25.0 m/s, desired following headway about 2.0 s; conservative acceleration and larger following gaps.",
            "- Normal: desired speed about 30.6 m/s, desired following headway about 1.5 s; balanced acceleration and gaps.",
            "- Force/Aggressive: desired speed about 33.3 m/s, desired following headway about 1.0 s; stronger acceleration and smaller following gaps.",
            "Traffic may prevent desired behaviour, so combine all evidence and do not classify from one value alone.",
            "Use motion evidence only; vehicle IDs contain no type information.",
            f"Sampling interval: {self.dt:.3f} s; ego current speed: {ego_speed:.2f} m/s.",
            "front_gap/headway/TTC are measured from the target vehicle to its own observed leader.",
            "NA means that no leader or meaningful closing TTC was observable.",
            "Vehicle summaries:",
        ]

        if self.behavior_prototypes:
            lines.insert(-1, "Scene-calibrated reference fingerprints learned from separate labelled seeds:")
            classes = self.behavior_prototypes.get("classes", self.behavior_prototypes)
            for label in ("Cautious", "Normal", "Aggressive"):
                values = classes.get(label, {})
                fields = []
                for key in (
                    "speed_mean",
                    "speed_end",
                    "speed_slope",
                    "acceleration_mean",
                    "headway_min",
                    "ttc_min",
                ):
                    stat = values.get(key)
                    if isinstance(stat, dict) and stat.get("mean") is not None:
                        fields.append(f"{key}={stat['mean']:.2f}±{stat.get('std', 0.0):.2f}")
                lines.insert(-1, f"- observed {label}: " + ", ".join(fields))
            lines.insert(-1, "Compare patterns jointly; overlapping reference ranges are expected.")

        for item in targets:
            vehicle_id = item["id"]
            summary = self._summarize_vehicle_history(history, vehicle_id)
            lines.extend(
                [
                    f"- id={vehicle_id}: frames={summary['observed_frames']},duration={summary['duration']:.1f}s,"
                    f" speed(start/mean/end/max)={summary['speed_start']:.1f}/{summary['speed_mean']:.1f}/"
                    f"{summary['speed_end']:.1f}/{summary['speed_max']:.1f}m/s,"
                    f" speed_slope={summary['speed_slope']:+.2f}m/s2,"
                    f" accel(mean/min/max)={summary['acceleration_mean']:+.2f}/"
                    f"{summary['acceleration_min']:+.2f}/{summary['acceleration_max']:+.2f}m/s2,"
                    f" lane_changes={summary['lane_changes']}",
                    f"  leader_visible={summary['front_observation_ratio']:.0%},"
                    f" front_gap(mean/min)={self._format_optional(summary['front_gap_mean'], 'm')}/"
                    f"{self._format_optional(summary['front_gap_min'], 'm')},"
                    f" headway(mean/min)={self._format_optional(summary['headway_mean'], 's')}/"
                    f"{self._format_optional(summary['headway_min'], 's')},"
                    f" min_TTC={self._format_optional(summary['ttc_min'], 's')}",
                    "  sampled trajectory: " + " | ".join(summary["trajectory"]),
                ]
            )

        example = {str(vehicle_id): ["p_yield", "p_normal", "p_force"] for vehicle_id in target_ids}
        lines.extend(
            [
                "Return one JSON object only. Each array must contain three non-negative numbers summing to 1.",
                json.dumps(example, ensure_ascii=False),
            ]
        )
        return "\n".join(lines), target_ids

    def _summarize_vehicle_history_relative(
        self, history: Sequence[dict], vehicle_id: int
    ) -> Optional[dict]:
        """Summarize a target's motion relative to the ego and to its leader.

        ``dx`` (frenet longitudinal coordinate relative to ego) is the key
        signal: negative values mean the target is behind the ego.  Absolute
        speed is reported but the prompt treats it as weak evidence.
        """
        samples = []
        for frame_index, frame in enumerate(history):
            sample = self._frame_vehicle(frame, vehicle_id)
            if sample is None:
                continue
            ego_state = frame.get("ego_state")
            ego_speed = float(ego_state[3]) if ego_state is not None and len(ego_state) > 3 else None
            sample = dict(sample)
            sample["frame_index"] = frame_index
            sample["ego_speed"] = ego_speed
            sample["front"] = self._front_context(frame, vehicle_id)
            # Same-lane stream speed gives a position-invariant reference that
            # does not depend on the (possibly biased) reported ego speed.
            frame_others = frame.get("others_info") or {}
            same_lane_speeds, all_other_speeds = [], []
            for other_id in list(frame_others.get("ids") or []):
                if other_id == vehicle_id:
                    continue
                other = self._frame_vehicle(frame, other_id)
                if other is None:
                    continue
                all_other_speeds.append(other["speed"])
                if other["lane_id"] == sample["lane_id"]:
                    same_lane_speeds.append(other["speed"])
            stream_speed = (statistics.fmean(same_lane_speeds) if same_lane_speeds
                            else (statistics.fmean(all_other_speeds) if all_other_speeds
                                  else None))
            sample["stream_speed"] = stream_speed
            sample["stream_relative_speed"] = (sample["speed"] - stream_speed
                                               if stream_speed is not None else None)
            samples.append(sample)
        if not samples:
            return None

        accelerations = []
        for previous, current in zip(samples, samples[1:]):
            elapsed = max((current["frame_index"] - previous["frame_index"]) * self.dt, self.dt)
            accelerations.append((current["speed"] - previous["speed"]) / elapsed)
        duration = max((samples[-1]["frame_index"] - samples[0]["frame_index"]) * self.dt, self.dt)
        dx_start, dx_end = samples[0]["ds"], samples[-1]["ds"]
        abs_dx_start, abs_dx_end = abs(dx_start), abs(dx_end)
        gap_change = abs_dx_end - abs_dx_start
        closing_speed = (abs_dx_start - abs_dx_end) / duration
        relative_speeds = [
            sample["speed"] - sample["ego_speed"]
            for sample in samples
            if sample["ego_speed"] is not None
        ]
        front_samples = [sample["front"] for sample in samples if sample["front"] is not None]
        finite_ttcs = [front["ttc"] for front in front_samples if front["ttc"] is not None]
        headways = [front["time_headway"] for front in front_samples]
        gaps = [front["gap"] for front in front_samples]
        leader_relative_speeds = [front["leader_relative_speed"] for front in front_samples]
        speed_values = [sample["speed"] for sample in samples]
        stream_speeds = [sample["stream_speed"] for sample in samples
                         if sample.get("stream_speed") is not None]
        stream_relative_speeds = [sample["stream_relative_speed"] for sample in samples
                                  if sample.get("stream_relative_speed") is not None]

        sample_indices = sorted({0, len(samples) // 4, len(samples) // 2,
                                 3 * len(samples) // 4, len(samples) - 1})
        trajectory = []
        for idx in sample_indices:
            sample = samples[idx]
            trajectory.append(
                f"t={sample['frame_index'] * self.dt:.1f}:v={sample['speed']:.1f},"
                f"dx={sample['ds']:.1f},dy={sample['dl']:.1f},lane={sample['lane_id']}"
            )
        return {
            "observed_frames": len(samples),
            "duration": duration,
            "speed_start": samples[0]["speed"],
            "speed_mean": statistics.fmean(sample["speed"] for sample in samples),
            "speed_end": samples[-1]["speed"],
            "speed_max": max(sample["speed"] for sample in samples),
            "speed_std": statistics.pstdev(speed_values) if len(speed_values) > 1 else 0.0,
            "speed_slope": (samples[-1]["speed"] - samples[0]["speed"]) / duration,
            "acceleration_mean": statistics.fmean(accelerations) if accelerations else 0.0,
            "acceleration_min": min(accelerations) if accelerations else 0.0,
            "acceleration_max": max(accelerations) if accelerations else 0.0,
            "hard_braking_frames": sum(a < -1.5 for a in accelerations),
            "hard_acceleration_frames": sum(a > 1.5 for a in accelerations),
            "lane_changes": sum(a["lane_id"] != b["lane_id"] for a, b in zip(samples, samples[1:])),
            "dx_start": dx_start,
            "dx_end": dx_end,
            "dx_rate": (dx_end - dx_start) / duration,
            "gap_change": gap_change,
            "gap_rate": gap_change / duration,
            "gap_trend": "opening" if gap_change > 0.5 else ("closing" if gap_change < -0.5 else "stable"),
            "closing_speed": max(0.0, closing_speed),
            "relative_speed_mean": statistics.fmean(relative_speeds) if relative_speeds else None,
            "front_observation_ratio": len(front_samples) / len(samples),
            "front_gap_mean": statistics.fmean(gaps) if gaps else None,
            "front_gap_min": min(gaps) if gaps else None,
            "headway_mean": statistics.fmean(headways) if headways else None,
            "headway_min": min(headways) if headways else None,
            "leader_relative_speed_mean": (statistics.fmean(leader_relative_speeds)
                                           if leader_relative_speeds else None),
            "stream_speed_mean": (statistics.fmean(stream_speeds)
                                  if stream_speeds else None),
            "stream_relative_speed_mean": (statistics.fmean(stream_relative_speeds)
                                           if stream_relative_speeds else None),
            "ttc_min": min(finite_ttcs) if finite_ttcs else None,
            "trajectory": trajectory,
        }

    def _leader_line(self, summary: dict) -> str:
        """Describe the target's own leader, or mark it neutral when distant."""
        headway = summary["headway_mean"]
        if headway is None or headway >= 4.0:
            return ("  leader=distant or not observed (headway >= 4 s or NA): "
                    "neutral, do not infer a type from it")
        return (f"  leader_visible={summary['front_observation_ratio']:.0%},"
                f" front_gap(mean/min)={self._format_optional(summary['front_gap_mean'], 'm')}/"
                f"{self._format_optional(summary['front_gap_min'], 'm')},"
                f" headway(mean/min)={self._format_optional(summary['headway_mean'], 's')}/"
                f"{self._format_optional(summary['headway_min'], 's')},"
                f" min_TTC={self._format_optional(summary['ttc_min'], 's')}")

    def _generate_prompt_relative(
        self, history: Sequence[dict], decisive_gap_rate: bool = False,
        binding_rule: bool = False, operational_definitions: bool = False,
    ) -> Tuple[Optional[str], List[int]]:
        """Prompt that normalizes away absolute speed and emphasizes gap dynamics.

        The legacy prompt described free-flow prototypes at 25--33 m/s.  In the
        low-speed merge benchmark the whole stream travels at 4--12 m/s, so the
        model collapsed every vehicle to Cautious.  This version describes the
        same three behaviours in ego-relative and leader-relative terms.

        ``decisive_gap_rate`` (v2) makes the observed geometric gap rate the
        primary signal and asks for calibrated confidence on unambiguous
        closing/opening evidence.  v1 remains unchanged for reproducibility.
        """
        if not history:
            return None, []

        latest = history[-1]
        ego_state = latest.get("ego_state")
        others = latest.get("others_info") or {}
        ids = list(others.get("ids") or [])
        if ego_state is None or len(ego_state) < 4 or not ids:
            return None, []

        current = []
        for vehicle_id in ids:
            item = self._frame_vehicle(latest, vehicle_id)
            if item is not None:
                item["id"] = vehicle_id
                item["distance"] = math.hypot(item["ds"], item["dl"])
                current.append(item)
        current.sort(key=lambda item: item["distance"])
        targets = current[:3]
        target_ids = [item["id"] for item in targets]
        if not target_ids:
            return None, []

        ego_speed = float(ego_state[3])
        lines = [
            "You classify persistent driving behaviour from relative motion evidence.",
            "For every vehicle, estimate probabilities in this exact order:",
            "[Yield, Normal, Force].",
            "",
            "Important: absolute speed is weak evidence. In congested or low-speed",
            "traffic every vehicle is slow, so a low speed alone does not mean",
            "Cautious. Judge how the vehicle moves relative to the ego and to its own",
            "leader, and how its gap evolves over the whole window.",
            "",
            "Reference patterns (soft, overlapping ordinal evidence, not thresholds):",
            "- Yield/Cautious: keeps a larger time headway, accelerates and brakes",
            "  gently, and tends to hold back or open a gap. If it is behind the ego,",
            "  the gap closes slowly or stays; if it is ahead, the ego closes slowly.",
            "- Normal: moderate headway and acceleration, and a roughly steady gap.",
            "- Force/Aggressive: keeps a shorter time headway, uses stronger",
            "  acceleration or late braking, and presses the gap. If it is behind the",
            "  ego, the gap closes quickly; if it is ahead, it pulls away or brakes late.",
            "The three classes are equally frequent in this benchmark. Do not default",
            "to Cautious or Normal when the evidence is weak; stay near the balanced",
            "prior [0.33, 0.33, 0.33] unless the trajectory supports a class.",
            "",
        ]
        if decisive_gap_rate:
            lines.extend(
                [
                    "Primary evidence: the ego-relative gap rate, in m/s.",
                    "- gap_rate is how fast the gap between the ego and this vehicle",
                    "  changes. Positive means the gap is opening; negative means it",
                    "  is closing.",
                    "- For a target behind the ego (dx < 0), a gap_rate of -1.5 m/s or",
                    "  lower means it is closing on the ego quickly: strong Force",
                    "  evidence. A gap_rate of +1.5 m/s or higher means it is dropping",
                    "  back: Yield evidence. A gap_rate between about -0.5 and +0.5",
                    "  m/s is steady: Normal evidence.",
                    "- Behind the ego, sustained closing is the decisive Force signal.",
                    "  The vehicle need not brake late or accelerate hard: a moderate",
                    "  speed with a persistent close rate is still Force.",
                    "- Do not classify from absolute speed or acceleration magnitude",
                    "  alone; gap_rate overrides both when they disagree.",
                    "- The geometric gap_rate is more reliable than relative_speed_mean.",
                    "  The reported ego speed can be biased (the executed trajectory",
                    "  can differ from the reported speed); when the two disagree, use",
                    "  gap_rate and the sampled dx values, not the speed difference.",
                    "- A large front gap or headway to a distant leader (above about",
                    "  4 s or 50 m) is neutral. It is not evidence for Cautious.",
                    "",
                    "Use this checklist for each vehicle:",
                    "1. Read gap_rate from the dx line first; it is the primary signal.",
                    "2. Confirm it against the sampled trajectory dx values.",
                    "3. Use headway and TTC only when the leader is close enough to",
                    "   matter (headway below about 4 s).",
                    "4. Use acceleration and speed slope only to support gap_rate.",
                    "",
                    "Calibration: when the gap_rate is decisive (for a target behind",
                    "the ego, at most -1.5 m/s or at least +1.5 m/s over the whole",
                    "window), do not hedge: put at least 0.7 on the corresponding",
                    "class. Weak or mixed evidence should stay near the prior.",
                    "",
                ]
            )
            if binding_rule:
                lines.extend(
                    [
                        "Binding rule for this benchmark (apply it per vehicle, only",
                        "when that vehicle is behind the ego, dx_signed < 0):",
                        "- gap_rate <= -1.0 m/s: the highest probability must be Force,",
                        "  with p_force at least 0.6.",
                        "- gap_rate >= +1.0 m/s: the highest probability must be Yield,",
                        "  with p_yield at least 0.6.",
                        "- -0.5 <= gap_rate <= +0.5 m/s: the highest probability must be",
                        "  Normal.",
                        "- A closing target is Force even if it accelerates gently and",
                        "  keeps a large headway to its own distant leader.",
                        "",
                    ]
                )
            if operational_definitions:
                lines.extend(
                    [
                        "Operational definitions for this benchmark (these override",
                        "the free-flow prototypes above):",
                        "For a vehicle behind the ego (dx_signed < 0), the signed",
                        "gap_rate is the label definition:",
                        "- gap_rate <= -1.0 m/s: Force.",
                        "- -0.5 < gap_rate < +0.5 m/s: Normal.",
                        "- gap_rate >= +1.0 m/s: Yield.",
                        "Apply the band to the computed gap_rate; do not substitute a",
                        "free-flow prototype or a prior.",
                        "",
                        "Canonical examples behind the ego over a 2 s window:",
                        "- 25.0 m -> 21.0 m (gap_rate -2.0 m/s): Force-dominant",
                        "  [0.1, 0.1, 0.8].",
                        "- 25.0 m -> 25.8 m (gap_rate +0.4 m/s): Normal-dominant",
                        "  [0.1, 0.8, 0.1]. A small drift in either direction stays",
                        "  Normal.",
                        "- 25.0 m -> 29.0 m (gap_rate +2.0 m/s): Yield-dominant",
                        "  [0.8, 0.1, 0.1].",
                        "",
                    ]
                )
        else:
            lines.extend(
                [
                    "Use this checklist for each vehicle:",
                    "1. Compare its relative speed (target minus ego) with the ego.",
                    "2. Read the ego-relative longitudinal gap dx: negative means behind,",
                    "   positive means ahead. Gap trend opening/closing is more informative",
                    "   than its absolute value.",
                    "3. If a leader is visible, use headway and minimum TTC to judge how",
                    "   closely it follows.",
                    "4. Use acceleration magnitude and speed slope to separate gentle from",
                    "   forceful longitudinal control.",
                    "",
                ]
            )
        lines.extend(
            [
                f"Sampling interval: {self.dt:.3f} s; ego current speed: {ego_speed:.2f} m/s.",
                "front_gap/headway/TTC are measured from the target vehicle to its own",
                "observed leader. NA means that no leader or meaningful closing TTC was",
                "observable.",
                "",
                "Vehicle summaries:",
            ]
        )

        for item in targets:
            vehicle_id = item["id"]
            summary = self._summarize_vehicle_history_relative(history, vehicle_id)
            if summary is None:
                continue
            relative_speed = summary["relative_speed_mean"]
            relative_text = "NA" if relative_speed is None else f"{relative_speed:+.2f}m/s"
            if decisive_gap_rate:
                geometry_lines = [
                    f"  gap_to_ego(start/end/change)={abs(summary['dx_start']):.1f}/"
                    f"{abs(summary['dx_end']):.1f}m/{summary['gap_change']:+.1f}m"
                    f"({summary['gap_trend']} over {summary['duration']:.1f}s),"
                    f" gap_rate={summary['gap_rate']:+.2f}m/s,"
                    f" dx_signed(start/end)={summary['dx_start']:+.1f}/"
                    f"{summary['dx_end']:+.1f}m (negative=behind),"
                    f" relative_speed_mean={relative_text},"
                    f" closing_speed={summary['closing_speed']:.1f}m/s",
                    self._leader_line(summary),
                ]
            else:
                geometry_lines = [
                    f"  ego_relative_dx(start/end/trend)={summary['dx_start']:+.1f}/"
                    f"{summary['dx_end']:+.1f}m/{summary['gap_trend']}"
                    f"({summary['gap_change']:+.1f}m over {summary['duration']:.1f}s),"
                    f" relative_speed_mean={relative_text},"
                    f" closing_speed={summary['closing_speed']:.1f}m/s",
                    f"  leader_visible={summary['front_observation_ratio']:.0%},"
                    f" front_gap(mean/min)={self._format_optional(summary['front_gap_mean'], 'm')}/"
                    f"{self._format_optional(summary['front_gap_min'], 'm')},"
                    f" headway(mean/min)={self._format_optional(summary['headway_mean'], 's')}/"
                    f"{self._format_optional(summary['headway_min'], 's')},"
                    f" min_TTC={self._format_optional(summary['ttc_min'], 's')}",
                ]
            lines.extend(
                [
                    f"- id={vehicle_id}: frames={summary['observed_frames']},"
                    f"duration={summary['duration']:.1f}s,"
                    f" speed(start/mean/end/max)={summary['speed_start']:.1f}/"
                    f"{summary['speed_mean']:.1f}/{summary['speed_end']:.1f}/"
                    f"{summary['speed_max']:.1f}m/s,"
                    f" speed_slope={summary['speed_slope']:+.2f}m/s2,"
                    f" accel(mean/min/max)={summary['acceleration_mean']:+.2f}/"
                    f"{summary['acceleration_min']:+.2f}/{summary['acceleration_max']:+.2f}m/s2,"
                    f" lane_changes={summary['lane_changes']}",
                    *geometry_lines,
                    "  sampled trajectory: " + " | ".join(summary["trajectory"]),
                ]
            )

        example = {str(vehicle_id): ["p_yield", "p_normal", "p_force"] for vehicle_id in target_ids}
        lines.extend(
            [
                "Return one JSON object only. Each array must contain three",
                "non-negative numbers summing to 1. Probabilities are calibrated:",
                "0.9 means strong evidence, while weak evidence stays near 0.33.",
                json.dumps(example, ensure_ascii=False),
            ]
        )
        return "\n".join(lines), target_ids

    def _generate_prompt_v6(self, history: Sequence[dict],
                            speed_offset_is_evidence: bool = False) -> Tuple[Optional[str], List[int]]:
        """Position-invariant style prompt (v6).

        v5 made the ego-relative gap rate the decisive signal and applied its
        binding rule only to vehicles behind the ego.  That is an interaction
        intent, not the vehicle's intrinsic style, and it left ahead/side
        background vehicles with no usable rule.  v6 asks for persistent style
        from the target's own leader-relative behaviour and treats ego-relative
        motion as weak context only.

        v6b (speed_offset_is_evidence=True) additionally treats a sustained
        speed offset from both the ego and the same-lane stream as evidence
        when the target has no observable leader.  v6's constant-speed rule
        otherwise makes exactly the type-scripted key vehicle abstain.
        """
        if not history:
            return None, []

        latest = history[-1]
        ego_state = latest.get("ego_state")
        others = latest.get("others_info") or {}
        ids = list(others.get("ids") or [])
        if ego_state is None or len(ego_state) < 4 or not ids:
            return None, []

        current = []
        for vehicle_id in ids:
            item = self._frame_vehicle(latest, vehicle_id)
            if item is not None:
                item["id"] = vehicle_id
                item["distance"] = math.hypot(item["ds"], item["dl"])
                current.append(item)
        current.sort(key=lambda item: item["distance"])
        targets = current[:3]
        target_ids = [item["id"] for item in targets]
        if not target_ids:
            return None, []

        ego_speed = float(ego_state[3])
        if speed_offset_is_evidence:
            leaderless_rule = [
                "- If its own leader is not visible, use the relative_speed bands",
                "  below, which compare the target with the ego. The same-lane",
                "  stream is supporting context only. Also use acceleration",
                "  magnitude, speed oscillations, late braking, and lane changes.",
                "  Stay near the prior only when all of these are weak.",
            ]
            secondary_intro = [
                "- First compare the target's speed with the ego's reported speed at",
                "  the same frame (relative_speed). For a vehicle behind with nearly",
                "  constant speed and no usable leader evidence, a sustained offset",
                "  is evidence, not a reason to abstain:",
            ]
            sign_rule = [
                "  Sign convention: relative_speed = target speed minus ego speed;",
                "  positive means the target is faster than the ego. stream_relative",
                "  = target speed minus same-lane stream mean; positive means the",
                "  target is faster than the stream.",
            ]
            secondary_bands = [
                "  * sustained relative_speed below about +1.0 m/s (target no",
                "    faster than the ego) leans Cautious;",
                "  * relative_speed between +1.0 and +3.0 m/s leans Normal;",
                "  * sustained relative_speed above about +3.0 m/s, especially with a",
                "    short gap, is strong Aggressive evidence.",
                "  * Decision rule for a leaderless vehicle: use relative_speed to",
                "    the ego, not stream_relative. <= +1.0 -> Cautious; +1.0 to",
                "    +3.0 -> Normal; > +3.0 -> Aggressive. If relative_speed is",
                "    within +/-0.5 m/s and stream_relative is also near 0, evidence",
                "    is weak: stay near uniform. This rule takes precedence over",
                "    gap direction and over stream_relative when they disagree; the",
                "    ego's executed speed can differ from its reported speed.",
                "  * Calibration example with ego reported 7.0 m/s and a leaderless",
                "    target: target 6.0 m/s -> relative -1.0 -> Cautious; target",
                "    8.5 m/s -> relative +1.5 -> Normal; target 10.5 m/s ->",
                "    relative +3.5 -> Aggressive. Do not override these with the",
                "    gap direction.",
            ]
            stream_rule = [
                "- Then compare the target's speed with the same-lane stream",
                "  (stream_relative_speed). This is supporting evidence only and must",
                "  not replace relative_speed in the bands above: the local stream",
                "  may itself be braking. Use it only to break near-ties, not to",
                "  override a clear relative_speed band.",
            ]
            no_leader_counterexample = [
                "- A vehicle with no visible leader and a constant speed is weak",
                "  evidence only if its speed is close to both the ego and the stream;",
                "  a large sustained offset is informative (below -> Cautious, above",
                "  -> Aggressive).",
            ]
            constant_speed_rule = [
                "If a vehicle has near-constant speed (speed_std below about 0.1 m/s),",
                "no hard acceleration or braking, no lane changes, and unavailable",
                "leader evidence, do not automatically abstain. When its speed is",
                "persistently offset from both the ego and the same-lane stream, use",
                "the offset: about +1.0 m/s or less -> Cautious; about +3.0 m/s or",
                "more -> Aggressive; otherwise Normal. Do not choose Aggressive just",
                "because another vehicle in the scene is aggressive.",
            ]
            geometry_rule = [
                "- For a leaderless vehicle the relative_speed band above decides the",
                "  label. Do not let the sampled dx trajectory or dx_rate override",
                "  it: the ego's executed speed can differ from its reported speed,",
                "  so geometric fall-back can contradict the reported relative speed.",
                "  If they disagree, keep the band implied by relative_speed.",
            ]
            step5_rule = [
                "5. Use ego-relative motion as secondary evidence, except that for a",
                "   leaderless vehicle the relative_speed band above is the primary",
                "   cue.",
            ]
        else:
            leaderless_rule = [
                "- If its own leader is not visible, use acceleration magnitude, speed",
                "  oscillations, late braking, and lane changes. Stay near the prior when",
                "  these are weak.",
            ]
            secondary_intro = [
                "- First compare the target's speed with the ego's reported speed at the",
                "  same frame (relative_speed). For a vehicle behind with nearly",
                "  constant speed and no usable leader evidence:",
            ]
            sign_rule = []
            secondary_bands = [
                "  * sustained relative_speed below -0.5 m/s (target slower than the",
                "    ego) and a smooth fall-back leans Cautious;",
                "  * relative_speed near 0 (within about +/-0.5 m/s) leans Normal;",
                "  * sustained relative_speed above +0.5 m/s leans Normal or Aggressive;",
                "    above about +2 m/s with a short gap leans Aggressive.",
            ]
            stream_rule = [
                "- Then compare the target's speed with the same-lane stream",
                "  (stream_relative_speed). This is supporting evidence only: the local",
                "  stream may itself be braking. A vehicle clearly above a steady stream",
                "  leans Aggressive; one clearly below a steady stream leans Cautious.",
            ]
            no_leader_counterexample = [
                "- A vehicle with no visible leader and a constant speed is weak evidence;",
                "  stay near [0.33, 0.33, 0.33].",
            ]
            constant_speed_rule = [
                "If a vehicle has near-constant speed (speed_std below about 0.1 m/s), no",
                "hard acceleration or braking, no lane changes, and unavailable leader",
                "evidence, its evidence is weak: stay near [0.33, 0.33, 0.33]. Do not",
                "choose Aggressive just because another vehicle in the scene is aggressive.",
            ]
            geometry_rule = [
                "- Use dx_rate for the direction of relative motion. The reported ego",
                "  speed can be biased; when it disagrees with dx_rate or the sampled dx",
                "  values, trust the geometry for direction, but keep the speed magnitude",
                "  as a level cue.",
                "- Falling back relative to a faster ego is not Cautious evidence if the",
                "  target is at or above the stream and its relative_speed is positive.",
            ]
            step5_rule = [
                "5. Use ego-relative motion only as secondary evidence, never as the",
                "   primary label.",
            ]
        lines = [
            "You infer the intrinsic driving style of each observed vehicle from its",
            "trajectory. For every vehicle, output probabilities in this exact order:",
            "[Cautious, Normal, Aggressive].",
            "",
            "The label is the vehicle's persistent style, not its momentary intent",
            "toward the ego.",
            "- Cautious: conservative longitudinal control; keeps a long time headway",
            "  to its own leader, accelerates and brakes gently, avoids late braking,",
            "  and rarely changes lanes.",
            "- Normal: moderate time headway and acceleration; follows the traffic",
            "  stream steadily.",
            "- Aggressive: short time headway to its own leader, strong acceleration",
            "  or late braking, presses into gaps, and changes lanes to gain position.",
            "",
            "Position-invariant principle:",
            "- The ego is only one reference. Classify a vehicle ahead, behind, or",
            "  alongside the ego with the same style definition.",
            "- Do not use the ego-relative gap or gap rate as the primary signal. A",
            "  vehicle closing on the ego may simply be faster; that is not",
            "  aggression. A vehicle the ego is catching up to may simply be slower;",
            "  that is not caution.",
            "- Primary evidence is how the vehicle follows its own leader: time",
            "  headway, minimum TTC, and whether it holds a steady gap or repeatedly",
            "  closes and brakes late.",
            *leaderless_rule,
            "- A leader farther than about 50 m or a headway above about 4 s is",
            "  effectively unobserved. It is not evidence for Cautious.",
            "",
            "Secondary evidence when leader_evidence=unavailable:",
            *secondary_intro,
            *sign_rule,
            *secondary_bands,
            *stream_rule,
            *geometry_rule,
            "",
            "Soft reference bands (overlapping, not hard rules; use the whole window):",
            "- Time headway to its own leader (only when the leader is close, i.e.",
            "  headway below about 4 s and gap below about 50 m): >= 2.0 s leans",
            "  Cautious; 1.2-2.0 s leans Normal; <= 1.2 s leans Aggressive.",
            "- Minimum TTC to its own leader: >= 4 s leans Cautious; 2-4 s leans",
            "  Normal; < 2 s leans Aggressive.",
            "- Sustained acceleration: |a| <= 0.5 m/s2 is gentle; 0.5-1.5 is moderate;",
            "  > 1.5 is strong. Strong positive acceleration or late braking leans",
            "  Aggressive.",
            "- Speed variability and repeated close-open cycles lean Aggressive; steady",
            "  speed leans Normal.",
            "- More than one lane change in the window, or a lane change toward a",
            "  smaller gap, leans Aggressive.",
            "- Absolute speed is weak evidence in congested or heterogeneous",
            "  traffic. However, a large, sustained speed offset from both the ego",
            "  and the same-lane stream is informative; do not ignore it.",
            "  Do not label a slow vehicle Cautious if it keeps a short headway and",
            "  brakes late.",
            "",
            "Counterexamples:",
            "- A vehicle behind the ego that closes the gap because the ego is slower,",
            "  while keeping a normal headway to its own leader, is Normal, not",
            "  Aggressive.",
            "- A vehicle ahead that the ego approaches because it is slower, while",
            "  keeping a long headway to its own leader and braking gently, is",
            "  Cautious, not Normal.",
            "- A vehicle in a dense platoon with a short gap may be constrained by",
            "  traffic. If it does not accelerate hard or brake late, do not call it",
            "  Aggressive solely from the short gap.",
            *no_leader_counterexample,
            "- A vehicle with a very large headway to a distant leader is unconstrained,",
            "  not Cautious; use its speed and acceleration instead.",
            "",
            "The three styles are equally frequent in this benchmark. Do not default to",
            "Normal or Cautious. Evidence is ordinal but overlapping; 0.9 means strong",
            "evidence, while weak evidence stays near 0.33.",
            "Classify each vehicle independently. Do not copy one vehicle's label to",
            "another or infer every vehicle from the scene-level impression.",
            *constant_speed_rule,
            "",
            "For each vehicle, use this order:",
            "1. Check whether leader_evidence is available. If it is unavailable,",
            "   skip the headway/TTC bands and use the secondary evidence above.",
            "2. If available, read its own-leader headway and minimum TTC over the",
            "   whole window.",
            "3. Check acceleration and braking, especially late braking and strong",
            "   positive acceleration.",
            "4. Check lane changes and lateral motion.",
            *step5_rule,
            "6. If evidence is mixed, keep the distribution near the prior.",
            "",
            f"Sampling interval: {self.dt:.3f} s; ego current speed: {ego_speed:.2f} m/s.",
            "front_gap/headway/TTC are measured from the target vehicle to its own",
            "observed leader. NA means no leader or no meaningful closing TTC was",
            "observable.",
            "",
            "Vehicle summaries:",
        ]

        for item in targets:
            vehicle_id = item["id"]
            summary = self._summarize_vehicle_history_relative(history, vehicle_id)
            if summary is None:
                continue
            dx_end = summary["dx_end"]
            if dx_end > 1.0:
                position = "ahead"
            elif dx_end < -1.0:
                position = "behind"
            else:
                position = "alongside"
            relative_speed = summary["relative_speed_mean"]
            relative_text = "NA" if relative_speed is None else f"{relative_speed:+.2f}m/s"
            leader_relative = summary["leader_relative_speed_mean"]
            leader_relative_text = ("NA" if leader_relative is None
                                    else f"{leader_relative:+.2f}m/s")
            stream_speed_text = self._format_optional(summary["stream_speed_mean"], "m/s")
            stream_relative = summary["stream_relative_speed_mean"]
            stream_relative_text = ("NA" if stream_relative is None
                                    else f"{stream_relative:+.2f}m/s")
            headway_available = (
                summary["headway_mean"] is not None
                and summary["headway_mean"] < 4.0
                and summary["front_gap_min"] is not None
                and summary["front_gap_min"] < 50.0
            )
            if speed_offset_is_evidence and not headway_available:
                # A relative speed to a distant/missing leader is not evidence.
                leader_relative_text = "NA"
            if headway_available:
                leader_evidence = (
                    f"available; headway(mean/min)="
                    f"{self._format_optional(summary['headway_mean'], 's')}/"
                    f"{self._format_optional(summary['headway_min'], 's')},"
                    f" min_TTC={self._format_optional(summary['ttc_min'], 's')},"
                    f" front_gap(mean/min)="
                    f"{self._format_optional(summary['front_gap_mean'], 'm')}/"
                    f"{self._format_optional(summary['front_gap_min'], 'm')}"
                )
            else:
                leader_evidence = (
                    "unavailable (leader too distant or missing); "
                    "do not use headway/TTC bands"
                )
            if speed_offset_is_evidence and not headway_available:
                # For a leaderless target the normalized speed offset is the
                # level cue.  The aggregate absolute speed line is omitted
                # because it dominates the decision and ignores the offset;
                # per-frame speeds remain in the sampled trajectory.
                motion_line = (
                    f"  accel(mean/min/max)={summary['acceleration_mean']:+.2f}/"
                    f"{summary['acceleration_min']:+.2f}/"
                    f"{summary['acceleration_max']:+.2f}m/s2,"
                    f" hard_braking_frames={summary['hard_braking_frames']},"
                    f" hard_acceleration_frames={summary['hard_acceleration_frames']},"
                    f" lane_changes={summary['lane_changes']}"
                )
                interaction_line = (
                    f"  interaction_context (supporting only; relative_speed is"
                    f" the style cue, do not use gap direction to override it):"
                    f" relative_speed={relative_text} (target minus ego; positive ="
                    f" target faster)"
                )
            else:
                motion_line = (
                    f"  speed(start/mean/end/std)={summary['speed_start']:.1f}/"
                    f"{summary['speed_mean']:.1f}/{summary['speed_end']:.1f}/"
                    f"{summary['speed_std']:.2f}m/s,"
                    f" speed_slope={summary['speed_slope']:+.2f}m/s2,"
                    f" accel(mean/min/max)={summary['acceleration_mean']:+.2f}/"
                    f"{summary['acceleration_min']:+.2f}/"
                    f"{summary['acceleration_max']:+.2f}m/s2,"
                    f" hard_braking_frames={summary['hard_braking_frames']},"
                    f" hard_acceleration_frames={summary['hard_acceleration_frames']},"
                    f" lane_changes={summary['lane_changes']}"
                )
                interaction_line = (
                    f"  interaction_context (secondary evidence; combine with speed"
                    f" level, acceleration, and gap trend):"
                    f" ego_relative_gap_rate={summary['gap_rate']:+.2f}m/s,"
                    f" dx_rate={summary['dx_rate']:+.2f}m/s,"
                    f" relative_speed={relative_text},"
                    f" closing_speed={summary['closing_speed']:.1f}m/s"
                )
            lines.extend(
                [
                    f"- id={vehicle_id}: position={position}"
                    f" (dx_signed={dx_end:+.1f}m, dy={item['dl']:+.1f}m),"
                    f" observed_frames={summary['observed_frames']}"
                    f" over {summary['duration']:.1f}s",
                    f"  own_style_evidence: leader_evidence={leader_evidence};"
                    f" leader_relative_speed={leader_relative_text};"
                    f" stream(mean)={stream_speed_text};"
                    f" stream_relative={stream_relative_text}",
                    motion_line,
                    interaction_line,
                    "  sampled trajectory: " + " | ".join(summary["trajectory"]),
                ]
            )

        example = {str(vehicle_id): ["p_cautious", "p_normal", "p_aggressive"]
                   for vehicle_id in target_ids}
        lines.extend(
            [
                "Return one JSON object only, keyed by vehicle id. Each array must",
                "contain three non-negative numbers summing to 1, in the order",
                "[Cautious, Normal, Aggressive].",
                json.dumps(example, ensure_ascii=False),
            ]
        )
        prompt = "\n".join(lines)
        dump_path = os.environ.get("LLM_DUMP_PROMPT_PATH")
        if dump_path:
            with open(dump_path, "a", encoding="utf-8") as handle:
                handle.write(prompt + "\n" + "=" * 80 + "\n")
        return prompt, target_ids

    @staticmethod
    def _extract_json(raw_response: str) -> Optional[dict]:
        if not raw_response:
            return None
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_response, re.DOTALL | re.IGNORECASE)
        candidate = fenced.group(1) if fenced else raw_response.strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start < 0 or end <= start:
                return None
            try:
                parsed = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _coerce_probabilities(value: object) -> Optional[List[float]]:
        if isinstance(value, dict):
            for key in ("probabilities", "probs", "probability"):
                if key in value:
                    value = value[key]
                    break
            else:
                lowered = {str(k).lower(): v for k, v in value.items()}
                if all(k in lowered for k in ("yield", "normal", "force")):
                    value = [lowered["yield"], lowered["normal"], lowered["force"]]
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return None
        try:
            probs = [float(number) for number in value]
        except (TypeError, ValueError):
            return None
        if any(not math.isfinite(number) or number < 0 for number in probs):
            return None
        total = sum(probs)
        if total <= 0:
            return None
        return [number / total for number in probs]

    def parse_llm_response(self, raw_response: str, target_ids: List[int]) -> Optional[Dict[int, List[float]]]:
        """Parse valid predictions without silently replacing failures by Normal."""
        data = self._extract_json(raw_response)
        if data is None:
            return None

        beliefs: Dict[int, List[float]] = {}
        for idx, vehicle_id in enumerate(target_ids):
            value = None
            # Direct IDs are preferred. Legacy positional keys remain accepted.
            for key in (str(vehicle_id), vehicle_id, f"car_{idx + 1}", f"vehicle_{idx + 1}"):
                if key in data:
                    value = data[key]
                    break
            probs = self._coerce_probabilities(value)
            if probs is not None:
                beliefs[vehicle_id] = probs
        return beliefs or None

    def _create_client(self):
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not configured")
        if OpenAI is not None:
            return OpenAI(api_key=self.api_key, base_url=self.base_url,
                          timeout=self.timeout_s, max_retries=self.max_retries)
        if ZhipuAI is not None:
            return ZhipuAI(api_key=self.api_key, base_url=self.base_url)
        raise RuntimeError("Install the 'openai' package to enable online LLM inference")

    def get_response(self, prompt: str, target_ids: List[int]) -> Optional[Dict[int, List[float]]]:
        """Call the configured OpenAI-compatible endpoint and parse its response."""
        started = time.perf_counter()
        self.last_error = None
        self.last_raw_response = None
        try:
            if self._client is None:
                self._client = self._create_client()
            client = self._client
            extra = {}
            if "flash" in self.model or self.model.startswith("qwen3"):
                extra['extra_body'] = {'enable_thinking': self.enable_thinking}
            completion = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=256,
                response_format={"type": "json_object"},
                **extra,
            )
            self.last_usage = completion.usage.model_dump() if completion.usage else None
            self.last_response_model = completion.model
            self.last_raw_response = completion.choices[0].message.content
            return self.parse_llm_response(self.last_raw_response, target_ids)
        except Exception as exc:  # API failures should not terminate the control loop.
            # Provider errors may echo request information; redact the credential.
            self.last_error = f"{type(exc).__name__}: {exc}".replace(self.api_key or '\x00', '[REDACTED]')
            return None
        finally:
            self.last_latency_s = time.perf_counter() - started

    def smooth_probabilities(self, beliefs: Optional[Dict[int, List[float]]]) -> Optional[Dict[int, List[float]]]:
        """EMA-filter persistent vehicle types across consecutive inference windows."""
        if not beliefs:
            return beliefs
        smoothed = {}
        for vehicle_id, current in beliefs.items():
            previous = self.smoothed_beliefs.get(vehicle_id)
            if previous is None or self.temporal_alpha <= 0:
                fused = list(current)
            else:
                fused = [
                    self.temporal_alpha * old + (1 - self.temporal_alpha) * new
                    for old, new in zip(previous, current)
                ]
                total = sum(fused)
                fused = [value / total for value in fused]
            self.smoothed_beliefs[vehicle_id] = fused
            smoothed[vehicle_id] = fused
        return smoothed

    def run(self) -> None:
        while not self.stop_event.is_set():
            should_run = self.trigger_queue.get()
            if not should_run or self.stop_event.is_set():
                break
            try:
                with self.lock:
                    history = list(self.history_buffer)
                prompt, target_ids = self._generate_prompt(history)
                if not prompt:
                    continue
                beliefs = self.smooth_probabilities(self.get_response(prompt, target_ids))
                with self.lock:
                    self.latest_beliefs = beliefs
                    self.belief_timestamp = time.time()
            except Exception as exc:  # Keep the daemon alive after malformed scene data.
                self.last_error = f"{type(exc).__name__}: {exc}"

    def feed_data(self, scene_snapshot: dict, request_inference: bool = True) -> None:
        """Add one frame and optionally request inference if the worker is idle."""
        with self.lock:
            self.history_buffer.append(copy.deepcopy(scene_snapshot))
        if request_inference and not self.stop_event.is_set() and not self.trigger_queue.full():
            try:
                self.trigger_queue.put_nowait(True)
            except queue.Full:  # Another producer won the race.
                pass

    def fetch_result(self):
        with self.lock:
            if self.latest_beliefs is None:
                return None, 0
            return copy.deepcopy(self.latest_beliefs), self.belief_timestamp

    def stop(self) -> None:
        """Ask the background worker to exit after its current API call."""
        self.stop_event.set()
        try:
            self.trigger_queue.put_nowait(False)
        except queue.Full:
            pass
