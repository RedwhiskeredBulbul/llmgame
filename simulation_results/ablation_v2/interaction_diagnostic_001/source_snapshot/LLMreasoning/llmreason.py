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
    ) -> None:
        super().__init__(daemon=True)
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
