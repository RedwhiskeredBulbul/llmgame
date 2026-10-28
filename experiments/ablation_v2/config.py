from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLM_VARIANTS = {"full_model", "hard_type", "without_interaction_response", "without_probe", "sync_llm"}
OFFLINE_VARIANTS = {"uniform_belief", "normal_type", "oracle_type", "oracle_without_probe", "oracle_without_interaction_response"}
VARIANTS = LLM_VARIANTS | OFFLINE_VARIANTS


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def source_hashes():
    paths = {ROOT / "main.py"}
    for directory in ("model", "simulation", "highway_env", "experiments"):
        paths.update((ROOT / directory).rglob("*.py"))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if not cfg["variants"] or not set(cfg["variants"]) <= VARIANTS:
        raise ValueError("Unknown or empty variants")
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Seeds must be nonempty and unique")
    if not set(cfg["scenarios"]) <= {"merge", "cutin", "traffic"}:
        raise ValueError("Unknown scenarios")
    if cfg["decision"]["solve_budget_ms"] <= 0:
        raise ValueError("An explicit positive solver budget is required")
    return cfg


def cases(cfg):
    # Balanced within each scenario: independent cells, stable under run reordering.
    for scenario in cfg["scenarios"]:
        for i, seed in enumerate(cfg["seeds"]):
            difficulty = i % 3
            behaviour = (i // 3 + i) % 3
            yield {"case_id": f"{scenario}-{seed}", "scenario": scenario, "seed": seed,
                   "difficulty": difficulty, "behaviour_index": behaviour}


def missing_credentials(cfg):
    from model.reasoning.worker import runtime_setting
    return sorted(set(cfg["variants"]) & LLM_VARIANTS) if not runtime_setting("DASHSCOPE_API_KEY") else []
