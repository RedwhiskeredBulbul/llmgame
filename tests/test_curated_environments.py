"""Exercise the registry, route adapter, seed handling, and trajectory API."""

import importlib
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.registration import registry

import highway_env
from highway_env.envs.common.abstract import AbstractEnv
from model.perception.env_adapter import (
    EnvironmentAdapter,
    lane_id_refline_dict,
    lane_id_targetlane_dict,
    lane_path_map_dict,
)


ENV_IDS = ("CutinEnv", "HighwayEnv", "AdversarialMergeEnv", "MergeEnv")


def vehicle_state(raw):
    return np.asarray(
        [[v.id, *v.position, v.heading, v.speed] for v in raw.road.vehicles]
    )


def trajectory_action(raw):
    x, y = raw.vehicle.position
    speed, heading = raw.vehicle.speed, raw.vehicle.heading
    t = np.arange(10) / 10
    trajectory = np.column_stack(
        (
            x + speed * t * np.cos(heading),
            y + speed * t * np.sin(heading),
            np.full(10, speed),
            np.full(10, heading),
            np.zeros(10),
            speed * t,
        )
    )
    return {"trajectory": trajectory, "action": np.zeros((10, 2)).tolist()}


def test_only_four_public_environments():
    import highway_env.envs as envs

    assert set(envs.__all__) == set(ENV_IDS)
    assert set(highway_env.ENVIRONMENTS) == set(ENV_IDS)
    assert {
        k
        for k, v in registry.items()
        if isinstance(v.entry_point, str) and v.entry_point.startswith("highway_env.")
    } == set(ENV_IDS)
    importlib.reload(highway_env)
    assert all(
        registry[name].entry_point == entry
        for name, entry in highway_env.ENVIRONMENTS.items()
    )
    for name in ENV_IDS:
        assert getattr(envs, name).__bases__ == (AbstractEnv,)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_reset_route_maps_step_and_render(env_id):
    env = gym.make(env_id, render_mode="rgb_array")
    try:
        env.reset(seed=13)
        raw = env.unwrapped
        initial = vehicle_state(raw)
        for a, destinations in raw.road.network.graph.items():
            for b, lanes in destinations.items():
                for index in range(len(lanes)):
                    logical = lane_path_map_dict[env_id][a, b, index]
                    assert logical in lane_id_refline_dict[env_id]
                    assert all(
                        target in lane_id_refline_dict[env_id]
                        for target in lane_id_targetlane_dict[env_id][logical]
                    )
        adapter = EnvironmentAdapter()
        for _ in range(3):
            adapter.update_data(env_id, env)
            obs, reward, terminated, truncated, info = env.step(trajectory_action(raw))
            assert np.isfinite(reward)
            assert np.isfinite(raw.vehicle.position).all()
        frame = env.render()
        assert frame.shape == (900, 1800, 3)
        env.reset(seed=13)
        np.testing.assert_allclose(vehicle_state(raw), initial)
        first_random = raw.np_random.random()
        env.reset(seed=13)
        assert raw.np_random.random() == first_random
    finally:
        env.close()


def test_experiment_source_hashes_reference_existing_files():
    from experiments.ablation_v2.config import source_hashes

    hashes = source_hashes()
    assert "highway_env/envs/cutin_env.py" in {Path(key).as_posix() for key in hashes}
    assert all(len(digest) == 64 for digest in hashes.values())


def test_migrated_vehicle_type_collection_keeps_ground_truth():
    from experiments.vehicle_types.collect import collect_cases

    cases = collect_cases([13], windows_per_seed=1, history_len=2, dt=0.1)
    assert len(cases) == 1
    assert cases[0]["case_id"].startswith("CutinEnv-")
    assert cases[0]["ground_truth"]
    assert set(cases[0]["ground_truth"].values()) <= {
        "Cautious",
        "Normal",
        "Aggressive",
    }


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_offline_model_pipeline_includes_controller(env_id):
    from simulation.runner import one_case

    records = one_case(env_id, seed=13, max_steps=1, use_llm=False)
    assert any(row["id"] == "AV" for row in records)
    assert all(
        np.isfinite([row[key] for key in ("x", "y", "heading", "v", "a")]).all()
        for row in records
    )
