"""Simulation loop: adapt observations, infer beliefs, solve game, track trajectory."""

import copy
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from model.game.solver import GameSolver
from model.reasoning.worker import LLMReasoningWorker
from model.control.stanley import StanleyController
from model.perception.env_adapter import EnvironmentAdapter
from model.perception.filter import InteractionFilter, set_limit_speed
from simulation.actions import build_action


def _record_states(records, step, data):
    states = [("AV", data.ego_state)]
    states.extend(zip(data.others_info["ids"], data.others_info["states"]))
    for vehicle_id, state in states:
        records.append(
            dict(
                time=step,
                id=vehicle_id,
                x=state[0],
                y=state[1],
                heading=state[2],
                v=state[3],
                a=state[4],
            )
        )


def one_case(env_name, seed, max_steps, *, record_video=False, use_llm=True):
    """Return pre-action vehicle states for one episode, preserving model defaults."""
    records = []
    worker = None
    env = gym.make(env_name, render_mode="rgb_array")
    try:
        if record_video:
            env = RecordVideo(
                env,
                video_folder=f"results/{env_name}/seed_{seed}",
                episode_trigger=lambda _: True,
                name_prefix="lane_change",
            )
        env.reset(seed=seed)
        data = EnvironmentAdapter()
        solver = GameSolver(roslog=False, AV_style="Conservative")
        interaction_filter = InteractionFilter()
        controller = StanleyController()
        if use_llm:
            worker = LLMReasoningWorker()
            worker.start()
        ego = env.unwrapped.controlled_vehicles[0]
        ego.color = (127, 255, 0)
        for step in range(max_steps):
            # Preserve the original passenger-instruction change at step 380.
            if step == 380:
                solver.init_config(AV_style="Aggressive")
                ego.color = (255, 97, 3)
            data.update_data(env_name, env)
            limit_speed = set_limit_speed(data, solver)
            others = interaction_filter.get_close_vehicles_game_lane(data, limit_speed)
            if worker is not None:
                worker.feed_data(
                    dict(
                        ego_state=copy.deepcopy(data.ego_state),
                        others_info=copy.deepcopy(others),
                        curr_lane_idx=data.curr_lane_idx,
                    )
                )
                beliefs, _ = worker.fetch_result()
                if beliefs is not None:
                    solver.inject_llm_beliefs(beliefs)
            trajectory, acceleration, predictions, _ = solver._prallel_solve_game(
                data_process=data, others_info=others
            )
            action = build_action(trajectory, acceleration, data.ego_state, controller)
            data.lane_changeing = solver.changing
            data.game_pred_trajs = predictions
            ego.game_traj = trajectory[:, :4]
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
            _record_states(records, step, data)
        return records
    finally:
        if worker is not None:
            worker.stop()
            worker.join(timeout=1.0)
        env.close()
