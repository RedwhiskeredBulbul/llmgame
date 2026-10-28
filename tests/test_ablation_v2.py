import contextlib
import copy
import io
import unittest
from types import SimpleNamespace

import numpy as np

from experiments.ablation_v2.belief_provider import hard_probabilities
from experiments.ablation_v2.config import load_config
from experiments.ablation_v2.controller import ExperimentGameSolver,scene_data
from experiments.ablation_v2.metrics import rectangle_clearance,summarize_steps
from experiments.ablation_v2.runner import make_env,build_action,initial_state
from model.perception.env_adapter import get_hv_preds
from model.game.solver import GameSolver


class DataTests(unittest.TestCase):
    def test_vehicle_dimensions_follow_filtered_vehicle_identity(self):
        def vehicle(vid,length):
            return SimpleNamespace(id=vid,LENGTH=length,WIDTH=2.,position=np.zeros(2),heading=0.,speed=3.,action={'acceleration':0.},type='car')
        for values in ([vehicle(888,5),vehicle(7,9)], [vehicle(7,9),vehicle(888,5)]):
            _,pred,ids,_=get_hv_preds(values,888)
            self.assertEqual(ids,[7])
            self.assertEqual(pred[0,0,4],9.)

    def test_hard_fallback_does_not_become_yield(self):
        np.testing.assert_allclose(hard_probabilities([1/3]*3,False),[1/3]*3)
        self.assertEqual(hard_probabilities([.5,.5,0.]),[0.,1.,0.])

    def test_geometric_clearance_and_overlap(self):
        a=SimpleNamespace(position=np.array([0.,0.]),heading=0.,LENGTH=5.,WIDTH=2.)
        b=copy.deepcopy(a)
        b.position[0]=8.
        self.assertAlmostEqual(rectangle_clearance(a,b),3.)
        b.position[0]=3.
        self.assertEqual(rectangle_clearance(a,b),0.)
        b.position[:]=[0.,4.]
        self.assertAlmostEqual(rectangle_clearance(a,b),2.)

    def test_failed_episode_keeps_horizon_penalty(self):
        result=summarize_steps([], 'runtime_error',40.,None)
        self.assertEqual(result['success'],0)
        self.assertEqual(result['restricted_task_time_s'],40.)
        self.assertIsNone(result['average_speed_ms'])


class PromptTests(unittest.TestCase):
    @staticmethod
    def history():
        samples = []
        for index in range(20):
            samples.append(dict(
                ego_state=[0., 0., 0., 7., 0., 0.],
                curr_lane_idx=1,
                others_info=dict(ids=[1], states=[[-20. + 0.2 * index, 0., 0., 12., 0., 0.]],
                                 frenet_relative_coords=[[-20. + 0.2 * index, 0.]],
                                 laneids=[0])))
        return samples

    def test_relative_prompt_reports_gap_dynamics_not_absolute_prototypes(self):
        from model.reasoning.worker import LLMReasoningWorker
        worker = LLMReasoningWorker(prompt_version='relative_evidence_v1', history_len=20)
        prompt, ids = worker._generate_prompt(self.history())
        self.assertEqual(ids, [1])
        self.assertIn('ego_relative_dx', prompt)
        self.assertIn('relative_speed_mean', prompt)
        self.assertIn('do not default', prompt.lower())
        self.assertNotIn('desired speed about', prompt)
        self.assertNotIn('30.6', prompt)

    def test_legacy_prompt_remains_available(self):
        from model.reasoning.worker import LLMReasoningWorker
        worker = LLMReasoningWorker(prompt_version='legacy', history_len=20)
        prompt, ids = worker._generate_prompt(self.history())
        self.assertEqual(ids, [1])
        self.assertIn('desired speed about', prompt)
        self.assertIsNone(worker.parse_llm_response('not json', ids))

    def test_unknown_prompt_version_is_rejected(self):
        from model.reasoning.worker import LLMReasoningWorker
        with self.assertRaises(ValueError):
            LLMReasoningWorker(prompt_version='unknown')


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=load_config('configs/ablation_v2_dev.json')['decision']
        cls.solver=ExperimentGameSolver(cls.config)

    def test_compiled_hv_reward_matches_parent_formula(self):
        s=self.solver
        s.weight_dict[7]=[2.,1.,0.]
        args=(1,[.2]*6,np.array([[.1]*6,[.2]*6]),np.zeros((2,6)),[0.,0.],[6.,7.],['AV','car'],[888,7],np.array([-1.,12.]),np.array([-1.,20.]))
        expected=float(GameSolver._cal_discrete_reward(s,*args))
        self.assertAlmostEqual(s._cal_discrete_reward(*args),expected,places=7)

    def test_invalid_probability_rejected_at_solver_boundary(self):
        with self.assertRaises(ValueError):
            self.solver.inject_llm_beliefs({7:[-1.,1.,1.]})

    def test_probe_ratio_is_applied_once_and_target_line_is_anchored(self):
        case=dict(scenario='merge',seed=1007,difficulty=1,behaviour_index=1)
        env=make_env(case)
        data=scene_data(env.unwrapped)
        solver=ExperimentGameSolver(self.config)
        for ego_y in (4., 3.4, 2.9):
            pose=(20.,ego_y,0.,6.)
            path,_=solver._get_target_path(0,data.lanes,1,pose,lat_ratio=.3)
            self.assertAlmostEqual(float(path[-1,1]),2.8,places=6)
            self.assertLess(np.max(np.abs(np.diff(path[:,1]))),.2)
        env.close()

    def test_ttcp_detects_equal_arrival_with_positive_conflict_coordinates(self):
        reward=float(self.solver._GameSolver__reward_delta_TTCP(0.,20.,5.,0.,20.,5.))
        self.assertLess(reward,.1)
        no_conflict=float(self.solver._GameSolver__reward_delta_TTCP(0.,-1.,5.,0.,20.,5.))
        self.assertAlmostEqual(no_conflict,5.)

    def test_merge_observation_phase_gates_target_lane_until_open_x(self):
        case=dict(scenario='merge',seed=1007,difficulty=1,behaviour_index=1)
        env=make_env(case)
        raw=env.unwrapped
        data=scene_data(raw)
        self.assertEqual(data.near_target_idxes,[0])
        raw.merge_open_x=raw.vehicle.position[0]+5.
        data=scene_data(raw)
        self.assertEqual(data.near_target_idxes,[data.curr_lane_idx])
        raw.vehicle.position[0]+=10.
        raw.vehicle.on_state_update()
        data=scene_data(raw)
        self.assertEqual(data.near_target_idxes,[0])
        env.close()

    def test_scene_reset_is_seeded_and_action_has_correct_dt(self):
        case=dict(scenario='merge',seed=1007,difficulty=1,behaviour_index=1)
        first=make_env(case)
        second=make_env(case)
        self.assertEqual(initial_state(first.unwrapped),initial_state(second.unwrapped))
        data=scene_data(first.unwrapped)
        solver=ExperimentGameSolver(self.config)
        with contextlib.redirect_stdout(io.StringIO()):
            traj,acc,_=solver.decide(data,{vid:[1/3]*3 for vid in data.others_info['ids']})
        self.assertAlmostEqual(traj[1,4],.1)
        first.step(build_action(traj,acc))
        np.testing.assert_allclose(first.unwrapped.vehicle.position,traj[1,:2])
        self.assertTrue(all(not r['infeasible_bounds'] for r in solver.solve_diagnostics))
        self.assertTrue(all(np.isfinite(r['objective']) for r in solver.solve_diagnostics))
        first.close();second.close()


if __name__=='__main__':
    unittest.main()
