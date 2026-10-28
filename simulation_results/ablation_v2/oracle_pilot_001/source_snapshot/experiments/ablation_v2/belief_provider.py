from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from LLMreasoning.llmreason import LLMReasoningWorker
from LLMreasoning.vehicle_type_evaluation import vehicle_ground_truth
from experiments.ablation_v2.config import LLM_VARIANTS, canonical_hash


def hard_probabilities(probs, valid=True):
    if not valid:
        return [1/3]*3
    result = [0.,0.,0.]
    # Tie rule: Normal first, then Yield, then Force.
    best = max((1,0,2), key=lambda i:probs[i])
    result[best] = 1.
    return result


class BeliefProvider:
    def __init__(self, variant, config, episode_id):
        self.variant, self.config, self.episode_id = variant, config, episode_id
        self.events, self.history = [], []
        self.beliefs = {}
        self.future = None
        self.request_count = 0
        self.observation_time = None
        self.observation_wall = None
        self.valid_ids = set()
        self.executor = None
        self.worker = None
        if variant in LLM_VARIANTS:
            self.worker = LLMReasoningWorker(model=config['model'], dt=.1, history_len=config['history_len'],
                          temporal_alpha=config['temporal_alpha'], timeout_s=config['timeout_s'], max_retries=config['max_retries'])
            if not self.worker.api_key:
                raise RuntimeError('DASHSCOPE_API_KEY not configured')
            self.executor = ThreadPoolExecutor(max_workers=1)

    def _request(self, history, sim_time, wall):
        prompt, ids = self.worker._generate_prompt(history)
        probs = self.worker.get_response(prompt, ids) if prompt else None
        return dict(probabilities=probs, observation_sim_time=sim_time, observation_wall=wall,
                    latency_ms=(self.worker.last_latency_s or 0.)*1000.,
                    error=self.worker.last_error, prompt_hash=canonical_hash(prompt),
                    response=self.worker.last_raw_response, usage=self.worker.last_usage)

    def _accept(self, event):
        probs = self.worker.smooth_probabilities(event.pop('probabilities'))
        event.update(episode_id=self.episode_id, event='response', request_id=self.request_count,
                     probabilities=probs)
        self.events.append(event)
        if probs:
            self.beliefs.update(probs)
            self.valid_ids = set(probs)
            self.observation_time = event['observation_sim_time']
            self.observation_wall = event['observation_wall']

    def update(self, step, data, env):
        ids = data.others_info['ids']
        if self.variant=='normal_type':
            return {vid:[0.,1.,0.] for vid in ids}, {'source':'fixed_normal'}
        if self.variant.startswith('oracle'):
            by_id = {v.id:v for v in env.road.vehicles}
            labels = {'Cautious':0,'Normal':1,'Aggressive':2}
            result = {}
            for vid in ids:
                label = labels.get(vehicle_ground_truth(by_id[vid]))
                result[vid] = [float(j==label) for j in range(3)] if label is not None else [1/3]*3
            return result, {'source':'oracle_ground_truth'}
        if self.variant=='uniform_belief':
            return {vid:[1/3]*3 for vid in ids}, {'source':'uniform'}
        snapshot = dict(ego_state=data.ego_state.tolist(), others_info=copy.deepcopy(data.others_info), curr_lane_idx=data.curr_lane_idx)
        self.history.append(snapshot)
        self.history = self.history[-self.config['history_len']:]
        if self.future is not None and self.future.done():
            self._accept(self.future.result())
            self.future = None
        due = step+1 >= self.config['history_len'] and (step+1-self.config['history_len']) % self.config['interval_steps']==0
        if due and self.future is None:
            self.request_count += 1
            history, sim_time, wall = copy.deepcopy(self.history), env.time, time.perf_counter()
            if self.variant=='sync_llm':
                self._accept(self._request(history,sim_time,wall))
            else:
                self.future = self.executor.submit(self._request,history,sim_time,wall)
        age = env.time-self.observation_time if self.observation_time is not None else None
        current = self.observation_time is not None and age <= self.config['max_belief_age_sim_s']
        result = {}
        for vid in ids:
            valid = current and vid in self.valid_ids
            prob = self.beliefs[vid] if valid else [1/3]*3
            result[vid] = hard_probabilities(prob,valid) if self.variant=='hard_type' else list(prob)
        return result, {'source':'llm' if current else 'uniform_fallback', 'belief_age_sim_s':age,
                        'belief_age_wall_s':time.perf_counter()-self.observation_wall if self.observation_wall else None}

    def close(self):
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
            if self.future and self.future.done() and not self.future.cancelled():
                event = self.future.result()
                event.update(event='response_after_episode', episode_id=self.episode_id)
                self.events.append(event)
        if self.worker and self.worker._client:
            self.worker._client.close()
