"""Document section 6.4: controlled delays, explicitly not real API inference."""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments.ablation_v2.belief_provider import BeliefProvider
from experiments.ablation_v2.runner import dump, write_csv


def main():
    cfg=json.loads(Path('config/ablation_v2_pilot.json').read_text(encoding='utf-8'))
    out=Path('simulation_results/ablation_v2/latency_sweep_001')
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    with patch.dict(os.environ,{'DASHSCOPE_API_KEY':'mock-not-a-credential'}):
        for delay in (0.,.5,1.,2.,4.):
            for repeat in range(3):
                for variant in ('full_model','sync_llm'):
                    provider=BeliefProvider(variant,dict(cfg['llm'],history_len=1,interval_steps=1),'delay_sweep')
                    def response(history,sim,wall):
                        time.sleep(delay)
                        return dict(probabilities={1:[.2,.5,.3]},observation_sim_time=sim,
                                    observation_wall=wall,latency_ms=delay*1000.)
                    provider._request=response
                    data=SimpleNamespace(ego_state=np.zeros(6),others_info={'ids':[1]},curr_lane_idx=0)
                    started=time.perf_counter()
                    provider.update(0,data,SimpleNamespace(time=0.))
                    blocked_ms=(time.perf_counter()-started)*1000.
                    provider.close()
                    rows.append(dict(delay_s=delay,repeat=repeat,variant=variant,control_block_ms=blocked_ms))
                    write_csv(out/'trials.csv',rows)
            print(f'Completed controlled delay {delay}s',flush=True)
    dump(out/'summary.json',dict(mock_only=True,real_api_requests=0,trials=len(rows),
        delays_s=[0.,.5,1.,2.,4.],repetitions=3,
        max_async_control_block_ms=max(r['control_block_ms'] for r in rows if r['variant']=='full_model'),
        rows=rows))


if __name__=='__main__':
    main()
