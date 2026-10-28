"""Offline development follow-up to the prespecified fixed-state interventions.

Not a held-out benchmark: geometry comes from mechanism_003, selected before
these closed-loop outcomes are observed. Never modifies the ongoing pilot.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from unittest.mock import patch

from experiments.ablation_v2 import runner
from experiments.ablation_v2.config import canonical_hash, source_hashes


CONDITIONS = [(-12., 1.5, 6.), (-6., 2., 6.), (0., 2., 6.),
              (8., -1., 6.), (-6., 1., 4.), (-12., 2., 8.)]
VARIANTS = ['uniform_belief', 'oracle_type', 'oracle_without_probe']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', required=True)
    p.add_argument('--wait-for', help='Wait for another run completion marker to avoid CPU contention')
    args = p.parse_args()
    while args.wait_for and not Path(args.wait_for).exists():
        print('Waiting for preceding run completion',flush=True)
        time.sleep(30.)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path('configs/ablation_v2_pilot.json').read_text(encoding='utf-8'))
    cfg.update(phase='development_diagnostic', variants=VARIANTS,
               protocol_version='interaction-diagnostic-v1-20261016')
    hashes = source_hashes()
    for relative in hashes:
        snapshot=out/'source_snapshot'/relative
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(relative,snapshot)
    runner.dump(out/'manifest.json', dict(config=cfg, conditions=CONDITIONS,
        source_hashes=hashes, code_hash=canonical_hash(hashes),
        limitation='Development diagnostic only; no API, no inferential timing comparison',
        geometry='Same six states as mechanism_003, crossed with all three background types'))
    original = runner.make_env
    results = []
    for state, (rx, dv, speed) in enumerate(CONDITIONS):
        for behaviour in range(3):
            case = dict(case_id=f'merge-state{state}-type{behaviour}', scenario='merge',
                        seed=1010+state, difficulty=1, behaviour_index=behaviour)

            def make_env(spec, rx=rx, dv=dv, speed=speed):
                env = original(spec)
                raw = env.unwrapped
                raw.vehicle.speed = speed
                raw.key_vehicle.position[0] = raw.vehicle.position[0]+rx
                raw.key_vehicle.speed = speed+dv
                raw.key_vehicle.on_state_update()
                return env

            for variant in VARIANTS:
                with patch.object(runner, 'make_env', make_env):
                    result = runner.run_episode(cfg, case, variant, out)
                results.append(result)
                runner.write_csv(out/'episodes.csv', results)
                print(f'[{len(results)}/54] {case["case_id"]} {variant}: '
                      f'{result["status"]} probe={result["probe_selected_steps"]}', flush=True)
    runner.dump(out/'completion.json', dict(planned=54, finished=len(results),
        runtime_errors=sum(r['status']=='runtime_error' for r in results)))


if __name__ == '__main__':
    main()
