from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.ablation_v2.config import load_config, missing_credentials


def main():
    parser=argparse.ArgumentParser(description='Versioned llmgame ablation experiments')
    parser.add_argument('--config',default='config/ablation_v2_dev.json')
    parser.add_argument('--mode',choices=['validate','mechanism','pilot','formal','report'],default='validate')
    parser.add_argument('--output-dir',default='simulation_results/ablation_v2/pilot_001')
    parser.add_argument('--run-dir')
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--seeds',help='Comma separated explicit seeds; pilot only')
    parser.add_argument('--variants',nargs='+')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--wait-for',nargs='+',help='Completion markers required before starting this run')
    args=parser.parse_args()
    while args.wait_for and not all(Path(p).exists() for p in args.wait_for):
        print('Waiting for preceding run completion markers',flush=True)
        time.sleep(30.)
    cfg=load_config(args.config)
    if args.mode=='report':
        from experiments.ablation_v2.reporting import report
        report(Path(args.run_dir or args.output_dir))
        return
    if args.max_steps:
        cfg['runtime']['max_steps']=args.max_steps
    if args.seeds:
        cfg['seeds']=[int(s) for s in args.seeds.split(',')]
    if args.variants:
        cfg['variants']=args.variants
    if args.mode=='formal' and (args.max_steps or args.seeds or args.variants or cfg['phase']!='frozen'):
        raise ValueError('Formal run requires frozen config, without overrides')
    if args.mode=='validate':
        print(json.dumps(dict(config_valid=True,missing_credentials_for=missing_credentials(cfg),
                              formal_ready=cfg['phase']=='frozen'),indent=2))
        return
    if args.mode=='mechanism':
        from experiments.ablation_v2.mechanisms import run_mechanisms
        run_mechanisms(cfg,Path(args.output_dir))
    else:
        from experiments.ablation_v2.runner import execute
        execute(cfg,args.output_dir,phase=args.mode)
        from experiments.ablation_v2.reporting import report
        report(Path(args.output_dir))


if __name__=='__main__':
    main()
