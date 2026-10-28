"""Run the LLM game decision model in one of the four supported scenes."""

import argparse
import json
from pathlib import Path
import pandas as pd
import highway_env
from simulation.runner import one_case
from simulation.metrics import calculate_episode_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env", choices=tuple(highway_env.ENVIRONMENTS), default="HighwayEnv"
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--no-llm", action="store_true", help="Disable online LLM belief updates"
    )
    args = parser.parse_args()
    records = one_case(
        args.env,
        args.seed,
        args.max_steps,
        record_video=args.record_video,
        use_llm=not args.no_llm,
    )
    output = Path("results") / args.env / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output / "vehicle_states.csv", index=False)
    metrics = calculate_episode_metrics(records, ego_id="AV")
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(metrics)


if __name__ == "__main__":
    main()
