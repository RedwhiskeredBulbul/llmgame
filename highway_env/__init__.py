"""Project simulator: only the four supported driving environments."""

import os
from gymnasium.envs.registration import register, registry

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
__version__ = "1.10.2"

ENVIRONMENTS = {
    "CutinEnv": "highway_env.envs.cutin_env:CutinEnv",
    "HighwayEnv": "highway_env.envs.highway_env:HighwayEnv",
    "AdversarialMergeEnv": "highway_env.envs.adversarial_merge_env:AdversarialMergeEnv",
    "MergeEnv": "highway_env.envs.merge_env:MergeEnv",
}


def register_envs():
    for env_id, entry_point in ENVIRONMENTS.items():
        if env_id not in registry:
            register(id=env_id, entry_point=entry_point)


register_envs()
