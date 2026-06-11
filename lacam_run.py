"""Run real LaCAM (Kei18/lacam3, via pogema-benchmark's inference.py) inside OUR
working pogema-1.3.0 pipeline — the centralized baseline for reviewer S3.

WHY NOT pogema-benchmark's own pipeline: its main-branch env code imports
`AgentsDensityWrapper` (present only in pogema 1.3.x), while its solver/run_episode
needs a pogema whose observations carry `global_obstacles` (present only in the alpha
pogema, which LACKS AgentsDensityWrapper). No single pip-installable pogema satisfies
both — only their prebuilt Docker image does, which Colab/Kaggle can't run. So we use
our proven stack (pogema 1.3.0 + pogema-toolbox 0.1.0 + pydantic v1, the Follower env)
and inject the three global keys LaCAM reads from the grid via GlobalObsWrapper.
LaCAM's `inference.py` imports cleanly with this stack and builds `liblacam.so` at
import time (needs cmake + the lacam3 submodule, both present in the notebook).

Run from `pogema-benchmark/algorithms/` (so `lacam` is importable and the default
`lacam_lib_path="lacam/liblacam.so"` resolves):

    python lacam_run.py <N> <seed> <out_dir> <max_steps> <maps_yaml>
"""
import os
os.environ['MPLBACKEND'] = 'Agg'      # before pogema_toolbox imports matplotlib
import sys
from pathlib import Path

import yaml
import gymnasium
from pogema_toolbox.evaluator import evaluation
from pogema_toolbox.registry import ToolboxRegistry
from pogema_toolbox.create_env import create_env_base, Environment
from lacam.inference import LacamInference, LacamInferenceConfig


class GlobalObsWrapper(gymnasium.Wrapper):
    """Add the global keys LaCAM reads (our POMAPF obs are local-only), from the grid.
    obstacles, positions and targets all come from the same grid frame, so they are
    mutually consistent (verified: agents land on free cells of global_obstacles)."""

    def _inject(self, obs):
        obst = self.grid.get_obstacles()
        xy = self.grid.get_agents_xy()
        txy = self.grid.get_targets_xy()
        for i, o in enumerate(obs):
            o['global_obstacles'] = obst
            o['global_xy'] = xy[i]
            o['global_target_xy'] = txy[i]
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._inject(obs), info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        return self._inject(obs), rew, term, trunc, info


# Some inference builds omit reset_states (the toolbox's run_episode calls it).
if not hasattr(LacamInference, 'reset_states'):
    LacamInference.reset_states = lambda self: setattr(self, 'lacam_agents', None)


def main():
    N, seed, out, steps, maps = (int(sys.argv[1]), int(sys.argv[2]), sys.argv[3],
                                 int(sys.argv[4]), sys.argv[5])
    ToolboxRegistry.register_env('Pogema-v0', create_env_base, Environment)
    ToolboxRegistry.register_algorithm(
        'LaCAM', LacamInference, LacamInferenceConfig,
        lambda env, cfg: GlobalObsWrapper(env))
    ToolboxRegistry.register_maps(yaml.safe_load(open(maps)))
    cfg = {
        'environment': {
            'name': 'Pogema-v0', 'collision_system': 'soft', 'on_target': 'restart',
            'observation_type': 'POMAPF', 'max_episode_steps': steps,
            'map_name': 'den520d',
            'num_agents': {'grid_search': [N]}, 'seed': {'grid_search': [seed]},
        },
        'algorithms': {'LaCAM': {
            'name': 'LaCAM', 'num_process': 1, 'parallel_backend': 'sequential',
            'preprocessing': 'globalobs',   # truthy -> evaluator applies GlobalObsWrapper
        }},
    }
    Path(out).mkdir(parents=True, exist_ok=True)
    evaluation(cfg, eval_dir=Path(out))
    print('DONE', out)


if __name__ == '__main__':
    main()
