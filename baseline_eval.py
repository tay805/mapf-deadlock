"""Baseline POGEMA eval for Follower only (no wandb, no FollowerLite).

FollowerLite is skipped because its C++ module needs onnxruntime C++ headers
that are not installed. wandb is disabled. Results + views are saved per-config
into experiments/<folder>/ as each folder completes, so partial progress
survives an interruption.

Usage:
    python baseline_eval.py                      # all 5 folders, full seeds
    python baseline_eval.py 05-warehouse         # one or more specific folders
    python baseline_eval.py --seeds=3            # all folders, first 3 seeds only
    python baseline_eval.py --seeds=3 02-mazes   # combine
"""
import sys
from pathlib import Path

import yaml

from pogema import BatchAStarAgent
from pogema_toolbox.create_env import create_env_base, Environment
from pogema_toolbox.evaluator import evaluation
from pogema_toolbox.eval_utils import save_evaluation_results
from pogema_toolbox.registry import ToolboxRegistry

from follower.inference import FollowerInference, FollowerInferenceConfig
from follower.preprocessing import follower_preprocessor

BASE_PATH = Path('experiments')
ALL_FOLDERS = [
    '01-random-20x20',
    '02-mazes',
    '03-den520d',
    '04-Paris_1',
    '05-warehouse',
]


def main(folders, max_seeds=None):
    ToolboxRegistry.register_env('Pogema-v0', create_env_base, Environment)
    ToolboxRegistry.register_algorithm('A*', BatchAStarAgent)
    ToolboxRegistry.register_algorithm(
        'Follower', FollowerInference, FollowerInferenceConfig, follower_preprocessor)

    with open('env/test-maps.yaml') as f:
        ToolboxRegistry.register_maps(yaml.safe_load(f))

    for folder in folders:
        config_path = BASE_PATH / folder / f'{folder}.yaml'
        eval_dir = BASE_PATH / folder
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Keep only Follower-based algorithms (drop FollowerLite, which won't build).
        cfg['algorithms'] = {
            k: v for k, v in cfg['algorithms'].items() if v.get('name') != 'FollowerLite'
        }
        # Optionally limit to the first N seeds for a fast first-pass baseline.
        if max_seeds is not None:
            seeds = cfg['environment'].get('seed')
            if isinstance(seeds, dict) and 'grid_search' in seeds:
                seeds['grid_search'] = seeds['grid_search'][:max_seeds]
        print(f'>>> {folder}: algorithms {list(cfg["algorithms"])}, '
              f'seeds {cfg["environment"].get("seed")}', flush=True)
        evaluation(cfg, eval_dir=eval_dir)
        save_evaluation_results(eval_dir)
        print(f'>>> {folder}: DONE', flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    max_seeds = None
    folders = []
    for a in args:
        if a.startswith('--seeds='):
            max_seeds = int(a.split('=', 1)[1])
        else:
            folders.append(a)
    main(folders or ALL_FOLDERS, max_seeds=max_seeds)
