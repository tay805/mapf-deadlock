"""Baseline POGEMA eval for Follower only (no wandb, no FollowerLite).

FollowerLite is skipped because its C++ module needs onnxruntime C++ headers
that are not installed. This runner has no wandb dependency (the toolbox's
save_evaluation_results imports wandb, so we inline the zip bundling instead).
Results + views are saved per-config into experiments/<folder>/ as each folder
completes, so partial progress survives an interruption.

Usage:
    python baseline_eval.py                      # all 5 folders, full seeds
    python baseline_eval.py 05-warehouse         # one or more specific folders
    python baseline_eval.py --seeds=3            # all folders, first 3 seeds only
    python baseline_eval.py --seeds=3 02-mazes   # combine
    python baseline_eval.py --out=/content/drive/MyDrive/mapf-deadlock-results
        # write results to <out>/<folder>/ (e.g. Google Drive) so they persist
        # instead of landing inside the repo's experiments/<folder>/
"""
import os
# Force headless matplotlib BEFORE pogema_toolbox imports it. Colab exports
# MPLBACKEND=<inline backend>, which the `conda run` subprocess inherits; that
# crashes the plot views with "No module named 'matplotlib_inline'". Hard-assign
# (not setdefault) to override the inherited value, then pin the backend.
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')

import shutil
import sys
from pathlib import Path

import yaml

from pogema import BatchAStarAgent
from pogema_toolbox.create_env import create_env_base, Environment
from pogema_toolbox.evaluator import evaluation
from pogema_toolbox.registry import ToolboxRegistry

from follower.inference import FollowerInference, FollowerInferenceConfig
from follower.preprocessing import follower_preprocessor

from deadlock_metric import DeadlockMetric
from deadlock_detector import (
    follower_preprocessor_with_detector, make_follower_preprocessor_with_detector,
)


def make_create_env_with_deadlock(topo=False):
    def create_env_with_deadlock(config):
        """create_env_base + DeadlockMetric (adds deadlock_rate etc. to
        infos[0]['metrics']; topo=True also reports topology concentration)."""
        return DeadlockMetric(create_env_base(config), topo=topo)
    return create_env_with_deadlock

BASE_PATH = Path('experiments')
ALL_FOLDERS = [
    '01-random-20x20',
    '02-mazes',
    '03-den520d',
    '04-Paris_1',
    '05-warehouse',
]


def main(folders, max_seeds=None, out_dir=None, deadlock=False, detector=False,
         resolve=False, resolve_t=30, resolve_k=1, topo=False):
    # --resolve implies --detector implies --deadlock so we always get the offline
    # ground-truth metric on the same episodes for comparison.
    detector = detector or resolve
    deadlock = deadlock or detector or topo
    env_factory = make_create_env_with_deadlock(topo=topo) if deadlock else create_env_base
    if resolve:
        preproc = make_follower_preprocessor_with_detector(
            resolve=True, resolve_t=resolve_t, resolve_k=resolve_k)
    elif detector:
        preproc = follower_preprocessor_with_detector
    else:
        preproc = follower_preprocessor
    ToolboxRegistry.register_env('Pogema-v0', env_factory, Environment)
    ToolboxRegistry.register_algorithm('A*', BatchAStarAgent)
    ToolboxRegistry.register_algorithm(
        'Follower', FollowerInference, FollowerInferenceConfig, preproc)

    with open('env/test-maps.yaml') as f:
        ToolboxRegistry.register_maps(yaml.safe_load(f))

    out_base = Path(out_dir) if out_dir else BASE_PATH
    for folder in folders:
        # Config is always read from the repo; results are written under out_base
        # (e.g. a Google Drive folder) so they persist across Colab sessions.
        config_path = BASE_PATH / folder / f'{folder}.yaml'
        eval_dir = out_base / folder
        eval_dir.mkdir(parents=True, exist_ok=True)
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
        # Bundle the folder's results into a zip (same as the toolbox helper, but
        # without its wandb.save() upload — keeps the runner wandb-free).
        shutil.make_archive(str(eval_dir), 'zip', eval_dir)
        print(f'>>> {folder}: DONE', flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    max_seeds = None
    out_dir = None
    deadlock = False
    detector = False
    resolve = False
    resolve_t = 30
    resolve_k = 1   # 1 = single-step (validated method); >1 = multi-step ablation
    topo = False
    folders = []
    i = 0
    while i < len(args):  # accept both "--opt=val" and "--opt val" forms
        a = args[i]
        if a.startswith('--seeds='):
            max_seeds = int(a.split('=', 1)[1])
        elif a == '--seeds':
            i += 1; max_seeds = int(args[i])
        elif a.startswith('--out='):
            out_dir = a.split('=', 1)[1]
        elif a == '--out':
            i += 1; out_dir = args[i]
        elif a == '--deadlock':
            deadlock = True
        elif a == '--detector':
            detector = True
        elif a == '--resolve':
            resolve = True
        elif a.startswith('--resolve-t='):
            resolve_t = int(a.split('=', 1)[1])
        elif a.startswith('--resolve-k='):
            resolve_k = int(a.split('=', 1)[1])
        elif a == '--topo':
            topo = True
        else:
            folders.append(a)
        i += 1
    main(folders or ALL_FOLDERS, max_seeds=max_seeds, out_dir=out_dir, deadlock=deadlock,
         detector=detector, resolve=resolve, resolve_t=resolve_t, resolve_k=resolve_k, topo=topo)
