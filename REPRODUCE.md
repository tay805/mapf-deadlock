# Reproducing "Deadlock in Dense Lifelong MAPF"

This repo extends **Follower** (learn-to-follow) with a deadlock metric, a two-regime
spatial characterization, a detect-and-resolve ablation, and density control
(static + adaptive metering), plus a centralized **LaCAM** baseline. All experiments
run on POGEMA (lifelong, `on_target:restart`, POMAPF, soft collisions, horizon 512) on
top of the released Follower checkpoint (committed under `model/follower/`).

## 1. Environment

Native Python on Colab/Kaggle is 3.12, which has **no wheels** for the pinned MAPF stack,
so use **Python 3.10**.

**Local (CPU is fine; the pure-Python Follower needs no GPU):**
```bash
python3.10 -m venv .venv && . .venv/bin/activate
pip install --prefer-binary \
  pogema==1.3.0 pogema-toolbox==0.1.0 sample-factory==2.1.1 \
  torch==1.13.1 'numpy<=1.23.1' 'pandas<=1.4' 'pydantic<2' \
  pyyaml 'dask==2024.8.0' 'distributed==2024.8.0' loguru cppimport 'pybind11==2.13.1' \
  matplotlib seaborn tabulate
```
Notes: `docker/requirements.txt` has conflicting pins — use the curated list above.
`FollowerLite` is skipped (needs onnxruntime C++ headers); the pure-Python `Follower`
path compiles only `follower_cpp/planner.cpp` via cppimport (pybind11 + a compiler).

**Cloud:** `notebooks/colab_baseline.ipynb` (baselines/metric/head-to-head, py3.10 conda)
and `notebooks/lacam_baseline.ipynb` (the LaCAM centralized baseline — builds the C++
solver). Both auto-handle the py3.10 env and save results to Drive / `/kaggle/working`.

## 2. The runner — `baseline_eval.py`

Follower-only POGEMA grid eval (wandb-free, headless matplotlib). Reads
`experiments/<folder>/<folder>.yaml`, runs `evaluation()`, writes per-config
`Follower.json` + a plot. Resumable (per-folder save).

Flags (accept `--opt val` and `--opt=val`):

| flag | effect |
|---|---|
| `<folder> ...` | restrict to specific `experiments/` folders |
| `--seeds=N` | first N seeds only · `--seed=K` single seed K |
| `--out=DIR` | write results under `DIR/<folder>/` (e.g. Drive) |
| `--deadlock` | add the deadlock metric (rate/recovery/run-length, T∈{5,10,20,40}) |
| `--topo` | also report betweenness/articulation/corridor **lift** (two-regime) |
| `--detector` | online 3-trigger detector (implies `--deadlock`) |
| `--resolve [--resolve-t=T --resolve-k=K]` | detect-and-resolve via cropped PIBT (implies `--detector`) |
| `--route [--route-strength=S]` | congestion routing proxy (betweenness penalty in A*) |
| `--meter=M --meter-mode={freeze,park,hide,adaptive}` | density control: cap M active; `hide`=off-grid depot, `adaptive`=self-tuning |
| `--central-pibt` | quick centralized full-grid PIBT probe (see caveat in `centralized_pibt.py`) |

## 3. Reproduce each result

```bash
# Baseline throughput (all 5 families)            -> §13 of the paper
python baseline_eval.py

# Deadlock metric, full 10-seed                    -> the den520d "money plot"
python baseline_eval.py --deadlock

# Over-saturation curve (den520d 128..640, 3 seeds)
python baseline_eval.py --deadlock 06-den520d-sat

# Two-regime / betweenness lift across maps (S4)
python baseline_eval.py --topo 12-phase-den312d 13-phase-boston   # + den520d/Paris
python phase_aggregate.py results/.../Follower.json ...           # -> lift vs deadlock

# Detect-and-resolve ablation (override vs waypoint)
python baseline_eval.py --resolve 08-val

# Head-to-head on the over-saturated scenario (S2), per condition:
python baseline_eval.py                         --seed=K 08-val --out=OUT/active/seedK
python baseline_eval.py --route --deadlock      --seed=K 08-val --out=OUT/routing/seedK
python baseline_eval.py --meter=256 --meter-mode=hide     --seed=K 08-val --out=OUT/static/seedK
python baseline_eval.py --meter=384 --meter-mode=adaptive --seed=K 08-val --out=OUT/adaptive/seedK
#   adaptive also emits depot-wait (fairness/latency, N2). maze: 11-maze640, meter 384/640.

# Centralized LaCAM baseline (S3) — see notebooks/lacam_baseline.ipynb (builds liblacam.so)
```

Committed result bundles live under `results/` (e.g. `mapf-deadlock-results-h2h`,
`-h2h-maze`, `-phase`, `lacam-sat`), each with a `SUMMARY.md`. Bootstrap CIs + paired
tests are in the notebook (Section 8).

## 4. File map

| file | role |
|---|---|
| `baseline_eval.py` | the runner (above) |
| `deadlock_metric.py` | BFS-non-progress metric + run-length / topology lift |
| `deadlock_detector.py` / `deadlock_resolver.py` / `pibt.py` | detect-and-resolve pipeline |
| `topology.py` | articulation / corridor / betweenness (sampled Brandes) |
| `metering.py` | static (`hide`/`freeze`/`park`) + `adaptive` density control |
| `routing_baseline.py` | congestion-routing proxy (betweenness penalty in A*) |
| `centralized_pibt.py` / `lacam_run.py` | centralized probes / real LaCAM integration |
| `pogema_patch.py` | cycle-safe move-revert (needed for high-density / adaptive runs) |
| `phase_aggregate.py` | aggregate `--topo` runs into lift-vs-density |

POGEMA congestion is hard to even simulate (warehouse caps at 192 spawn cells; very high
density stresses move-revert — hence `pogema_patch.py`); `den520d` is the clean
over-saturation map.
