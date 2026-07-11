<div align="center">

# Deadlocks in Dense Lifelong MAPF
### A Journey from Why to Adaptive Density Control

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Benchmark: POGEMA](https://img.shields.io/badge/benchmark-POGEMA-green.svg)](https://github.com/AIRI-Institute/pogema)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)

</div>

In dense lifelong Multi-Agent Path Finding (MAPF), throughput rises with agent
density, reaches a peak, and then collapses. Throughput on its own reports that a
system has stalled without explaining why. This repository turns that failure
into something measurable and then acts on it, through a three-stage pipeline
that measures deadlock, locates it in space, and meters the number of active
agents online.

This is the code and data release for the paper *Deadlocks in Dense Lifelong
MAPF: A Journey from Why to Adaptive Density Control*.

## Contributions

1. **A deadlock metric for learned lifelong MAPF.** A per-agent stuck signal
   built from breadth-first non-progress, reported as a deadlock rate, a
   recovery time, an unrecovered fraction, and a per-cell spatial heatmap. It is
   implemented as a policy-agnostic wrapper and validated causally against a
   known jamming perturbation.
2. **A two-regime characterization.** A betweenness-lift order parameter that
   shows deadlock concentrating at a few bottlenecks below saturation and
   dispersing across the workspace once the system saturates. The transition is
   driven by congestion severity rather than by map topology and holds across
   four maps.
3. **A mechanistic negative result.** In the delocalized regime, both
   detect-and-resolve and congestion-aware routing fail to recover throughput,
   validated against a released Guided-PIBT solver and set against a centralized
   LaCAM reference.
4. **Adaptive Density Control (ADC).** A closed-loop controller that self-tunes
   the active-agent count online from the same deadlock signal, remaining
   effective where local repair and centralized search are not.

## Installation

The project targets Python 3.10. Create a virtual environment and install the
dependencies:

```bash
python3.10 -m venv .venv && . .venv/bin/activate
pip install --prefer-binary \
  pogema==1.3.0 pogema-toolbox==0.1.0 sample-factory==2.1.1 \
  torch==1.13.1 'numpy<=1.23.1' 'pandas<=1.4' 'pydantic<2' \
  pyyaml 'dask==2024.8.0' 'distributed==2024.8.0' loguru cppimport 'pybind11==2.13.1' \
  matplotlib seaborn tabulate
```

The base policy runs on CPU and compiles `follower_cpp/planner.cpp` through
cppimport, which needs pybind11 and a C++ compiler. The `notebooks/` folder
holds Colab and Kaggle runners for the same experiments.

To restrict NumPy CPU threads and avoid a common slowdown:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

## Usage

`baseline_eval.py` is the runner. It reads `experiments/<folder>/<folder>.yaml`,
runs the grid, and writes a per-config result JSON plus a plot. Runs are
per-folder resumable.

```bash
# Baseline throughput across the map families
python baseline_eval.py

# Deadlock metric (rate, recovery, run length, thresholds T in {5,10,20,40})
python baseline_eval.py --deadlock

# Two-regime betweenness lift across maps
python baseline_eval.py --topo 12-phase-den312d 13-phase-boston

# Detect-and-resolve ablation
python baseline_eval.py --resolve 08-val

# Density control: static cap and adaptive self-tuning
python baseline_eval.py --meter=256 --meter-mode=hide     08-val
python baseline_eval.py --meter=384 --meter-mode=adaptive 08-val
```

Run `python baseline_eval.py --help` for the full flag table.

## Repository layout

| Path | Role |
|---|---|
| `baseline_eval.py` | Evaluation runner and command-line flags |
| `deadlock_metric.py` | Breadth-first non-progress metric, run length, topology lift |
| `deadlock_detector.py`, `deadlock_resolver.py`, `pibt.py` | Detect-and-resolve pipeline |
| `topology.py` | Articulation, corridor, and betweenness analysis |
| `metering.py` | Static (`hide`, `freeze`, `park`) and `adaptive` density control |
| `routing_baseline.py` | Congestion-routing proxy, a betweenness penalty in A* |
| `centralized_pibt.py`, `lacam_run.py` | Centralized probes and the LaCAM baseline |
| `pogema_patch.py` | Cycle-safe move resolution for high-density runs |
| `phase_aggregate.py` | Aggregates `--topo` runs into lift versus density |
| `experiments/` | Scenario configurations |
| `results/` | Committed result bundles, each with a summary |
| `notebooks/` | Colab and Kaggle runners |

## Hardware and software

The experiments were run on Google Colab using an NVIDIA T4 GPU with a 32 GB
high-RAM runtime, with local development on a MacBook M1 Pro with 16 GB of RAM.
All code targets Python 3.10. The base policy is pure Python and runs on CPU, so
a GPU is optional for evaluation and is used mainly to accelerate the centralized
and notebook baselines.

## Base policy and benchmark

All experiments run on [POGEMA](https://github.com/AIRI-Institute/pogema) in
lifelong mode (`on_target:restart`, `POMAPF` observations, soft collisions,
horizon 512). The intervention wraps a single fixed decentralized base policy,
and every comparison changes only the wrapper, which removes the cross-codebase
confounds common in MAPF evaluation. The base checkpoint is committed under
`model/`.

## License

Released under the MIT License. See [LICENSE](LICENSE).
