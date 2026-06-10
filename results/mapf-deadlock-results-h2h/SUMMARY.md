# Head-to-head: density control vs baselines (reviewer S2)

Scenario: **den520d-384** (over-saturated; past the throughput peak), POGEMA
`on_target:restart`, soft collisions, horizon 512. **10 seeds** (0–9) per condition,
one episode each, run per-seed (incremental/resumable). Same fixed Follower base
for all conditions — only the density-control wrapper differs.

## Throughput (mean ± 95% bootstrap CI, paired bootstrap vs all-active)

| method | mean tp | 95% CI | Δ% vs active | P(>active) |
|---|---|---|---|---|
| all-active (baseline) | 1.926 | [1.864, 1.990] | — | — |
| congestion-routing (proxy) | 1.916 | [1.876, 1.958] | −0.5% | 0.41 |
| **static metering (off-grid depot, M=256)** | **2.273** | [2.216, 2.336] | **+18.1%** | **1.00** |
| **adaptive (closed-loop, self-tuning M)** | **2.081** | [2.029, 2.140] | **+8.1%** | **1.00** |

Bootstrap = 10,000 resamples; paired test resamples per-seed deltas (paired by seed).
Adaptive self-tuned to **M = 335 on average** (range 244–382 across seeds).

## Findings

1. **Static metering is a robust win (+18%)** — its CI [2.22, 2.34] does not overlap
   all-active; paired P=1.00. Headline result.
2. **Adaptive significantly beats all-active (+8%, non-overlapping CI, P=1.00)** but is
   weaker than the static oracle cap. Reason: it self-tunes to M≈335 on average, while
   the throughput peak is nearer the static cap (256), so its hill-climb stops shedding
   early and leaves throughput on the table — conservative but knowledge-free.
3. **The congestion-routing proxy does not help (−0.5%, P=0.41).** Naive rerouting
   around high-betweenness cells is insufficient; *where/how* excess agents are shed
   (off-grid depot) is what captures the headroom.

## Why 10 seeds mattered

3-seed pilot had adaptive at +15% and routing at +1%. With 10 seeds: adaptive →
**+8%** and routing → **−0.5%**. The pilot over-stated adaptive; the 10-seed CIs are
the numbers to cite.

## Reproduce

`notebooks/colab_baseline.ipynb` §8 (per-seed, resumable). Per condition:
`baseline_eval.py [flags] --seed=K 08-val --out=.../<cond>/seed{K}` where flags are
`(active: none) (routing: --route --deadlock) (static: --meter=256 --meter-mode=hide)
(adaptive: --meter=384 --meter-mode=adaptive)`. Adaptive requires `pogema_patch.py`
(cycle-safe move-revert) or the episode hangs.
