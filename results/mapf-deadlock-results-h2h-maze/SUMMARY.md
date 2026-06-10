# Head-to-head on maze-640 (reviewer S2 + S4 second map)

Scenario: **maze-640** (`test-mazes-s40_wc4_od30`, 640 agents, over-saturated),
POGEMA `on_target:restart`, soft collisions, horizon 512. **10 seeds** (0–9) per
condition, one episode each, per-seed (incremental/resumable). Same fixed Follower
base; only the density-control wrapper differs. Static cap M=384 (near maze peak).

## Throughput (mean ± 95% bootstrap CI, paired bootstrap vs all-active)

| method | mean tp | 95% CI | Δ% vs active | P(>active) |
|---|---|---|---|---|
| all-active (baseline) | 3.230 | [3.167, 3.298] | — | — |
| congestion-routing (proxy) | 3.287 | [3.220, 3.349] | +1.7% | 0.93 (n.s.) |
| **static metering (off-grid depot, M=384)** | **3.534** | [3.492, 3.571] | **+9.4%** | **1.00** |
| **adaptive (closed-loop, self-tuning M)** | **3.382** | [3.311, 3.446] | **+4.7%** | **1.00** |

Bootstrap = 10,000 resamples; paired test resamples per-seed deltas. Adaptive
self-tuned to **M = 527 on average** (range 458–590) — conservative vs the static cap.

## Findings (mirror den520d)

1. **Static metering robustly wins (+9.4%, CI excludes all-active, P=1.00).**
2. **Adaptive significantly wins (+4.7%, P=1.00) but is conservative** — self-tunes to
   M≈527, well above the static optimum (384), capturing about half the static gain.
3. **Routing does NOT significantly help (+1.7%, P=0.93).** Even on the more-localized
   maze, rerouting around bottlenecks cannot recover throughput under over-saturation.

## Why 10 seeds mattered (vs pilot)

3-seed/1-seed pilot: routing +6%, static +9%, adaptive +9% (1 seed). At 10 seeds:
routing → **+1.7% (n.s.)**, static → +9.4% (held), adaptive → **+4.7%**. The pilot's
+6% routing was the sole support for a "routing wins when localized" sub-claim; it does
not survive 10 seeds. Core result (density control robustly beats routing under
over-saturation in both topologies) is unaffected and now solid.
