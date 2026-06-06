# Head-to-head: interventions on the same Follower base (over-saturated)

Δ = throughput vs all-active. Same base/harness/scenario; only intervention varies.

| map (scenario) | phase | all-active | routing | static metering | adaptive (self-tune) |
|---|---|---|---|---|---|
| den520d-384 (3 seeds) | delocalized (lift~1×) | 1.92 | 1.94 (+1%, FAILS) | 2.24 (+16%) | 2.21 (+15%) |
| mazes-640 (3 seeds; adaptive 1-seed) | localized (lift 1.9×, dl 44%) | 3.15 | 3.34 (+6%) | 3.42 (+9%) | 3.42 (+9%) |

Findings:
- Static density control = ROBUST (wins both maps).
- Routing = FRAGILE (fails in delocalized regime; helps only localized).
- Adaptive controller self-tunes to ~peak active count on both (den520d ~288 / peak 256; mazes ~372 / peak 384) without prior knowledge; matches static.
- Infra limits: warehouse caps at 192 spawns (can't over-saturate); pogema move-recursion segfaults at very high active density (≥~640).
