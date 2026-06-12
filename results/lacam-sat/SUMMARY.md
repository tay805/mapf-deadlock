# LaCAM centralized baseline on den520d (reviewer S3)

Real LaCAM (Kei18/lacam3) run in our pogema-1.3.0 pipeline via `lacam_run.py` +
`GlobalObsWrapper` (injects the global obstacle map + positions + goals LaCAM reads,
from the grid). Same map / on_target:restart / soft collisions / 512 steps / throughput
metric as Follower → directly comparable.

## Throughput vs agents

| agents | LaCAM | seeds | Follower (3s) | Δ vs Follower |
|---|---|---|---|---|
| 128 | 2.673 ± 0.04 | 3 | 2.23 | +20% |
| 256 | 4.941 ± 0.02 | 3 | 2.32 | +113% |
| 384 | 6.021 ± 0.15 | 3 | 1.92 | +214% |
| 512 | 5.061 | 1/3 (2 FAILED) | 1.56 | +224% |
| 640 | 0.584 | 1/3 (2 FAILED) | 1.06 | −45% |

Density control (decentralized, metering den520d-640 present): ~192 active → **2.47**.

## Findings — a three-regime reframe

1. **Follower's collapse past 256 is mostly a COORDINATION gap, not a hard density
   ceiling.** Centralized LaCAM (global info) extracts 2–3× more — 6.0 at 384 vs
   Follower 1.9 — so the map sustains far more throughput than the decentralized policy
   reaches. Solid at 3 seeds (128–384).
2. **Centralized planning becomes intractable at extreme density.** At 512 only 1/3
   seeds solved within LaCAM's time limit (2 failed); at 640 only 1/3 (2 failed), and
   the one that solved collapsed to 0.58 (< all-active 1.06). LaCAM's runtime explodes.
3. **Density control wins at extreme over-saturation:** at 640, metering to ~192 active
   gives 2.47 — beating LaCAM (0.58, mostly fails) and all-active (1.06). Elsewhere it
   is a cheap *decentralized* partial recovery of the coordination gap.

## Positioning (honest)
Density control is not "the" fix for a fundamental capacity limit — a centralized
planner does much better at moderate over-saturation. It is a **lightweight,
decentralized, practical** lever (no global solver, ~1.04× runtime) that recovers part
of the coordination gap, and the **only effective option at extreme over-saturation**
where centralized planning is intractable. This honestly answers the reviewer's
centralized-baseline request and supersedes the earlier broken-PIBT-probe claim
("centralized also collapses").
