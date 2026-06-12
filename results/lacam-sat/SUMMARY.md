# LaCAM centralized baseline on den520d (reviewer S3)

Real LaCAM (Kei18/lacam3) run in our pogema-1.3.0 pipeline via `lacam_run.py` +
`GlobalObsWrapper` (injects the global obstacle map + positions + goals LaCAM reads,
from the grid). Same map / on_target:restart / soft collisions / 512 steps / throughput
metric as Follower, so directly comparable. **Partial sweep (10/15 instances).**

## Throughput vs agents

| agents | LaCAM (seeds) | Follower (3s) | Δ vs Follower | density control |
|---|---|---|---|---|
| 128 | 2.673 (3) | 2.23 | +20% | — |
| 256 | 4.941 (3) | 2.32 | +113% | — |
| 384 | 6.127 (2) | 1.92 | +219% | ~2.27 (metered) |
| 512 | 5.061 (1) | 1.56 | +224% | — |
| 640 | 0.584 (1) | 1.06 | −45% | 2.47 (metered to ~192) |

Missing instances (re-run to fill): n384_s1, n512_s0, n512_s2, n640_s1, n640_s2.

## Findings (a real reframe)

1. **Follower's collapse past 256 is mostly a COORDINATION gap, not a hard density
   ceiling.** Centralized LaCAM keeps scaling — 6.1 at 384 (~3× Follower) — so the map
   sustains far more throughput than the decentralized policy extracts.
2. **There is still an extreme-density ceiling:** at 640 even LaCAM collapses to 0.58
   (1 seed; likely its planning time-limit — centralized runtime explodes with density).
3. **Density control wins at extreme over-saturation:** at 640, metering to ~192 active
   gives 2.47 — beating LaCAM (0.58) and all-active (1.06). Elsewhere it is a cheap
   *decentralized* partial recovery of the coordination gap.

## Caveats / TODO
- 512/640 are 1-seed; the "even LaCAM collapses at 640" claim needs ≥3 seeds (640 may
  be a timeout artifact). Re-run the missing instances.
- Sanity-check our LaCAM throughput against POGEMA's published LaCAM lifelong numbers to
  confirm the integration (expect same ballpark; LaCAM is typically the strongest).
- This CONTRADICTS the earlier centralized-PIBT-probe conclusion ("centralized also
  collapses") — that probe was the broken/budget-capped PIBT; real LaCAM does not
  collapse until 640. Repositions density control accordingly.
