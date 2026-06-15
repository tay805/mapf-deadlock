# Real routing baseline: Guided-PIBT (reviewer S5)

Validates our betweenness-penalty **routing proxy** against the real method it stands
for: **Guided-PIBT** = the official implementation of Chen et al. 2023 ("Traffic Flow
Optimisation for Lifelong MAPF", our `trafficflow` cite). Built twice (traffic-flow
guidance **ON** vs **OFF** = plain PIBT) and run on its bundled **den520d** lifelong
benchmarks (the same map family as our paper), seed 0, 500 timesteps.
Throughput = `numTaskFinished / makespan`.

## Throughput vs agents (ON = routing, OFF = plain PIBT)

| agents | ON | OFF | gain |
|---|---|---|---|
| 2000 | 7.45 | 7.40 | +0.7% |
| 4000 | 10.95 | 11.14 | −1.7% |
| 6000 | 12.95 | 12.86 | +0.7% |
| 8000 | 13.46 | 13.43 | +0.2% |
| 10000 | 13.59 | 13.51 | +0.6% |
| 12000 | 13.76 | 14.07 | −2.2% |

## Finding — the proxy is faithful

**Real traffic-flow routing yields no significant throughput gain over plain PIBT**
(−2.2% to +0.7%, all within noise). This matches our betweenness proxy's result
(routing −0.5% on den520d-384) and the maze proxy (+1.7%, n.s.): **congestion routing
does not recover throughput** — exactly what the proxy reports. The ON/OFF builds differ
(numbers aren't identical), so guidance is active; it just doesn't help here.

## Honest caveats
- We tested the **traffic-heuristic** variant (`GUIDANCE=OFF FLOW_GUIDANCE=ON OBJECTIVE=2`,
  the "THv" config) — the closest analogue to our betweenness penalty. The full
  Guided-PIBT (guide-path + LNS) may help more, especially on structured warehouse maps.
- Guided-PIBT's `den520d` is the full **256×256** map at **2k–12k** agents in its own
  **centralized** harness — a different regime than our POGEMA (decentralized, 64×64
  den520d, 32–640 agents). Throughput here keeps **rising** (no collapse) → consistent
  with the LaCAM coordination-gap finding (centralized planning handles high density).
- Paris benchmark is broken upstream (SIGFPE); den520d/warehouse/sortation/room work.

## Build notes (notebooks/guided_pibt_baseline.ipynb)
Boost 1.91 + CMake 4.x CONFIG mode fails on header-only `boost_system`; pin Boost 1.83 +
CMake 3.28 in a dedicated conda env + classic FindBoost (MODULE). Build with the conda
compiler (Boost.Log ABI), run with `LD_LIBRARY_PATH` to conda libs, `--simulationTime` > 0
(default −1 → SIGFPE).
