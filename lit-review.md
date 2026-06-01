# MAPF Deadlocks — Literature Review & Project Gap

**Project focus:** Explicit deadlock **detection + resolution** in decentralized *lifelong* MAPF.
**Base codebase:** [Follower / learn-to-follow](https://github.com/Cognitive-AI-Systems/learn-to-follow) (MIT) on [POGEMA](https://github.com/AIRI-Institute/pogema).
**Date:** 2026-06-01

---

## 1. Problem framing

- **MAPF** = collision-free paths for many agents to goals.
- **Lifelong MAPF (LMAPF)** = agents get a new goal the moment they reach the current one (warehouses, AGV fleets).
- **Deadlock/livelock** is the central failure mode and worsens with map density and agent count.

## 2. How our base (Follower) handles deadlocks today

Follower has **no explicit deadlock detection or resolution** — only *preventive* heuristics:
- **Global:** per-agent A* on an individual path, ignoring other agents.
- **Local:** a shared PPO policy follows waypoints, making detours to avoid collisions (no reward shaping).
- **Anti-stuck:** *dynamic cost* (A* penalizes cells frequently occupied by others) + *static cost* (penalize congestion-attractor cells); costs reset on goal arrival.

Stated limitations: assumes perfect localization, static map, synchronous moves; no completeness/optimality guarantees.

## 3. Key papers

| Paper | Mechanism | Deadlock handling | Scale tested | Opening for us |
|---|---|---|---|---|
| **Follower** (AAAI-24, base) — [repo](https://github.com/Cognitive-AI-Systems/learn-to-follow) | A* + PPO local policy | Preventive only (dynamic/static cost) | up to 256 agents, lifelong | No detection, no resolver — agents still jam |
| **Hybrid RL-MAPF** — [2511.22685](https://arxiv.org/abs/2511.22685) | RL + on-demand local Push-and-Rotate MAPF | **Explicit: 3-trigger detect + local resolve** | **only ≤8 agents, 2 toy maps, not lifelong** | Doesn't scale; no grid/lifelong eval |
| **PIBT** — [1901.11282](https://arxiv.org/abs/1901.11282) | Priority inheritance + backtracking, 1-step online | Finite-time arrival **only on biconnected graphs** | hundreds of agents | Guarantee fails exactly at dead-ends/corridors (= deadlock sites) |
| **Transient MAPF** — [2412.04256](https://arxiv.org/abs/2412.04256) | Relax simultaneous-arrival constraint | Documents where mitigations fail | classical MAPF | Not learning/decentralized |
| **Standby-Based DA** — [2201.06014](https://arxiv.org/pdf/2201.06014) | Articulation-point standby nodes + token | Detect via graph structure, wait at standby | small maze maps | Suboptimal, manual params, centralized token |
| **Highway / Guidance** — [2304.04217](https://arxiv.org/pdf/2304.04217), [2411.16506](https://arxiv.org/html/2411.16506v1) | Directional bias graphs | Reduce head-on deadlocks | medium | Congestion/suboptimality tradeoff |
| **RHCR** — [AAAI-21](https://cdn.aaai.org/ojs/17344/17344-13-20838-1-2-20210518.pdf) | Windowed centralized replanning | Implicit via replanning | scales poorly w/ agents | Runtime explodes at high density |

### Hybrid RL-MAPF (2511.22685) — the detection+resolution template
- **Detection** = union of 3 triggers: (1) speed/non-progress stalemate, (2) waypoint index stuck for budget `T_wp`, (3) "core-pair" reciprocal risk via time-to-collision. Gated by warm-up/cooldown.
- **Resolution** = crop subgrid around implicated agents → solve local MAPF with Push-and-Rotate → emit dense waypoints → same RL policy tracks them → resume global plan.
- **Weakness (our opening):** 2 toy maps (doorway, corridor), ≤8 agents, only RL-only baseline, no overhead analysis, not lifelong.

### PIBT (1901.11282) — the scalable resolver primitive
- One-timestep online planning; built for iterative/lifelong MAPF; scales to hundreds.
- **Completeness only on biconnected graphs** — warehouse/maze dead-ends are *not* biconnected, so the guarantee fails exactly where deadlocks form. Motivates a topology-aware detector. Prefer PIBT over Push-and-Rotate as the local resolver (online-native, scalable).

## 4. POGEMA benchmark — our metrics & baselines ([2407.14931](https://arxiv.org/abs/2407.14931))

- **Metrics:** Throughput (LMAPF), Success Rate, Makespan, **Coordination** = `1 − collisions/(agents×steps)`, **Cooperation** (5×5 "Puzzles", 2–4 agents), Scalability, OOD generalization.
- **No explicit deadlock-rate or recovery-time metric** → contribution slot for us.
- **Baselines (same maps/scales):** Follower (base), MATS-LP (planning+MCTS), RHCR & LaCAM (search), MAPF-GPT, MARL (QMIX/QPLEX/VDN).
- **Maps/scales:** Random & Mazes (17–21², 8–64 agents), Warehouse (33×46, 32–192), Puzzles (5×5, 2–4), Cities (up to 256×256, 1–256).

## 5. Project gap statement

> Add an **explicit deadlock detector** (3-trigger, adapted to grid/POGEMA) + a **scalable local resolver (PIBT on a cropped subgrid)** to Follower's decentralized lifelong pipeline. Detect specifically at the **non-biconnected topology** (dead-ends, narrow corridors) where Follower's cost heuristic saturates and PIBT loses its completeness guarantee. Evaluate on POGEMA at **hundreds of agents** — the scale the Hybrid RL-MAPF paper (≤8 agents, 2 toy maps) never reached — and report against POGEMA baselines (Follower, MATS-LP, RHCR, LaCAM) using a **new deadlock-rate + recovery-time metric** POGEMA currently lacks.

Why solid: **novel** (no baseline combines detection + PIBT-resolution in lifelong decentralized setting; fills POGEMA's missing deadlock metric), **grounded** (all components are maintained, permissive-licensed codebases — zero from-scratch), **measurable** (throughput + coordination + new deadlock metrics vs named baselines at named scales), **theoretically motivated** (detector targets exactly where PIBT's biconnectivity guarantee fails).

## 6. Sources

- Hybrid RL-MAPF — https://arxiv.org/abs/2511.22685
- Follower / learn-to-follow — https://github.com/Cognitive-AI-Systems/learn-to-follow
- PIBT — https://arxiv.org/abs/1901.11282 · winPIBT — https://arxiv.org/abs/1905.10149
- Transient MAPF — https://arxiv.org/abs/2412.04256
- Standby-Based Deadlock Avoidance — https://arxiv.org/pdf/2201.06014
- Highway LMAPF — https://arxiv.org/pdf/2304.04217 · Guidance Graph Opt — https://arxiv.org/html/2411.16506v1
- RHCR — https://cdn.aaai.org/ojs/17344/17344-13-20838-1-2-20210518.pdf
- POGEMA benchmark — https://arxiv.org/abs/2407.14931 · repo — https://github.com/AIRI-Institute/pogema
- SRMT lifelong shared memory — https://arxiv.org/html/2501.13200v1
