# Two-regime spatial structure across maps (reviewer S4)

Betweenness **lift** = (fraction of stuck agent-steps at top-decile betweenness cells)
/ (those cells' share of free space). lift ≫ 1 = deadlock localized at bottlenecks;
lift ≈ 1 = delocalized. Run with `baseline_eval.py --topo` (adds `betweenness_lift` to
the per-episode metrics). 3 seeds per (map, agent count). Aggregate with
`phase_aggregate.py`.

## Lift and deadlock vs agent count (3 seeds)

**Game maps (saturate):**
| map | 64 | 128 | 256 | 384 |
|---|---|---|---|---|
| den312d lift | 8.04 | 5.03 | 3.06 | 2.42 |
| den312d deadlock | 0.001 | 0.141 | 0.381 | 0.521 |
| den520d lift¹ | 1.8 | 3.1 | 1.1 | — |
| den520d deadlock¹ | 0.001 | 0.017 | 0.323 | — |

**City maps:**
| map | 64 | 128 | 256 | 384 |
|---|---|---|---|---|
| Boston lift | 6.82 | 4.01 | 2.09 | 1.85 |
| Boston deadlock | 0.033 | 0.070 | 0.241 | 0.385 |
| Paris lift¹ | 6.4 | 4.8 | 3.2 | — |
| Paris deadlock¹ | 0.015 | 0.024 | 0.057 | — |

¹ den520d/Paris from the earlier 3-seed phase runs (paper Fig. 2).

## Finding

On **every** map, betweenness lift falls as deadlock intensifies: ~5–8× when deadlock
is rare (localized at bottlenecks) → ~1–3× at saturation (delocalized). The trend is
shared by **both game maps and both city maps** — the controlling variable is
**deadlock severity / saturation, not map topology family**. den520d delocalizes most
completely (1.1× at 32% deadlock); Paris never saturates (<6% deadlock) so stays
localized.

**Important:** this *refutes* a clean "game maps delocalize, city maps stay localized"
split — den312d (game) holds lift 2.4× even at 52% deadlock, while Boston (city)
delocalizes to 1.85×. So the paper plots lift vs **deadlock rate** (normalizes map
size), showing the saturation-driven regime shift is robust across 4 maps. Points
below 1% deadlock are omitted (lift unreliable with few stuck events).
