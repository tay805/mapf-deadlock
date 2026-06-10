"""Aggregate betweenness-lift vs agent count from --topo phase runs.

Reads the per-folder Follower.json produced by `baseline_eval.py --topo` and prints,
per map, mean betweenness_lift and deadlock_rate by agent count (across seeds), plus
pgfplots-ready coordinates for the crossover figure (S4). Usage:

    python phase_aggregate.py /tmp/phase/den312d/12-phase-den312d/Follower.json [...]
"""
import json
import sys
from collections import defaultdict


def load(path):
    rows = json.load(open(path))
    by_n = defaultdict(lambda: {'lift': [], 'dl': []})
    name = None
    for r in rows:
        gs = r['env_grid_search']
        n = gs['num_agents']
        name = r.get('map_name') or gs.get('map_name') or name
        m = r['metrics']
        if 'betweenness_lift' in m:
            by_n[n]['lift'].append(m['betweenness_lift'])
            by_n[n]['dl'].append(m.get('deadlock_rate', float('nan')))
    return name, by_n


def main(paths):
    for p in paths:
        name, by_n = load(p)
        print(f"\n=== {name or p} ===")
        coords = []
        for n in sorted(by_n):
            lift = by_n[n]['lift']
            dl = by_n[n]['dl']
            ml = sum(lift) / len(lift)
            md = sum(dl) / len(dl)
            print(f"  {n:>4} agents: lift={ml:.2f}  deadlock={md:.3f}  (n={len(lift)})")
            coords.append(f"({n},{ml:.2f})")
        print("  pgfplots:", "".join(coords))


if __name__ == '__main__':
    main(sys.argv[1:])
