"""Congestion metering: cap the number of concurrently-active agents.

Over the throughput peak, adding agents *reduces* throughput. Metering keeps M
agents active; the rest "park". Two variants:

  MeterWrapper        : park = freeze in place (naive; fails, parked agents block
                        the arteries -> diagnostic baseline).
  ParkingMeterWrapper : park = route excess agents to LOW-betweenness cells (off
                        the arteries) and idle there -> clears the traffic network.

The static cap M is a proof-of-concept; the adaptive, deadlock-metric-driven
controller that picks M online is the actual contribution.
"""
from collections import deque

import gymnasium

from deadlock_resolver import MOVES


def _bfs_field(obst, goal):
    """Shortest-path distance from goal over free cells (dict cell->dist)."""
    h, w = obst.shape
    if obst[goal]:
        return {}
    dist = {goal: 0}
    dq = deque([goal])
    while dq:
        c = dq.popleft()
        nd = dist[c] + 1
        for dx, dy in MOVES[1:]:
            nc = (c[0] + dx, c[1] + dy)
            if 0 <= nc[0] < h and 0 <= nc[1] < w and not obst[nc] and nc not in dist:
                dist[nc] = nd
                dq.append(nc)
    return dist


class MeterWrapper(gymnasium.Wrapper):
    """Park excess agents by freezing them in place (naive baseline)."""

    def __init__(self, env, m):
        super().__init__(env)
        self.m = int(m)

    def step(self, action):
        action = list(action)
        for i in range(self.m, len(action)):
            action[i] = 0   # park: action 0 = stay
        return self.env.step(action)


class HideMeterWrapper(gymnasium.Wrapper):
    """Park excess agents OFF-GRID via POGEMA's hide_agent (frees their cell) — a
    depot. Tests whether the 'less is more' headroom is capturable when parked
    agents don't occupy navigation cells."""

    def __init__(self, env, m):
        super().__init__(env)
        self.m = int(m)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        n = len(self.grid.get_agents_xy())
        for i in range(self.m, n):
            self.grid.hide_agent(i)          # despawn -> off-grid depot
        return obs, info


class ParkingMeterWrapper(gymnasium.Wrapper):
    """Park excess agents by routing them to low-betweenness cells, then idling."""

    def __init__(self, env, m):
        super().__init__(env)
        self.m = int(m)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._setup()
        return obs, info

    def _setup(self):
        import topology
        obst = self.grid.get_obstacles()
        pos = [tuple(p) for p in self.grid.get_agents_xy()]
        n = len(pos)
        self._parked = list(range(self.m, n))
        # candidate parking spots = lowest-betweenness free cells (off the arteries)
        cand = topology.low_betweenness_cells(obst, k=max(1, len(self._parked) * 2))
        # greedy nearest assignment, distinct spots
        used = set()
        self._target = {}
        for a in self._parked:
            best, bd = None, 1e18
            ax, ay = pos[a]
            for c in cand:
                if c in used:
                    continue
                d = abs(c[0] - ax) + abs(c[1] - ay)
                if d < bd:
                    bd, best = d, c
            self._target[a] = best or pos[a]
            used.add(self._target[a])
        # one BFS field per distinct target (cached)
        self._field = {}
        for a in self._parked:
            t = self._target[a]
            if t not in self._field:
                self._field[t] = _bfs_field(obst, t)

    def step(self, action):
        action = list(action)
        pos = [tuple(p) for p in self.grid.get_agents_xy()]
        for a in self._parked:
            t = self._target[a]
            p = pos[a]
            if p == t:
                action[a] = 0                       # arrived -> idle
                continue
            f = self._field[t]
            cur = f.get(p, 1e18)
            best_move, best_d = 0, cur              # default: stay
            for ai, (dx, dy) in enumerate(MOVES[1:], start=1):
                d = f.get((p[0] + dx, p[1] + dy))
                if d is not None and d < best_d:
                    best_d, best_move = d, ai
            action[a] = best_move                   # step toward parking spot
        return self.env.step(action)
