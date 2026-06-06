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
import sys
from collections import deque

import gymnasium

from deadlock_resolver import MOVES

sys.setrecursionlimit(30000)   # pogema _revert_action recurses per move-chain
                               # (deep at high agent density); set in-process here


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


class AdaptiveMeterWrapper(gymnasium.Wrapper):
    """Closed-loop density controller. Starts all-active and SELF-TUNES the number
    of active agents online to maximize throughput — no prior knowledge of the peak.
    Signal = recent throughput (hill-climb); when reducing, it hides the MOST
    DEADLOCKED agents (longest since reaching a goal) — directly removing the jam.
    Uses pogema hide_agent/show_agent (off-grid depot)."""

    def __init__(self, env, m0=None, ctrl_interval=20, step=48, m_min=32):
        super().__init__(env)
        self.m0 = m0
        self.ctrl_interval = ctrl_interval
        self.step_size = step
        self.m_min = m_min

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.n = len(self.grid.get_agents_xy())
        self.M = self.n if self.m0 is None else min(self.m0, self.n)
        self.hidden = set()
        self.since_goal = [0] * self.n
        self.t = 0
        self.win_goals = 0
        self.last_tp = None
        self.shedding = True     # monotone: only SHED agents (never reactivate)
        self.M_trace = []
        self._apply_M()
        return obs, info

    def _apply_M(self):
        # Hide the most-deadlocked active agents down to M; or reactivate from the
        # depot up to M (used to revert a one-step overshoot past the peak).
        active = [i for i in range(self.n) if i not in self.hidden]
        if len(active) > self.M:
            order = sorted(active, key=lambda i: -self.since_goal[i])
            for i in order[:len(active) - self.M]:
                if self.grid.hide_agent(i):
                    self.hidden.add(i)
        elif len(active) < self.M:
            need = self.M - len(active)
            for i in list(self.hidden):
                if need <= 0:
                    break
                try:
                    if self.grid.show_agent(i):
                        self.hidden.discard(i); need -= 1
                except KeyError:
                    pass                               # cell occupied -> skip

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self.t += 1
        wog = self.was_on_goal
        for i in range(self.n):
            if i in self.hidden:
                continue
            if wog[i]:
                self.since_goal[i] = 0
                self.win_goals += 1
            else:
                self.since_goal[i] += 1
        if self.t % self.ctrl_interval == 0:           # control update
            tp = self.win_goals / self.ctrl_interval
            if self.shedding:
                if self.last_tp is not None and tp < self.last_tp - 1e-9:
                    self.M = min(self.n, self.M + self.step_size)  # revert overshoot
                    self.shedding = False                          # ...and hold at peak
                elif self.M > self.m_min:
                    self.M -= self.step_size           # keep shedding while it helps
            self.last_tp = tp
            self.win_goals = 0
            self.M_trace.append(self.M)
            self._apply_M()
        if (all(term) or all(trunc)):
            info[0].setdefault('metrics', {})['final_active_M'] = self.M
            info[0]['metrics']['mean_active_M'] = (
                sum(self.M_trace) / len(self.M_trace) if self.M_trace else self.M)
        return obs, rew, term, trunc, info


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
