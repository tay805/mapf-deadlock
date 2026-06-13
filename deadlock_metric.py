"""Deadlock metric for POGEMA (lifelong) — a gymnasium Wrapper.

Measures, per episode, how much agents are *stuck* = making no progress toward
their current goal for >= T steps, where progress = reduction in BFS shortest-
path distance to the goal (robust on mazes/dead-ends, unlike Manhattan).

It mirrors pogema's metric wrappers (pogema/wrappers/metrics.py): it accumulates
per step and injects results into infos[0]['metrics'] at episode end, so it rides
the same channel as avg_throughput with no changes to the eval loop.

Emitted per-episode scalars:
  deadlock_rate_T{t} : agent-steps with no-progress-counter >= t, / total agent-steps
  deadlock_rate      : alias for the primary T
  mean_recovery_time : avg length of deadlock runs that end before the episode does
  unrecovered_rate   : fraction of deadlock runs still stuck at episode end
  deadlock_events_per_agent : number of deadlock runs / num_agents

Definition of the per-agent counter (O(1)/step):
  best_dist = closest BFS distance reached to the current goal so far.
  each step: d = dist(pos); if d < best_dist -> progress, counter=0; else counter+=1.
  on goal change (lifelong arrival) -> recompute, reset. on-goal -> counter=0.
"""
from collections import deque, OrderedDict

import numpy as np
from gymnasium import Wrapper

_CACHE_CAP = 1024  # max BFS distance fields cached per episode


def _bfs_dist(free, goal):
    """BFS shortest-path distance field from `goal` over FREE cells (4-connected).
    `free` is a bool array (True = traversable). Unreachable cells = -1."""
    h, w = free.shape
    dist = np.full((h, w), -1, dtype=np.int16)
    gx, gy = goal
    if not (0 <= gx < h and 0 <= gy < w) or not free[gx, gy]:
        return dist
    dist[gx, gy] = 0
    dq = deque([(gx, gy)])
    while dq:
        x, y = dq.popleft()
        nd = dist[x, y] + 1
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < h and 0 <= ny < w and free[nx, ny] and dist[nx, ny] < 0:
                dist[nx, ny] = nd
                dq.append((nx, ny))
    return dist


class DeadlockMetric(Wrapper):
    def __init__(self, env, thresholds=(5, 10, 20, 40), primary_t=10, topo=False):
        super().__init__(env)
        self.thresholds = tuple(sorted(set(thresholds) | {primary_t}))
        self.primary_t = primary_t
        self.topo = topo   # also report WHERE deadlocks happen vs map topology

    def reset(self, seed=None, **kwargs):
        obs, info = self.env.reset(seed=seed, **kwargs)
        self._setup()
        return obs, info

    # --- BFS distance fields (cached per goal cell, capped) ---
    def _dist_field(self, goal):
        f = self._cache.get(goal)
        if f is None:
            f = _bfs_dist(self._free, goal)
            if len(self._cache) >= _CACHE_CAP:
                self._cache.popitem(last=False)  # evict oldest
            self._cache[goal] = f
        else:
            self._cache.move_to_end(goal)
        return f

    def _dist(self, pos, goal):
        d = self._dist_field(goal)[pos[0], pos[1]]
        return None if d < 0 else int(d)

    def _setup(self):
        grid = self.grid
        self._free = (grid.get_obstacles() == grid.config.FREE)
        self._cache = OrderedDict()
        pos = grid.get_agents_xy()
        tgt = grid.get_targets_xy()
        n = len(pos)
        self._n = n
        self._goal = [tuple(tgt[i]) for i in range(n)]
        self._best = [self._dist(tuple(pos[i]), self._goal[i]) for i in range(n)]
        self._counter = np.zeros(n, dtype=np.int64)
        self._steps = 0
        self._dl_steps = {t: 0 for t in self.thresholds}
        self._in_dl = np.zeros(n, dtype=bool)
        self._run_len = np.zeros(n, dtype=np.int64)
        self._recovered = []
        self._unrecovered = 0
        self._heat = {}        # grid cell -> count of deadlocked (counter>=T) agent-steps

    def step(self, action):
        obs, rew, term, trunc, infos = self.env.step(action)
        self._update()
        if all(term) or all(trunc):
            infos[0].setdefault('metrics', {}).update(self._finalize())
        return obs, rew, term, trunc, infos

    def _update(self):
        self._steps += 1
        pos = self.grid.get_agents_xy()
        tgt = self.grid.get_targets_xy()
        on_goal = self.was_on_goal
        T = self.primary_t
        for i in range(self._n):
            g = tuple(tgt[i]); p = tuple(pos[i])
            if g != self._goal[i]:                 # new goal = lifelong arrival
                self._goal[i] = g
                self._best[i] = self._dist(p, g)
                self._counter[i] = 0
            else:
                d = self._dist(p, g)
                if d is None:                      # unreachable (shouldn't happen)
                    self._counter[i] += 1
                elif self._best[i] is None or d < self._best[i]:
                    self._best[i] = d
                    self._counter[i] = 0
                else:
                    self._counter[i] += 1
            if on_goal[i]:                         # on goal is never "stuck"
                self._counter[i] = 0

            c = self._counter[i]
            for t in self.thresholds:
                if c >= t:
                    self._dl_steps[t] += 1
            if c >= T:                             # recovery tracking at primary T
                self._heat[p] = self._heat.get(p, 0) + 1   # WHERE the deadlock is
                if not self._in_dl[i]:
                    self._in_dl[i] = True
                    self._run_len[i] = 0
                self._run_len[i] += 1
            elif self._in_dl[i]:
                self._recovered.append(int(self._run_len[i]))
                self._in_dl[i] = False

    def _finalize(self):
        self._unrecovered += int(self._in_dl.sum())
        denom = max(1, self._n * self._steps)
        out = {f'deadlock_rate_T{t}': self._dl_steps[t] / denom for t in self.thresholds}
        out['deadlock_rate'] = out[f'deadlock_rate_T{self.primary_t}']
        out['mean_recovery_time'] = float(np.mean(self._recovered)) if self._recovered else 0.0
        n_runs = len(self._recovered) + self._unrecovered
        out['unrecovered_rate'] = (self._unrecovered / n_runs) if n_runs else 0.0
        out['deadlock_events_per_agent'] = n_runs / max(1, self._n)
        # Deadlock-run-length distribution (S6 false-positive analysis): a run that
        # recovers within a few steps past the T budget is a transient detour/yield
        # (a candidate false positive); long/unrecovered runs are true jams. We also
        # report what share of all deadlock agent-steps comes from short vs long runs.
        r = np.array(self._recovered, dtype=float) if self._recovered else np.zeros(1)
        out['dl_run_median'] = float(np.median(r))
        out['dl_run_p90'] = float(np.percentile(r, 90))
        out['dl_run_frac_le5'] = float(np.mean(r <= 5))      # transient (yield/detour-like)
        out['dl_run_steps_in_short'] = float(r[r <= 5].sum())
        out['dl_run_steps_in_long'] = float(r[r > 5].sum())  # persistent share of dl-steps
        if self.topo:
            out.update(self._topo_concentration())
        return out

    def _topo_concentration(self):
        """Does deadlock concentrate at articulation points / corridor cells?
        lift = (fraction of deadlock-steps at feature) / (fraction of free cells
        that are feature). lift >> 1 => deadlock is predicted by static topology."""
        import topology
        obst = (~self._free).astype(np.int8)        # OBSTACLE where not free
        ap = topology.articulation_points(obst)
        cor = topology.corridor_cells(obst)
        btw = topology.high_betweenness_cells(obst)   # top-10% bottleneck cells
        free = max(1, int(self._free.sum()))
        tot = max(1, sum(self._heat.values()))

        def lift(cells):
            at = sum(c for cell, c in self._heat.items() if cell in cells)
            base = len(cells) / free
            return at / tot, base, (at / tot) / max(1e-9, base)

        a_f, a_b, a_l = lift(ap)
        c_f, c_b, c_l = lift(cor)
        b_f, b_b, b_l = lift(btw)
        return {
            'articulation_lift': a_l, 'free_articulation_frac': a_b,
            'corridor_lift': c_l, 'free_corridor_frac': c_b,
            'betweenness_lift': b_l, 'free_betweenness_frac': b_b,
            'deadlock_at_betweenness_frac': b_f,
        }
