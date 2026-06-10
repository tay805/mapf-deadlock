"""Centralized PIBT baseline (reviewer S3: a centralized planner for context).

POGEMA/pogema-toolbox ship no centralized MAPF solver (their only built-in A* is a
DECOUPLED per-agent reactive planner). We add full-grid, one-step PIBT --- the
centralized primitive underneath LaCAM and the MAPF-competition winners --- run with
global state every step. Full LaCAM/RHCR is left to future work.

This is an env WRAPPER, not a policy: it ignores the incoming (policy) action and
overwrites it with PIBT's joint move computed from the true global state
(`grid.get_agents_xy/get_targets_xy/get_obstacles`). So whatever nominal algorithm is
registered (we use a trivial stay-agent), the environment actually executes
centralized PIBT. Goal BFS fields are cached across steps (the map is static), giving
~0.1--0.2 s/step instead of the ~1 s cold cost.

Scientific point: if even a centralized planner with perfect global information
collapses at over-saturation (and cannot match density control), the bottleneck is
DENSITY, not decentralization --- which is exactly our thesis.

CAVEAT (why this is a quick probe, not the paper's centralized baseline): this minimal
PIBT is a crop-resolver primitive. Run full-grid at high density its priority-
inheritance backtracking thrashes; the `max_calls` cap below bounds per-step cost but
makes the high-density throughput budget-sensitive (non-monotonic), so it is NOT a
faithful centralized solver there. Stable and informative only up to ~the peak (e.g.
den520d@256: 2.78, beating Follower's 2.32). The paper's centralized baseline is the
REAL LaCAM run via notebooks/lacam_baseline.ipynb (reviewer S3).
"""
import gymnasium

from pibt import pibt_solve

# POGEMA action index <-> (dx, dy); matches deadlock_resolver / grid_config.MOVES.
MOVES = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_OF = {m: i for i, m in enumerate(MOVES)}


class CentralizedPIBTWrapper(gymnasium.Wrapper):
    """Override actions with a full-grid one-step PIBT joint plan each step."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obst_arr = self.grid.get_obstacles()
        h, w = obst_arr.shape
        self._bounds = (0, 0, h - 1, w - 1)
        self._obst = {(x, y) for x in range(h) for y in range(w) if obst_arr[x, y]}
        self._fields = {}          # goal cell -> BFS distance field (cached, map is static)
        self._n = len(self.grid.get_agents_xy())
        return obs, info

    def step(self, action):
        pos = {i: tuple(p) for i, p in enumerate(self.grid.get_agents_xy())}
        goals = {i: tuple(g) for i, g in enumerate(self.grid.get_targets_xy())}
        nxt = pibt_solve(pos, goals, self._obst, self._bounds, fields=self._fields,
                         max_calls=40 * self._n)   # cap backtracking so a dense step can't thrash
        pibt_action = [0] * self._n
        for a in range(self._n):
            c = nxt.get(a)
            if c is None:
                continue
            d = (c[0] - pos[a][0], c[1] - pos[a][1])
            pibt_action[a] = ACTION_OF.get(d, 0)
        return self.env.step(pibt_action)


class StayAgent:
    """Trivial nominal algorithm; its actions are discarded by the wrapper."""

    def act(self, observations):
        return [0] * len(observations)

    def reset_states(self):
        pass
