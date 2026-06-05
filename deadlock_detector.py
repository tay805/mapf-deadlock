"""Online deadlock detector for Follower (P3).

Runs INSIDE Follower's decision loop (subclasses FollowerWrapper, right after the
A* planner produces per-agent paths), so it is decentralized and uses only what
each agent already computes. Unlike deadlock_metric.py (which observes from the
sidelines to *measure* deadlock), this detector *flags* stuck agents online so a
resolver (P4) can act on them.

Three triggers (union), gated by a warm-up:
  T1 non-progress : shortest-path distance to goal (= planner path length) hasn't
                    decreased for >= t1 steps. (Same idea as the metric; here the
                    distance comes from the planner path, no extra BFS.)
  T2 stall        : agent's cell is unchanged for >= t2 steps and it's not on goal.
  T3 face-off     : agent's next waypoint is another agent's cell AND that agent's
                    next waypoint is this agent's cell (mutual swap / head-on).

Emits per-episode rates into infos[0]['metrics'] so a run can compare the online
detector against the offline deadlock_rate on the SAME episodes:
  detector_flag_rate, trigger_nonprogress_rate, trigger_stall_rate,
  trigger_faceoff_rate  (each = flagged agent-steps / total agent-steps).
"""
import numpy as np

from follower.preprocessing import (
    FollowerWrapper, CutObservationWrapper, ConcatPositionalFeatures,
)
from deadlock_resolver import JamStats, resolve_jams, resolve_jams_paths, MOVES


class DeadlockDetector:
    def __init__(self, t1=10, t2=8, warmup=10, resolve_t=30):
        self.t1 = t1            # non-progress budget (detection / flagging)
        self.t2 = t2            # stall budget
        self.warmup = warmup
        self.resolve_t = resolve_t  # stricter budget: only RESOLVE deeply-stuck agents
        self.deep_flagged = None
        self.n = None

    def reset(self, n):
        self.n = n
        self._goal = [None] * n
        self._best = [None] * n          # best (smallest) path length to current goal
        self._np_counter = np.zeros(n, dtype=np.int64)   # non-progress
        self._last_pos = [None] * n
        self._stall = np.zeros(n, dtype=np.int64)        # stall (unchanged cell)
        self._step = 0
        self._tot = 0
        self._flagged = 0
        self._np = 0
        self._st = 0
        self._fo = 0

    def step(self, positions, goals, next_wp, dists, on_goal):
        """positions/goals/next_wp: list of (x,y); dists[i]=path length to goal or
        None (unreachable/at goal); on_goal[i]: bool. Returns bool flag array."""
        self._step += 1
        n = self.n
        trig_np = np.zeros(n, dtype=bool)
        trig_st = np.zeros(n, dtype=bool)
        trig_fo = np.zeros(n, dtype=bool)

        for i in range(n):
            g, p, d = goals[i], positions[i], dists[i]
            # T1: non-progress toward goal
            if g != self._goal[i]:                 # new goal -> reset
                self._goal[i] = g
                self._best[i] = d
                self._np_counter[i] = 0
            elif on_goal[i]:
                self._np_counter[i] = 0
            elif d is None:                        # no path = not progressing
                self._np_counter[i] += 1
            elif self._best[i] is None or d < self._best[i]:
                self._best[i] = d
                self._np_counter[i] = 0
            else:
                self._np_counter[i] += 1
            # T2: stall (cell unchanged) while not on goal
            if on_goal[i]:
                self._stall[i] = 0
            elif self._last_pos[i] is not None and p == self._last_pos[i]:
                self._stall[i] += 1
            else:
                self._stall[i] = 0
            self._last_pos[i] = p

        # T3: mutual face-off (A wants B's cell and B wants A's cell)
        pos_to_idx = {positions[i]: i for i in range(n)}
        for i in range(n):
            j = pos_to_idx.get(next_wp[i])
            if j is not None and j != i and next_wp[j] == positions[i]:
                trig_fo[i] = True

        if self._step > self.warmup:
            trig_np = self._np_counter >= self.t1
            trig_st = self._stall >= self.t2
            # "deeply stuck" = candidates the resolver may act on (conservative).
            self.deep_flagged = (self._np_counter >= self.resolve_t) | (self._stall >= self.resolve_t)
        else:
            trig_fo[:] = False
            self.deep_flagged = np.zeros(n, dtype=bool)

        flagged = trig_np | trig_st | trig_fo
        self._tot += n
        self._flagged += int(flagged.sum())
        self._np += int(trig_np.sum())
        self._st += int(trig_st.sum())
        self._fo += int(trig_fo.sum())
        return flagged

    def finalize(self):
        d = max(1, self._tot)
        return {
            'detector_flag_rate': self._flagged / d,
            'trigger_nonprogress_rate': self._np / d,
            'trigger_stall_rate': self._st / d,
            'trigger_faceoff_rate': self._fo / d,
        }


class FollowerWrapperWithDetector(FollowerWrapper):
    """FollowerWrapper + online DeadlockDetector run right after the planner."""

    def __init__(self, env, config, detector=None, resolve=False, resolve_mode='waypoint',
                 resolve_k=6):
        super().__init__(env, config)
        self.detector = detector or DeadlockDetector()
        self.jam_stats = JamStats()
        self.resolve = resolve
        self.resolve_mode = resolve_mode   # 'waypoint' (multi-step plan) or 'override'
        self.resolve_k = resolve_k         # multi-step PIBT horizon
        self.last_flagged = None
        self.last_positions = None
        self.last_desired = None
        self._obst_arr = None
        self._resolve_calls = 0      # jams resolved over the episode
        self._resolve_overrides = 0  # agent-steps whose action we overrode
        self._steps = 0

    def reset_state(self):
        super().reset_state()
        self.detector.reset(len(self.get_global_agents_xy()))
        self.jam_stats.reset(self.grid.get_obstacles().shape)
        self._obst_arr = self.grid.get_obstacles()
        self.last_flagged = None
        self.last_positions = None
        self.last_desired = None
        self._resolve_calls = 0
        self._resolve_overrides = 0
        self._steps = 0

    def _run_detector(self, observations, paths):
        """Update the detector + jam stats from current observations/paths. Stores
        last_flagged / last_positions / last_desired. Returns the deep-stuck mask."""
        positions, goals, next_wp, dists, on_goal = [], [], [], [], []
        for k, o in enumerate(observations):
            p = tuple(o['xy']); g = tuple(o['target_xy'])
            path = paths[k]
            positions.append(p); goals.append(g)
            on_goal.append(p == g)
            if path and len(path) >= 2:
                next_wp.append(tuple(path[1])); dists.append(len(path) - 1)
            else:
                next_wp.append(p); dists.append(None if not path else 0)
        self.last_flagged = self.detector.step(positions, goals, next_wp, dists, on_goal)
        # grid-frame positions/desired (same frame as obstacles) for clustering/PIBT.
        grid_positions = [tuple(p) for p in self.grid.get_agents_xy()]
        desired = []
        for k, gp in enumerate(grid_positions):
            path = paths[k]
            dxy = (0, 0)
            if path and len(path) >= 2:
                dxy = (path[1][0] - path[0][0], path[1][1] - path[0][1])
                if abs(dxy[0]) + abs(dxy[1]) != 1:
                    dxy = (0, 0)
            cell = (gp[0] + dxy[0], gp[1] + dxy[1])
            h, w = self._obst_arr.shape
            if not (0 <= cell[0] < h and 0 <= cell[1] < w) or self._obst_arr[cell]:
                cell = gp
            desired.append(cell)
        self.last_positions = grid_positions
        self.last_desired = desired
        self.jam_stats.update(self.last_flagged, grid_positions)
        return self.detector.deep_flagged

    def _bake_paths(self, observations, paths):
        """Bake (possibly redirected) per-agent paths into obs['obstacles'] exactly
        like FollowerWrapper.observation (inference: intrinsic reward not needed)."""
        new_goals = []
        for k, path in enumerate(paths):
            obs = observations[k]
            if not path:
                new_goals.append(obs['target_xy']); path = []
            else:
                new_goals.append(path[1] if len(path) >= 2 else obs['target_xy'])
            obs['obstacles'][obs['obstacles'] > 0] *= -1
            r = obs['obstacles'].shape[0] // 2
            for (gx, gy) in path:
                x, y = self.get_relative_xy(*obs['xy'], gx, gy, r)
                if x is not None and y is not None:
                    obs['obstacles'][x, y] = 1.0
                else:
                    break
        self.prev_goals = new_goals
        self.intrinsic_reward = [0.0] * len(observations)

    def observation(self, observations):
        if self.resolve and self.resolve_mode == 'waypoint':
            # D1/B: replace deeply-stuck agents' PATH with a multi-step PIBT escape
            # path, then let the policy execute it (collision-aware) — no raw override.
            self.re_plan.update(observations)
            paths = list(self.re_plan.get_path())
            deep = self._run_detector(observations, paths)
            if deep is not None and deep.any():
                targets = [tuple(t) for t in self.grid.get_targets_xy()]
                esc, (n_jams, n_over) = resolve_jams_paths(
                    self.last_positions, targets, deep, self._obst_arr,
                    K=self.resolve_k, min_jam=1)
                for a, gpath in esc.items():
                    obs_xy = tuple(observations[a]['xy'])
                    g0 = gpath[0]
                    # grid path -> obs/planner frame via deltas (frame-independent)
                    paths[a] = [(obs_xy[0] + (c[0] - g0[0]), obs_xy[1] + (c[1] - g0[1]))
                                for c in gpath]
                self._resolve_calls += n_jams
                self._resolve_overrides += n_over
            self._bake_paths(observations, paths)
            return observations
        # detector-only, or raw-override mode (override applied in step())
        observations = super().observation(observations)
        paths = self.re_plan.get_path()
        self._run_detector(observations, paths)
        return observations

    def step(self, action):
        deep = self.detector.deep_flagged
        if self.resolve and self.resolve_mode == 'override' and deep is not None and deep.any():
            overrides, (n_jams, n_over) = resolve_jams(
                self.last_positions, self.last_desired, deep, self._obst_arr, min_jam=1)
            action = list(action)
            for a, act in overrides.items():
                action[a] = act
            self._resolve_calls += n_jams
            self._resolve_overrides += n_over
        self._steps += 1
        obs, rew, done, tr, info = super().step(action)
        if all(done) or all(tr):
            m = info[0].setdefault('metrics', {})
            m.update(self.detector.finalize())
            m.update(self.jam_stats.finalize())
            if self.resolve:
                m['resolver_jams_per_step'] = self._resolve_calls / max(1, self._steps)
                m['resolver_override_rate'] = self._resolve_overrides / max(1, self._steps * len(action))
        return obs, rew, done, tr, info


def make_follower_preprocessor_with_detector(resolve=False, resolve_t=30,
                                             resolve_mode='waypoint', resolve_k=6):
    def _preproc(env, algo_config):
        cfg = algo_config.training_config.preprocessing
        det = DeadlockDetector(resolve_t=resolve_t)
        env = FollowerWrapperWithDetector(env=env, config=cfg, detector=det,
                                          resolve=resolve, resolve_mode=resolve_mode,
                                          resolve_k=resolve_k)
        env = CutObservationWrapper(env, target_observation_radius=cfg.network_input_radius)
        env = ConcatPositionalFeatures(env)
        return env
    return _preproc


# detector-only (no action override) and detector+resolver variants
follower_preprocessor_with_detector = make_follower_preprocessor_with_detector(resolve=False)
follower_preprocessor_with_resolver = make_follower_preprocessor_with_detector(resolve=True)
