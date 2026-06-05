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


class DeadlockDetector:
    def __init__(self, t1=10, t2=8, warmup=10):
        self.t1 = t1            # non-progress budget
        self.t2 = t2            # stall budget
        self.warmup = warmup
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
        else:
            trig_fo[:] = False

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

    def __init__(self, env, config, detector=None):
        super().__init__(env, config)
        self.detector = detector or DeadlockDetector()
        self.last_flagged = None

    def reset_state(self):
        super().reset_state()
        self.detector.reset(len(self.get_global_agents_xy()))
        self.last_flagged = None

    def observation(self, observations):
        observations = super().observation(observations)   # runs planner + mutates obs
        paths = self.re_plan.get_path()
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
        return observations

    def step(self, action):
        obs, rew, done, tr, info = super().step(action)
        if all(done) or all(tr):
            info[0].setdefault('metrics', {}).update(self.detector.finalize())
        return obs, rew, done, tr, info


def follower_preprocessor_with_detector(env, algo_config):
    cfg = algo_config.training_config.preprocessing
    env = FollowerWrapperWithDetector(env=env, config=cfg)
    env = CutObservationWrapper(env, target_observation_radius=cfg.network_input_radius)
    env = ConcatPositionalFeatures(env)
    return env
