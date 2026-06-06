"""Congestion-aware ROUTING baseline (competitor proxy).

Represents the routing family — Traffic-Flow-Opt (2308.11234), Guidance-Graph
(2402.01446), Highways (2304.04217) — whose shared idea is: bias routing AWAY
from high-traffic bottlenecks. We inject a betweenness-centrality penalty map into
Follower's A* planner via its set_penalties() interface (the planner's static cost
already reads `penalties[cell]`), so agents route around high-betweenness cells.

This is built on the SAME Follower base + harness, so a head-to-head vs our
density controller isolates the INTERVENTION (routing vs metering) as the only
variable. Strength is tunable (--route-strength).
"""
from follower.preprocessing import (
    FollowerWrapper, CutObservationWrapper, ConcatPositionalFeatures,
)
import topology


class RoutingFollowerWrapper(FollowerWrapper):
    def __init__(self, env, config, strength=4.0):
        super().__init__(env, config)
        self.strength = strength
        self._penalty = None
        self._injected = False

    def reset_state(self):
        super().reset_state()
        obst = self.grid.get_obstacles()
        free, bc = topology.betweenness_scores(obst)
        bcmax = float(bc.max()) if len(bc) and bc.max() > 0 else 1.0
        h, w = obst.shape
        pen = [[1.0] * w for _ in range(h)]          # static-cost penalty map
        for (x, y), b in zip(free, bc):
            pen[x][y] = 1.0 + self.strength * (b / bcmax)   # avoid bottlenecks
        self._penalty = pen
        self._injected = False

    def observation(self, observations):
        observations = super().observation(observations)   # creates planners + paths
        if not self._injected:
            planners = getattr(self.re_plan._agent, 'planner', None)
            if planners:
                for p in planners:
                    p.set_penalties(self._penalty)          # guidance-graph proxy
                self._injected = True
        return observations


def make_routing_preprocessor(strength=4.0):
    def _preproc(env, algo_config):
        cfg = algo_config.training_config.preprocessing
        env = RoutingFollowerWrapper(env, cfg, strength=strength)
        env = CutObservationWrapper(env, target_observation_radius=cfg.network_input_radius)
        env = ConcatPositionalFeatures(env)
        return env
    return _preproc
