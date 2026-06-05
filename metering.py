"""Congestion metering: cap the number of concurrently-active agents.

Over the throughput peak, adding agents *reduces* throughput (deadlock delocalizes,
gridlock grows). Metering keeps only M agents active; the rest "park" (stay put),
reducing effective density to hold the system near its peak. Proof-of-concept:
static cap M (park agents with index >= M). The adaptive, metric-driven controller
that picks M online is the actual contribution.
"""
import gymnasium


class MeterWrapper(gymnasium.Wrapper):
    def __init__(self, env, m):
        super().__init__(env)
        self.m = int(m)

    def step(self, action):
        action = list(action)
        for i in range(self.m, len(action)):
            action[i] = 0   # park: action 0 = stay (MOVES[0] = (0,0))
        return self.env.step(action)
