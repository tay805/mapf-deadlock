"""Minimal one-timestep PIBT (Priority Inheritance with Backtracking).

Okumura et al. 2019 (arXiv:1901.11282), pared down to a single planning step on a
small cropped subgrid. Used by the P4 resolver: given the agents inside a jam crop
and where each wants to head, return ONE collision-free move per agent (no vertex
and no edge/swap conflicts).

API:
    pibt_solve(pos, goals, obstacles, bounds, priority=None) -> {agent: next_cell}
  pos, goals : dict agent_id -> (x, y). goals = the cell each agent heads toward
               (e.g. its Follower waypoint), used only to rank candidate moves.
  obstacles  : set of blocked (x, y) cells (static obstacles in the crop).
  bounds     : (x0, y0, x1, y1) inclusive crop box.
  priority   : optional dict agent_id -> number (higher plans first). Default =
               distance-to-goal (farther first), which is the usual PIBT choice.
Cells stay inside `bounds`; waiting (no move) is always a candidate.
"""
from collections import deque

MOVES = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]   # wait + 4-neighborhood


def _bfs_field(goal, passable, bounds):
    """Shortest-path distance from `goal` over passable cells within bounds."""
    x0, y0, x1, y1 = bounds
    dist = {}
    if not passable(goal):
        return dist
    dist[goal] = 0
    dq = deque([goal])
    while dq:
        c = dq.popleft()
        nd = dist[c] + 1
        for dx, dy in MOVES[1:]:
            nc = (c[0] + dx, c[1] + dy)
            if x0 <= nc[0] <= x1 and y0 <= nc[1] <= y1 and passable(nc) and nc not in dist:
                dist[nc] = nd
                dq.append(nc)
    return dist


def pibt_solve(pos, goals, obstacles, bounds, priority=None):
    x0, y0, x1, y1 = bounds
    agents = list(pos)
    occ = set(obstacles)

    def in_b(c):
        return x0 <= c[0] <= x1 and y0 <= c[1] <= y1

    def passable(c):
        return c not in occ

    # Obstacle-aware distance fields within the crop (one BFS per distinct goal).
    fields = {}
    for a in agents:
        g = goals[a]
        if g not in fields:
            fields[g] = _bfs_field(g, lambda c: in_b(c) and passable(c), bounds)

    def dist(c, g):
        # Reachable -> true crop distance; else Manhattan + big offset (still a gradient).
        return fields[g].get(c, abs(c[0] - g[0]) + abs(c[1] - g[1]) + 10 ** 6)

    agent_at = {pos[a]: a for a in agents}
    if priority is None:
        priority = {a: dist(pos[a], goals[a]) for a in agents}
    order = sorted(agents, key=lambda a: (-priority[a], a))

    nxt = {}          # agent -> chosen next cell (decided)
    taken = set()     # next cells already reserved

    def candidates(a):
        px, py = pos[a]
        cs = [(px + dx, py + dy) for dx, dy in MOVES]
        cs = [c for c in cs if in_b(c) and passable(c)]
        cs.sort(key=lambda c: dist(c, goals[a]))     # prefer getting closer to goal
        return cs

    def pibt(ai, aj):
        """Try to give ai a valid move. aj = higher-priority caller (for swap guard)."""
        for v in candidates(ai):
            if v in taken:                          # vertex conflict
                continue
            if aj is not None and v == pos[aj]:     # edge/swap conflict with caller
                continue
            nxt[ai] = v
            taken.add(v)
            ak = agent_at.get(v)                    # agent currently occupying v
            if ak is not None and ak != ai and ak not in nxt:
                if not pibt(ak, ai):                # ak must move out (priority inheritance)
                    del nxt[ai]
                    taken.discard(v)
                    continue                         # backtrack: try ai's next candidate
            return True
        # No move worked -> stay if possible.
        p = pos[ai]
        if p in taken:                              # someone took our cell: cannot stay
            return False
        nxt[ai] = p
        taken.add(p)
        return False

    for a in order:
        if a not in nxt:
            pibt(a, None)
    return nxt
