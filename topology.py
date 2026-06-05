"""Static map topology for deadlock prediction.

The project's thesis hook is that deadlocks form at *non-biconnected* sites —
dead-ends and narrow corridors. Here we compute, from the static obstacle grid
alone, the topological features that name those sites:

  articulation_points : cells whose removal disconnects the free-space graph
                        (the formal cut vertices = exactly the non-biconnected
                        sites where PIBT's completeness guarantee fails).
  corridor_cells      : free cells with <= 2 free orthogonal neighbours
                        (dead-ends deg 1, corridors deg 2) — a cheap "narrowness".

Used to test whether measured deadlock concentrates at these cells (topology
predicts deadlock), and later to place proactive guidance there.

Coordinates are grid-frame (x, y) matching grid.get_obstacles()/get_agents_xy().
OBSTACLE=1, FREE=0.
"""
import numpy as np

NEI = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def free_neighbors(obst, c):
    h, w = obst.shape
    x, y = c
    out = []
    for dx, dy in NEI:
        nx, ny = x + dx, y + dy
        if 0 <= nx < h and 0 <= ny < w and not obst[nx, ny]:
            out.append((nx, ny))
    return out


def corridor_cells(obst):
    """Free cells with <= 2 free orthogonal neighbours (dead-ends + corridors)."""
    h, w = obst.shape
    out = set()
    for x in range(h):
        for y in range(w):
            if not obst[x, y] and len(free_neighbors(obst, (x, y))) <= 2:
                out.add((x, y))
    return out


def articulation_points(obst):
    """Cut vertices of the free-space graph (4-connectivity), iterative DFS so big
    maps don't blow the recursion stack. Returns a set of (x, y) cells."""
    h, w = obst.shape
    free = [(x, y) for x in range(h) for y in range(w) if not obst[x, y]]
    idx = {c: i for i, c in enumerate(free)}
    n = len(free)
    disc = [-1] * n
    low = [0] * n
    parent = [-1] * n
    is_ap = [False] * n
    timer = [0]

    for s in range(n):
        if disc[s] != -1:
            continue
        # iterative DFS; stack holds (node, neighbor-iterator index, child-count)
        stack = [(s, 0, 0)]
        disc[s] = low[s] = timer[0]; timer[0] += 1
        children_root = 0
        nbrs = {s: free_neighbors(obst, free[s])}
        while stack:
            u, ni, _ = stack[-1]
            us = free[u]
            neigh = nbrs.setdefault(u, free_neighbors(obst, us))
            if ni < len(neigh):
                stack[-1] = (u, ni + 1, stack[-1][2])
                v = idx[neigh[ni]]
                if disc[v] == -1:
                    parent[v] = u
                    disc[v] = low[v] = timer[0]; timer[0] += 1
                    if u == s:
                        children_root += 1
                    stack.append((v, 0, 0))
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])
            else:
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    low[p] = min(low[p], low[u])
                    if parent[u] != -1 and low[u] >= disc[p] and p != s:
                        is_ap[p] = True
        if children_root > 1:
            is_ap[s] = True

    return {free[i] for i in range(n) if is_ap[i]}


def summarize(obst):
    """Convenience: counts + the two cell sets."""
    free = int((obst == 0).sum())
    ap = articulation_points(obst)
    cor = corridor_cells(obst)
    return {
        'free_cells': free,
        'articulation': ap,
        'corridor': cor,
        'frac_articulation': len(ap) / max(1, free),
        'frac_corridor': len(cor) / max(1, free),
    }
