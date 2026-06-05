"""P4 resolver — stage 1: cluster detector-flagged agents into jams + crop region.

A single flagged agent isn't actionable; a *jam* is a connected group of flagged
agents close together. We:
  1. cluster flagged agents into jams (connected components by proximity),
  2. crop a padded subgrid (bounding box) around each jam,
  3. collect ALL agents inside that crop (flagged or not) — they are the local
     MAPF problem the resolver (PIBT, later stage) must solve together.

Pure functions here; the env wrapper calls them. PIBT + hand-back come next.
"""
from pibt import pibt_solve

# POGEMA action index -> (dx, dy)  (grid_config.MOVES) and its inverse.
MOVES = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_OF = {m: i for i, m in enumerate(MOVES)}


def cluster_jams(flagged_idx, positions, link_radius=2):
    """Connected components of flagged agents. Two flagged agents are linked if
    their Chebyshev distance <= link_radius. Returns list of sets of agent ids."""
    flagged = list(flagged_idx)
    parent = {i: i for i in flagged}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a_i in range(len(flagged)):
        ax, ay = positions[flagged[a_i]]
        for b_i in range(a_i + 1, len(flagged)):
            b = flagged[b_i]
            bx, by = positions[b]
            if max(abs(ax - bx), abs(ay - by)) <= link_radius:
                parent[find(flagged[a_i])] = find(b)

    comps = {}
    for i in flagged:
        comps.setdefault(find(i), set()).add(i)
    return list(comps.values())


def crop_box(cluster, positions, margin, grid_shape):
    """Padded, clipped bounding box (x0, y0, x1, y1 inclusive) around a jam."""
    xs = [positions[i][0] for i in cluster]
    ys = [positions[i][1] for i in cluster]
    h, w = grid_shape
    return (max(0, min(xs) - margin), max(0, min(ys) - margin),
            min(h - 1, max(xs) + margin), min(w - 1, max(ys) + margin))


def agents_in_box(positions, box):
    """All agent ids whose cell lies inside the crop box (the local problem)."""
    x0, y0, x1, y1 = box
    return [i for i, (x, y) in enumerate(positions) if x0 <= x <= x1 and y0 <= y <= y1]


def resolve_jams(grid_pos, desired, flagged_bool, obst_arr,
                 margin=3, link_radius=2, min_jam=2):
    """Targeted resolver: for each *multi-agent* jam (>= min_jam flagged agents),
    run PIBT over ONLY the flagged agents, treating bystanders in the crop as
    static obstacles (PIBT routes around them; we don't override healthy agents'
    policy actions). Returns (overrides {agent: action_index}, (num_jams, num_overrides)).

    grid_pos : list agent->(x,y) (grid frame, same frame as obst_arr).
    desired  : list agent->(x,y) preferred next cell (planner waypoint, grid frame).
    flagged_bool : per-agent bool from the detector.
    obst_arr : 2D obstacle grid (OBSTACLE=1/FREE=0), same frame as grid_pos.
    """
    flagged_idx = [i for i, f in enumerate(flagged_bool) if f]
    if not flagged_idx:
        return {}, (0, 0)
    grid_shape = obst_arr.shape
    jams = [j for j in cluster_jams(flagged_idx, grid_pos, link_radius) if len(j) >= min_jam]
    overrides = {}
    flagged_set = set(flagged_idx)
    for jam in jams:
        box = crop_box(jam, grid_pos, margin, grid_shape)
        x0, y0, x1, y1 = box
        in_crop = agents_in_box(grid_pos, box)
        jam_agents = [a for a in in_crop if a in jam]            # only flagged -> PIBT-controlled
        bystanders = [grid_pos[a] for a in in_crop if a not in flagged_set]
        pos = {a: grid_pos[a] for a in jam_agents}
        goals = {a: desired[a] for a in jam_agents}
        obst = {(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)
                if obst_arr[x, y]}
        obst.update(bystanders)                                 # healthy agents = static blockers
        nxt = pibt_solve(pos, goals, obst, box)
        for a in jam_agents:
            dx = nxt[a][0] - grid_pos[a][0]
            dy = nxt[a][1] - grid_pos[a][1]
            act = ACTION_OF.get((dx, dy))
            if act is not None:
                overrides[a] = act
    return overrides, (len(jams), len(overrides))


class JamStats:
    """Accumulates per-step jam statistics over an episode (validation of stage 1):
    how many jams form, how big, how large the crops, how many agents involved."""

    def __init__(self, margin=3, link_radius=2):
        self.margin = margin
        self.link_radius = link_radius
        self.grid_shape = None
        self._steps = 0
        self._steps_with_jam = 0
        self._n_jams = 0
        self._jam_sizes = []          # agents flagged per jam
        self._crop_agents = []        # all agents inside each jam's crop
        self._crop_cells = []         # crop area in cells

    def reset(self, grid_shape):
        self.grid_shape = grid_shape
        self._steps = 0
        self._steps_with_jam = 0
        self._n_jams = 0
        self._jam_sizes = []
        self._crop_agents = []
        self._crop_cells = []

    def update(self, flagged_bool, positions):
        self._steps += 1
        flagged_idx = [i for i, f in enumerate(flagged_bool) if f]
        if not flagged_idx:
            return
        jams = cluster_jams(flagged_idx, positions, self.link_radius)
        self._steps_with_jam += 1
        self._n_jams += len(jams)
        for jam in jams:
            box = crop_box(jam, positions, self.margin, self.grid_shape)
            self._jam_sizes.append(len(jam))
            self._crop_agents.append(len(agents_in_box(positions, box)))
            self._crop_cells.append((box[2] - box[0] + 1) * (box[3] - box[1] + 1))

    def finalize(self):
        import statistics as st
        mean = lambda xs: float(st.mean(xs)) if xs else 0.0
        return {
            'jam_steps_frac': self._steps_with_jam / max(1, self._steps),
            'jams_per_step': self._n_jams / max(1, self._steps),
            'mean_jam_size': mean(self._jam_sizes),
            'max_jam_size': max(self._jam_sizes) if self._jam_sizes else 0,
            'mean_crop_agents': mean(self._crop_agents),
            'mean_crop_cells': mean(self._crop_cells),
        }
