"""P4 resolver — stage 1: cluster detector-flagged agents into jams + crop region.

A single flagged agent isn't actionable; a *jam* is a connected group of flagged
agents close together. We:
  1. cluster flagged agents into jams (connected components by proximity),
  2. crop a padded subgrid (bounding box) around each jam,
  3. collect ALL agents inside that crop (flagged or not) — they are the local
     MAPF problem the resolver (PIBT, later stage) must solve together.

Pure functions here; the env wrapper calls them. PIBT + hand-back come next.
"""


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
