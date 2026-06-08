"""Make POGEMA's move-revert iterative (importing this applies the patch).

POGEMA's soft-collision `move_agents` calls `_revert_action`, which is TAIL-recursive
along a chain of dependent moves. At high agent density these chains get very deep and
overflow the C stack -> segfault. We replace it with an exactly-equivalent iterative
version (tail recursion -> while loop), which removes the depth limit. This unblocks
high-density and adaptive-metering runs (reviewer S2).
"""
from pogema.envs import Pogema


def _revert_action_iterative(self, agent_idx, used_cells, cell, actions):
    while True:
        actions[agent_idx] = 0
        used_cells[cell].remove(agent_idx)
        new_cell = self.grid.positions_xy[agent_idx]
        if new_cell in used_cells and len(used_cells[new_cell]) > 0:
            used_cells[new_cell].append(agent_idx)
            agent_idx = used_cells[new_cell][0]   # follow the chain (was recursion)
            cell = new_cell
        else:
            used_cells.setdefault(new_cell, []).append(agent_idx)
            return actions, used_cells


Pogema._revert_action = _revert_action_iterative
