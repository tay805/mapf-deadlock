"""Make POGEMA's move-revert iterative AND cycle-safe (importing applies the patch).

POGEMA's soft-collision `move_agents` calls `_revert_action`, which is TAIL-recursive
along a chain of dependent moves. Two failure modes at high agent density:

  1. Long (but finite) chains overflow the C stack -> segfault (exit 139).
  2. The chain can form a CYCLE (agent A's revert leads back to A). The shipped
     recursive code then recurses forever -> ALSO a stack-overflow segfault, so the
     two modes look identical from outside. A naive iterative rewrite removes the
     stack limit and so converts mode (2) into an INFINITE LOOP that hangs at 100%
     CPU and never finishes the episode (observed: adaptive metering never saving).

Fix: iterative (no depth limit, kills mode 1) PLUS a `seen` set that breaks a cycle
the moment an agent is revisited (kills mode 2). On a cycle we stop reverting that
chain — equivalent to leaving the rotation in place, which is collision-free anyway.
Set POGEMA_REVERT_DEBUG=1 to log cycle hits and max chain length per process.
"""
import os

from pogema.envs import Pogema

_DEBUG = os.environ.get('POGEMA_REVERT_DEBUG') == '1'
_stats = {'cycles': 0, 'max_chain': 0}


def _revert_action_iterative(self, agent_idx, used_cells, cell, actions):
    seen = set()
    chain = 0
    while True:
        if agent_idx in seen:           # cycle: recursion would loop forever here
            _stats['cycles'] += 1
            if _DEBUG and _stats['cycles'] % 1000 == 1:
                print(f"[revert] cycle #{_stats['cycles']} (chain={chain})", flush=True)
            used_cells.setdefault(cell, [])
            return actions, used_cells
        seen.add(agent_idx)
        chain += 1
        if chain > _stats['max_chain']:
            _stats['max_chain'] = chain
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
