"""Tests for the minimal PIBT (pibt.py). Run: python test_pibt.py"""
import random

from pibt import pibt_solve


def check_valid(pos, nxt):
    cells = list(nxt.values())
    assert len(cells) == len(set(cells)), f"VERTEX conflict: {nxt}"           # no two -> same cell
    for a in pos:
        for b in pos:
            if a < b and nxt[a] == pos[b] and nxt[b] == pos[a]:
                raise AssertionError(f"SWAP conflict {a}<->{b}: {nxt}")         # no edge swap
    for a in pos:
        (px, py), (nx, ny) = pos[a], nxt[a]
        assert abs(px - nx) + abs(py - ny) <= 1, f"illegal move {a}: {pos[a]}->{nxt[a]}"


def test_corridor_pocket():
    bounds = (0, 0, 2, 5)
    obst = {(x, y) for x in range(3) for y in range(6)}
    for y in range(1, 5):
        obst.discard((1, y))      # 1-wide corridor
    obst.discard((0, 2))          # side pocket
    pos = {0: (1, 1), 1: (1, 4)}
    goals = {0: (1, 4), 1: (1, 1)}
    nxt = pibt_solve(pos, goals, obst, bounds)
    check_valid(pos, nxt)
    assert nxt[0] != pos[0] or nxt[1] != pos[1]      # at least one progresses


def test_swap_avoid():
    bounds = (0, 0, 2, 2)
    pos = {0: (1, 0), 1: (1, 2)}
    goals = {0: (1, 2), 1: (1, 0)}
    nxt = pibt_solve(pos, goals, set(), bounds)
    check_valid(pos, nxt)


def test_vertex_contention():
    bounds = (0, 0, 2, 2)
    pos = {0: (0, 1), 1: (2, 1)}
    goals = {0: (1, 1), 1: (1, 1)}
    nxt = pibt_solve(pos, goals, set(), bounds)
    check_valid(pos, nxt)
    assert sum(1 for a in pos if nxt[a] == (1, 1)) == 1   # exactly one wins the cell


def test_random_stress():
    random.seed(0)
    for _ in range(2000):
        bounds = (0, 0, 4, 4)
        cells = [(x, y) for x in range(5) for y in range(5)]
        obst = set(random.sample(cells, random.randint(0, 6)))
        free = [c for c in cells if c not in obst]
        k = min(len(free), random.randint(1, 8))
        starts = random.sample(free, k)
        pos = {i: starts[i] for i in range(k)}
        goals = {i: random.choice(free) for i in range(k)}
        nxt = pibt_solve(pos, goals, obst, bounds)
        check_valid(pos, nxt)
        for a in pos:
            assert nxt[a] not in obst                       # never step onto an obstacle


if __name__ == '__main__':
    test_corridor_pocket()
    test_swap_avoid()
    test_vertex_contention()
    test_random_stress()
    print("ALL PIBT TESTS PASSED")
