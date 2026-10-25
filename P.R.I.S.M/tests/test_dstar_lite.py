"""D* Lite must agree with a from-scratch Dijkstra before and after every change."""
import heapq
import math

import numpy as np
import pytest

from prism.nav.dstar_lite import DStarLite


def dijkstra(ds: DStarLite, goal):
    d = np.full(ds.cost.shape, np.inf)
    d[goal] = 0.0
    pq = [(0.0, goal)]
    while pq:
        dd, u = heapq.heappop(pq)
        if dd > d[u]:
            continue
        for v, step in ds._neighbours(u):
            c = ds.edge_cost(u, v, step)
            if dd + c < d[v]:
                d[v] = dd + c
                heapq.heappush(pq, (d[v], v))
    return d


def test_matches_dijkstra_through_changes():
    rng = np.random.default_rng(1)
    cost = rng.uniform(1.0, 4.0, (40, 40))
    cost[rng.random((40, 40)) < 0.15] = np.inf
    start, goal = (2, 2), (37, 37)
    cost[start] = cost[goal] = 1.0
    ds = DStarLite(cost, start, goal, res_m=0.1)
    ds.compute()
    assert math.isclose(ds.path_cost(), dijkstra(ds, goal)[start], rel_tol=1e-9)
    for k in range(8):
        cells = [tuple(rng.integers(0, 40, 2)) for _ in range(30)]
        cells = [c for c in cells if c not in (goal,)]
        new = [np.inf if rng.random() < 0.5 else float(rng.uniform(1, 4)) for _ in cells]
        ds.update_costs(cells, new)
        path = ds.path()
        if len(path) > 3:
            ds.move_start(path[2])
        ds.compute()
        truth = dijkstra(ds, goal)[ds.start]
        got = ds.path_cost()
        if math.isfinite(truth):
            assert math.isclose(got, truth, rel_tol=1e-9), (k, got, truth)
        else:
            assert not math.isfinite(got)


def test_path_is_connected_and_reaches_goal():
    cost = np.ones((30, 30))
    cost[5:25, 15] = np.inf
    ds = DStarLite(cost, (15, 2), (15, 28))
    assert ds.compute()
    p = ds.path()
    assert p[0] == (15, 2) and p[-1] == (15, 28)
    for a, b in zip(p, p[1:]):
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1
        assert math.isfinite(cost[b])


def test_sealed_goal_is_unreachable():
    cost = np.ones((20, 20))
    cost[:, 10] = np.inf
    ds = DStarLite(cost, (10, 2), (10, 18))
    assert not ds.compute()
    assert ds.exhausted and ds.path() == []


def test_capped_search_resumes():
    cost = np.ones((80, 80))
    ds = DStarLite(cost, (0, 0), (79, 79))
    ok = ds.compute(max_expansions=10)
    assert not ds.converged
    while not ds.converged:
        ok = ds.compute(max_expansions=50)
    assert ok and len(ds.path()) == 80


def test_no_corner_cutting():
    cost = np.ones((3, 3))
    cost[0, 1] = cost[1, 0] = np.inf
    ds = DStarLite(cost, (0, 0), (1, 1))
    assert not ds.compute()


def test_rejects_out_of_grid():
    with pytest.raises(ValueError):
        DStarLite(np.ones((5, 5)), (0, 0), (9, 9))
