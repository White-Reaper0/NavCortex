"""D* Lite incremental replanning on a 2-D cost grid.

Koenig, S. & Likhachev, M. (2002), "D* Lite", AAAI 2002.

Why D* Lite and not A* from scratch every cycle
-----------------------------------------------
The vehicle discovers the world as it drives: a dead end is only seen once the
camera looks into it.  D* Lite searches *backwards* from Point B, so when the
vehicle moves or a handful of cells change cost (a wall appears, a person steps
in, a dynamic cell expires), only the part of the search tree those changes touch
is repaired.  The first search costs about as much as A*; every replan after that
is usually a small fraction of it.

Grid model
----------
* ``cost[r, c]`` is the traversal cost *per metre* of entering/leaving that cell.
  ``np.inf`` means blocked.  Unseen ground is given a finite cost by the caller
  (see `goal_map.GlobalCostGrid`): unseen ground costs, it is never free.
* 8-connected.  Edge cost = step length (1 or sqrt 2 cells, in metres) x the mean
  of the two cells' costs.  A diagonal step is blocked if either cell it cuts the
  corner of is blocked, so the path never squeezes between two diagonal obstacles.
* The heuristic is the octile distance x the cheapest cost any cell can have,
  which keeps it admissible and consistent.

This is the "basic" D* Lite of Fig. 3 in the paper with the key modifier ``k_m``
from the optimised version, and a lazy-deletion binary heap.
"""
from __future__ import annotations

import heapq
import math
from typing import Iterable, Optional

import numpy as np

_SQRT2 = math.sqrt(2.0)
_NEIGH = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2))
INF = math.inf
#: keys are sums of float path costs; two keys that are equal in exact arithmetic
#: can differ by an ulp, and a strict ``<`` would then end the repair one vertex
#: early.  Keys are compared with this tolerance instead.
_EPS = 1e-9


def _key_less(a: tuple[float, float], b: tuple[float, float]) -> bool:
    if a[0] < b[0] - _EPS:
        return True
    if a[0] > b[0] + _EPS:
        return False
    return a[1] < b[1] - _EPS


class DStarLite:
    """Incremental shortest path from a moving start to a fixed goal."""

    def __init__(self, cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
                 res_m: float = 1.0, min_cost: Optional[float] = None):
        self.cost = np.asarray(cost, np.float64).copy()
        if self.cost.ndim != 2:
            raise ValueError("cost must be a 2-D grid")
        self.H, self.W = self.cost.shape
        self.res = float(res_m)
        finite = self.cost[np.isfinite(self.cost)]
        mc = float(finite.min()) if finite.size else 1.0
        self.h_scale = float(min_cost if min_cost is not None else mc) * self.res
        if self.h_scale <= 0:
            raise ValueError("cell costs must be positive")
        self.goal = self._check(goal)
        self.start = self._check(start)
        self._last = self.start
        self.km = 0.0
        self.g = np.full((self.H, self.W), INF)
        self.rhs = np.full((self.H, self.W), INF)
        self.rhs[self.goal] = 0.0
        self._open: dict[tuple[int, int], tuple[float, float]] = {}
        self._heap: list = []
        self._push(self.goal, self._key(self.goal))
        self.expansions = 0           # vertices expanded by the last compute()
        self.total_expansions = 0
        self.converged = False
        self.exhausted = False

    # ------------------------------------------------------------------ helpers
    def _check(self, rc) -> tuple[int, int]:
        r, c = int(rc[0]), int(rc[1])
        if not (0 <= r < self.H and 0 <= c < self.W):
            raise ValueError(f"cell {rc} is outside the {self.H}x{self.W} grid")
        return r, c

    def _h(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
        lo, hi = (dr, dc) if dr < dc else (dc, dr)
        return (hi + (_SQRT2 - 1.0) * lo) * self.h_scale

    def _key(self, s: tuple[int, int]) -> tuple[float, float]:
        m = min(self.g[s], self.rhs[s])
        return (m + self._h(self.start, s) + self.km, m)

    def _push(self, s, k) -> None:
        self._open[s] = k
        heapq.heappush(self._heap, (k[0], k[1], s))

    def _top(self):
        """Peek the smallest live entry (lazy deletion of stale heap items)."""
        h = self._heap
        while h:
            k0, k1, s = h[0]
            cur = self._open.get(s)
            if cur is not None and cur[0] == k0 and cur[1] == k1:
                return (k0, k1), s
            heapq.heappop(h)
        return (INF, INF), None

    def edge_cost(self, a: tuple[int, int], b: tuple[int, int], step: float) -> float:
        ca, cb = self.cost[a], self.cost[b]
        if not (math.isfinite(ca) and math.isfinite(cb)):
            return INF
        if step > 1.0:      # diagonal: refuse to cut a blocked corner
            if not (math.isfinite(self.cost[a[0], b[1]]) and math.isfinite(self.cost[b[0], a[1]])):
                return INF
        return step * self.res * 0.5 * (ca + cb)

    def _neighbours(self, s):
        r, c = s
        for dr, dc, step in _NEIGH:
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.H and 0 <= cc < self.W:
                yield (rr, cc), step

    def _best_rhs(self, u) -> float:
        best = INF
        for v, step in self._neighbours(u):
            gv = self.g[v]
            if gv == INF:
                continue
            c = self.edge_cost(u, v, step)
            if c + gv < best:
                best = c + gv
        return best

    def _update_vertex(self, u) -> None:
        if u != self.goal:
            self.rhs[u] = self._best_rhs(u)
        if self.g[u] != self.rhs[u]:
            self._push(u, self._key(u))
        else:
            self._open.pop(u, None)

    # ------------------------------------------------------------------ public API
    def compute(self, max_expansions: int = 2_000_000) -> bool:
        """Repair the search tree.  Returns True if the start can reach the goal.

        With a finite `max_expansions` the repair may stop early; `self.converged`
        then reads False and the next call simply carries on (all search state
        persists), so a large search can be spread over several control cycles.
        `self.exhausted` is True only when the open list ran dry: the start is
        then provably unreachable.
        """
        n = 0
        start = self.start
        self.converged = self.exhausted = False
        while True:
            k_old, u = self._top()
            k_start = self._key(start)
            if u is None:
                self.converged = True
                self.exhausted = not (math.isfinite(self.g[start]) or math.isfinite(self.rhs[start]))
                break
            if not _key_less(k_old, k_start) and self.rhs[start] == self.g[start]:
                self.converged = True
                break
            if n >= max_expansions:
                break
            n += 1
            k_new = self._key(u)
            if _key_less(k_old, k_new):
                self._push(u, k_new)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                self._open.pop(u, None)
                for v, _ in self._neighbours(u):
                    self._update_vertex(v)
            else:
                self.g[u] = INF
                self._update_vertex(u)
                for v, _ in self._neighbours(u):
                    self._update_vertex(v)
        self.expansions = n
        self.total_expansions += n
        return math.isfinite(self.g[start]) or math.isfinite(self.rhs[start])

    def move_start(self, start: tuple[int, int]) -> None:
        """The vehicle moved.  Call before `compute()`."""
        s = self._check(start)
        if s == self.start:
            return
        self.km += self._h(self._last, s)
        self._last = s
        self.start = s

    def update_costs(self, cells: Iterable[tuple[int, int]], new_costs: Iterable[float]) -> int:
        """Change the cost of some cells.  Returns how many actually changed."""
        touched: set[tuple[int, int]] = set()
        n = 0
        for rc, nc in zip(cells, new_costs):
            s = (int(rc[0]), int(rc[1]))
            nc = float(nc)
            old = self.cost[s]
            if old == nc or (not math.isfinite(old) and not math.isfinite(nc)):
                continue
            self.cost[s] = nc
            n += 1
            r, c = s
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.H and 0 <= cc < self.W:
                        touched.add((rr, cc))
        for u in touched:
            self._update_vertex(u)
        return n

    def set_cost_grid(self, cost: np.ndarray) -> int:
        """Diff a whole new cost grid against the current one and apply the changes."""
        cost = np.asarray(cost, np.float64)
        same = (cost == self.cost) | (~np.isfinite(cost) & ~np.isfinite(self.cost))
        rr, cc = np.nonzero(~same)
        return self.update_costs(zip(rr.tolist(), cc.tolist()), cost[rr, cc].tolist())

    def path_cost(self) -> float:
        return float(self.g[self.start]) if math.isfinite(self.g[self.start]) \
            else float(self.rhs[self.start])

    def path(self, max_len: Optional[int] = None) -> list[tuple[int, int]]:
        """Greedy descent of g + c from the start to the goal.  [] if unreachable."""
        if not math.isfinite(self.path_cost()):
            return []
        out = [self.start]
        s = self.start
        seen = {s}
        limit = max_len if max_len is not None else self.H * self.W
        while s != self.goal and len(out) < limit:
            best, nxt = INF, None
            for v, step in self._neighbours(s):
                c = self.edge_cost(s, v, step)
                if c == INF:
                    continue
                val = c + self.g[v]
                if val < best:
                    best, nxt = val, v
            if nxt is None or nxt in seen or best == INF:
                break
            out.append(nxt)
            seen.add(nxt)
            s = nxt
        return out


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    H = W = 120
    cost = np.ones((H, W))
    cost[20:100, 60] = np.inf                   # a long wall with a gap at the bottom
    t0 = time.perf_counter()
    ds = DStarLite(cost, (60, 10), (60, 110), res_m=0.12)
    ok = ds.compute()
    t1 = time.perf_counter()
    p = ds.path()
    print(f"initial: reachable={ok} path={len(p)} cells cost={ds.path_cost():.2f} "
          f"expansions={ds.expansions} {1e3*(t1-t0):.1f} ms")
    # close the gap below -> path must go round the top
    ds.update_costs([(r, 60) for r in range(100, H)], [np.inf] * (H - 100))
    ds.move_start(p[5])
    t0 = time.perf_counter()
    ok = ds.compute()
    t1 = time.perf_counter()
    p2 = ds.path()
    print(f"replan:  reachable={ok} path={len(p2)} cells cost={ds.path_cost():.2f} "
          f"expansions={ds.expansions} {1e3*(t1-t0):.1f} ms")
    assert all(r < 20 for r, c in p2 if c == 60), "path must pass above the wall"
    ds.update_costs([(r, 60) for r in range(0, 20)], [np.inf] * 20)
    ok = ds.compute()
    print(f"sealed:  reachable={ok} path={len(ds.path())}")
    assert not ok
    print("self-test passed")
