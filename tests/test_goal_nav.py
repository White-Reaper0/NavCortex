"""Point A -> Point B: closed loop through the real runtime in the 2-D kinematic sim."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from prism.nav import bev_utils as bu
from prism.nav.goal_planner import GoalPlanner
from prism.nav.planner import Planner

ROOT = Path(__file__).resolve().parent.parent


def _sim():
    spec = importlib.util.spec_from_file_location("sim_goal_nav", ROOT / "tools" / "sim_goal_nav.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sim_goal_nav"] = mod          # dataclasses look the module up by name
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("scenario", ["open", "dead_end", "pedestrian"])
def test_reaches_point_b_without_collision(scenario):
    r = _sim().run_episode(scenario, seed=0)
    assert r["success"], r
    assert r["collisions"] == 0 and r["interventions"] == 0


def test_dead_end_uses_dstar():
    r = _sim().run_episode("dead_end", seed=1)
    assert r["dstar_cycles"] > 0 and r["success"]


def test_open_ground_heads_straight_for_b():
    gp = GoalPlanner()
    gp.set_goal(0.0, 5.0)
    st = bu.synthetic_state("clear")
    s = gp.step(st, (0.0, 0.0, 0.0), 0.0)
    assert s.mode == "open"
    assert abs(s.steer_yaw) < 1e-6


def test_wall_ahead_switches_to_dstar_and_steers_off_axis():
    gp = GoalPlanner()
    gp.set_goal(0.0, 5.0)
    st = bu.synthetic_state("wall", wall_dist_m=1.2, wall_height_m=0.15)
    s = gp.step(st, (0.0, 0.0, 0.0), 0.0)
    assert s.mode == "dstar"
    assert s.path_world is not None and len(s.path_world) > 2
    assert abs(s.steer_yaw) > np.deg2rad(5)


def test_arrival():
    gp = GoalPlanner()
    gp.set_goal(0.0, 0.1)
    s = gp.step(bu.synthetic_state("clear"), (0.0, 0.0, 0.0), 0.0)
    assert s.mode == "arrived" and s.reason.startswith("G0")


def test_unseen_ground_costs_but_is_not_blocked():
    gp = GoalPlanner()
    cost = gp.grid.cost_grid(0.0)
    assert np.all(np.isfinite(cost))
    assert np.all(cost == gp.cfg.grid.cost_unknown)
    assert gp.cfg.grid.cost_unknown > gp.cfg.grid.cost_safe


def test_planner_has_17_arcs_and_goal_progress():
    pl = Planner()
    assert len(pl.candidates) == 17
    st = bu.synthetic_state("clear")
    # a goal to the left: the best arc should turn left (positive yaw)
    trajs = pl.plan(st, None, goal_yaw=np.deg2rad(60), goal_xy=(-1.5, 1.0))
    i, best = pl.best(trajs)
    assert best is not None and best.yaw[-1] > 0.2
