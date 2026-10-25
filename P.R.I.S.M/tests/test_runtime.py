"""PrismNavigator end to end with injected perception (no network weights)."""
import numpy as np

from prism.config import PROC_H, PROC_W, DECISIONS, STOP
from prism.nav import bev_utils as bu
from prism.perception.dynamic_layer import DYNAMIC_CLASS
from prism.runtime import PrismNavigator
from prism.types import BEVMap, DepthResult, OdometryResult, SegResult


def _bev(kind="clear"):
    st = bu.synthetic_state(kind)
    obs = st[bu.CH_OBS] > 0.5
    return BEVMap(height=bu.state_height_m(st), trav_prob=st[bu.CH_SAFE:bu.CH_UNK + 1].copy(),
                  trav=bu.state_trav_label(st), conf=st[bu.CH_CONF].copy(),
                  age=np.where(obs, 2.0, 1e4).astype(np.float32),
                  hits=obs.astype(np.float32) * 4, terrain=np.zeros(obs.shape, np.uint8))


def _perception(person=False):
    def run(p):
        p.bev = _bev()
        p.odom = OdometryResult(d_trans=0.0, d_yaw=0.0, tracking_ok=True)
        label = np.ones((PROC_H, PROC_W), np.uint8)
        depth = np.full((PROC_H, PROC_W), np.nan, np.float32)
        if person:
            label[150:260, 290:350] = DYNAMIC_CLASS
            depth[150:260, 290:350] = 1.6
        valid = np.isfinite(depth)
        p.seg = SegResult(label=label)
        p.depth = DepthResult(rel_inv=np.zeros_like(depth), depth_m=depth, valid=valid)
        return p
    return run


def test_clear_road_goes_towards_b():
    nav = PrismNavigator(perception=_perception(), use_loop_closure=False)
    nav.set_goal(0.0, 6.0)
    out = nav.step(None, 0.0)
    assert out.goal.mode == "open"
    assert out.v_mps > 0 and DECISIONS[out.decision.kind] != "STOP"
    d = out.as_dict()
    assert d["decision"] in DECISIONS and d["rule"].startswith(("R", "G"))


def test_person_in_front_stops_or_steers_and_expires():
    nav = PrismNavigator(perception=_perception(person=True), use_loop_closure=False)
    nav.set_goal(0.0, 6.0)
    out = nav.step(None, 0.0)
    assert nav.dynamic.last.active.sum() > 0               # image path -> dynamic cells
    assert out.goal.mode != "open"                         # the straight line is blocked
    assert out.decision.kind == STOP or abs(out.w_radps) > 0.1
    nav._perception = _perception(person=False)           # the person walks away
    out = nav.step(None, 0.5)
    assert nav.dynamic.last.active.sum() > 0               # still remembered at 0.5 s
    out = nav.step(None, 1.0)
    assert nav.dynamic.last.active.sum() == 0              # expired in under 1 s


def test_arrival_is_g0_stop():
    nav = PrismNavigator(perception=_perception(), use_loop_closure=False)
    nav.set_goal(0.0, 0.1)
    out = nav.step(None, 0.0)
    assert out.rule == "G0" and out.v_mps == 0.0 and out.w_radps == 0.0
