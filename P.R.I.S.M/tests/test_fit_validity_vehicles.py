import copy

import numpy as np
import pytest

from prism.config import CFG, PROC_H, PROC_W
from prism.perception.fit_validity import FitValidityTracker, region_validity
from prism.perception.geometry import GroundFit, ray_grid
from prism import vehicles


def _plane_q(fit):
    m = ray_grid(PROC_H, PROC_W)
    inv = m @ (fit.normal / fit.height)
    return np.where(inv > 0, inv / fit.a, 0.0).astype(np.float32)


def _fit():
    n = np.array([0.0, np.cos(np.deg2rad(-6)), np.sin(np.deg2rad(-6))], np.float32)
    return GroundFit(a=2.0, b=0.0, normal=n / np.linalg.norm(n), height=0.12, ok=True)


def test_consistent_ground_is_valid_and_bad_tile_is_not():
    fit = _fit()
    q = _plane_q(fit)
    seg = np.full((PROC_H, PROC_W), 1, np.uint8)
    base = q > 0
    v, bad, n = region_validity(q, base, seg, fit)
    assert bad == 0 and n > 0 and v.all()
    q2 = q.copy()
    q2[300:340, 0:80] *= 3.0                        # depth patch that disagrees with the plane
    v2, bad2, _ = region_validity(q2, base, seg, fit)
    assert bad2 >= 1 and not v2[320, 40] and v2[320, 600]


def test_frame_invalid_without_any_good_fit_and_after_long_failure():
    fit = _fit()
    q = _plane_q(fit)
    seg = np.ones(q.shape, np.uint8)
    tr = FitValidityTracker(max_stale_frames=3)
    r = tr(q, q > 0, seg, fit, fit_ok_this_frame=False)
    assert not r.frame_ok and not r.valid.any()
    assert tr(q, q > 0, seg, fit, True).frame_ok
    for _ in range(3):
        assert tr(q, q > 0, seg, fit, False).frame_ok         # short dropout: reuse last fit
    r = tr(q, q > 0, seg, fit, False)
    assert not r.frame_ok and not r.valid.any()


@pytest.fixture
def restore_cfg():
    cam, ugv = copy.deepcopy(CFG.cam), copy.deepcopy(CFG.ugv)
    yield
    CFG.cam.__dict__.update(cam.__dict__)
    CFG.ugv.__dict__.update(ugv.__dict__)


def test_profiles_load_and_apply(restore_cfg):
    names = vehicles.list_profiles()
    assert {"rc_pov", "wave_rover", "example_large_ugv"} <= set(names)
    vehicles.apply_vehicle_profile("wave_rover")
    assert CFG.cam.height_above_ground_m == pytest.approx(0.12)
    assert CFG.ugv.clearance_m == pytest.approx(0.045)
    assert CFG.ugv.max_step_m == pytest.approx(0.030)


def test_same_evidence_judged_per_vehicle(restore_cfg):
    """A 10 cm step blocks the rover but not the big chassis - no retraining."""
    from prism.nav import bev_utils as bu
    from prism.nav.planner import Planner
    st = bu.synthetic_state("wall", wall_dist_m=1.0, wall_height_m=0.10)
    st[bu.CH_SAFE:bu.CH_UNK + 1] = bu.synthetic_state("clear")[bu.CH_SAFE:bu.CH_UNK + 1]
    vehicles.apply_vehicle_profile("rc_pov")
    fwd = [t for t in Planner().plan(st) if t.action == 0][0]
    assert not fwd.feasible and "R-P1" in fwd.reject_reason
    vehicles.apply_vehicle_profile("example_large_ugv")
    fwd = [t for t in Planner().plan(st) if t.action == 0][0]
    assert "R-P1" not in fwd.reject_reason


def test_bad_profile_rejected(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"vehicle": {"max_step_m": 0.2, "clearance_m": 0.1}}')
    with pytest.raises(ValueError):
        vehicles.validate(vehicles.load_profile(p))
