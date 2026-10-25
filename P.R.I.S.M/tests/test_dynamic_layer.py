import numpy as np

from prism.config import OBSTACLE
from prism.nav import bev_utils as bu
from prism.perception.dynamic_layer import DynamicObstacleLayer, DYNAMIC_CLASS


def _person(x, y, n=60):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([x + 0.15 * np.cos(a), y + 0.15 * np.sin(a), np.full(n, 0.5)], 1)
    return pts[None].astype(np.float32), np.full((1, n), DYNAMIC_CLASS, np.uint8)


def test_marks_inflates_and_expires_under_one_second():
    lay = DynamicObstacleLayer()
    pts, lab = _person(0.0, 2.0)
    u = lay.update(0.0, lab, pts)
    assert u.new_cells > 0 and u.active.sum() > 0 and u.changed
    # inflated: wider than the person's own 30 cm
    cols = np.nonzero(u.active.any(0))[0]
    assert (cols.max() - cols.min() + 1) * 0.06 > 0.30 + 0.2
    u = lay.update(0.5, None, None)
    assert u.active.sum() > 0                      # still there half a second later
    u = lay.update(0.99, None, None)
    assert u.active.sum() == 0 and u.expired_cells > 0   # gone in under one second
    assert lay.ttl_s < 1.0


def test_apply_writes_certain_obstacle():
    lay = DynamicObstacleLayer()
    pts, lab = _person(0.0, 1.5)
    u = lay.update(0.0, lab, pts)
    st = lay.apply(bu.synthetic_state("clear"), u.active)
    a = u.active
    assert np.allclose(st[bu.CH_SAFE + OBSTACLE][a], 1.0)
    assert np.allclose(st[bu.CH_CONF][a], 1.0)


def test_rolls_with_ego_motion():
    lay = DynamicObstacleLayer()
    pts, lab = _person(0.0, 3.0)
    u0 = lay.update(0.0, lab, pts)
    r0 = np.nonzero(u0.active.any(1))[0].mean()
    u1 = lay.update(0.1, None, None, d_trans=1.0, d_yaw=0.0)
    r1 = np.nonzero(u1.active.any(1))[0].mean()
    # the vehicle moved 1 m forward: the person is ~1 m (~16 rows) nearer
    assert 13 < (r1 - r0) < 19


def test_noise_is_ignored():
    lay = DynamicObstacleLayer()
    pts, lab = _person(0.0, 2.0, n=5)
    assert lay.update(0.0, lab, pts).active.sum() == 0
