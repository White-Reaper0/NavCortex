import math

import numpy as np
import pytest

from prism.perception.pose_graph import (PoseGraph2D, relative_prism, compose,
                                           _to_std, _from_std, _gtsam)
from prism.perception.imu_fusion import ImuFusion, G


def _square_run(rng, yaw_bias=0.004):
    true = []
    x = y = yaw = 0.0
    for _ in range(4):
        for _ in range(20):
            true.append((x, y, yaw))
            x += -math.sin(yaw) * 0.2
            y += math.cos(yaw) * 0.2
        yaw += math.pi / 2
    true.append((x, y, yaw))
    return np.array(true)


@pytest.mark.parametrize("backend", ["numpy"] + (["gtsam"] if _gtsam() is not None else []))
def test_loop_closure_removes_drift_and_rejects_false_loop(backend):
    rng = np.random.default_rng(0)
    true = _square_run(rng)
    pg = PoseGraph2D(backend=backend)
    est = np.array(true[0])
    pg.add_node(est)
    for k in range(1, len(true)):
        z = relative_prism(true[k - 1], true[k]) + np.array(
            [rng.normal(0, 0.01), rng.normal(0, 0.01), 0.004])
        est = _from_std(compose(_to_std(est), z))
        pg.add_node(est)
        pg.add_odometry(k - 1, k, z, sigma=(0.02, 0.02, 0.01))
    before = np.linalg.norm(pg.poses()[-1, :2] - true[-1, :2])
    pg.add_loop(0, len(true) - 1, relative_prism(true[0], true[-1]))
    pg.add_loop(5, 51, relative_prism(true[5], true[51]) + np.array([2.0, 1.5, 1.0]))
    info = pg.optimize()
    after = np.linalg.norm(pg.poses()[-1, :2] - true[-1, :2])
    assert after < 0.2 * before
    assert info["accepted_loops"] == 1 and info["rejected_loops"] == 1


def _drive(fus, vo_scale=0.5, T=25.0, seed=0, const_speed=False):
    rng = np.random.default_rng(seed)
    t = v = 0.0
    acc_d = 0.0
    next_vo, t_last = 0.0, 0.0
    out = []
    while t < T:
        a = 0.0 if const_speed else 0.6 * math.sin(2 * math.pi * t / 4.0)
        if const_speed and t < 1.0:
            a = 0.5
        dt = 1.0 / 200
        v = max(0.0, v + a * dt)
        fus.add_imu(t, np.array([0, 0, rng.normal(0, 0.005)]),
                    np.array([0, (a if v > 0 else 0) + rng.normal(0, 0.05), G]))
        acc_d += v * dt
        t += dt
        if t >= next_vo:
            out.append(fus.fuse(t, vo_scale * acc_d, 0.0, t - t_last))
            t_last, acc_d = t, 0.0
            next_vo += 1 / 30
    return out


def test_imu_recovers_vo_scale():
    fus = ImuFusion()
    _drive(fus, vo_scale=0.5, T=45.0)
    assert abs(fus.scale - 2.0) < 0.3
    assert abs(fus.last_estimate - 2.0) < 0.4


def test_no_false_zupt_at_constant_speed():
    fus = ImuFusion()
    out = _drive(fus, vo_scale=1.0, T=8.0, const_speed=True)
    moving = [m for m in out[60:]]            # after the 1 s acceleration
    assert not any(m.stationary for m in moving)
    assert sum(m.d_trans for m in moving) > 0


def test_without_imu_vo_passes_through():
    fus = ImuFusion()
    m = fus.fuse(0.1, 0.03, 0.01, 1 / 30)
    assert m.source == "vo" and m.d_trans == 0.03 and m.d_yaw == 0.01


def test_loop_closure_manager_corrects_current_pose():
    from prism.perception.loop_closure import LoopClosureManager
    rng = np.random.default_rng(3)
    true = _square_run(rng)
    # drifting odometry: a small yaw bias per step
    est = [np.array(true[0])]
    for k in range(1, len(true)):
        z = relative_prism(true[k - 1], true[k]) + np.array([0.0, 0.0, 0.004])
        est.append(_from_std(compose(_to_std(est[-1]), z)))
    kf_true = {}

    def verifier(i, j):
        return relative_prism(kf_true[i], kf_true[j]), 80

    lc = LoopClosureManager(kf_dist_m=0.35, min_gap_kf=10, backend="numpy", verifier=verifier)
    for k, p in enumerate(est):
        kf = lc.maybe_keyframe(k, p)
        if kf is not None:
            kf_true[kf] = true[k]
    before = np.linalg.norm(est[-1][:2] - true[-1][:2])
    res = lc.on_revisit(matched_frame=0, cur_gray=None, cur_pose=est[-1])
    assert res is not None and res[1].accepted
    after = np.linalg.norm(res[0][:2] - true[-1][:2])
    assert after < 0.3 * before
