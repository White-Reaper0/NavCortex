"""Goal-directed navigation settings (Point A -> Point B).

Kept out of `prism/config.py` on purpose: that file is frozen contract (see
CONTRACT.md) and every perception module depends on it byte-for-byte.  The
settings below are only read by the goal layer added on top of the arc planner:
`goal_planner.py`, `dstar_lite.py`, `goal_map.py` and
`perception/dynamic_layer.py`.

Units follow the rest of prism: metres, seconds, radians, and the vehicle frame
is X right, Y forward, yaw CCW from +Y.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GoalConfig:
    """When is Point B reached, and how is it pursued."""
    arrive_tol_m: float = 0.25          # inside this radius the mission is complete
    lookahead_m: float = 1.60           # D* Lite path point the arcs steer towards
    #: "open ground": the straight segment to B (or to the edge of the local map,
    #: whichever is nearer) must be at least this observed-and-free to skip D* Lite
    open_min_observed: float = 0.70
    #: re-run the global search at least this often even if nothing changed
    replan_period_s: float = 1.0
    #: per-arc progress is measured as reduction in distance to the steer point
    progress_weight: float = 1.6
    #: cap on D* Lite expansions per control cycle; a bigger search resumes next cycle
    max_expansions: int = 6000


@dataclass
class GlobalGridConfig:
    """World-fixed coarse grid that D* Lite searches over.

    The rolling 2.5D map only covers 7.7 m x 11.5 m around the vehicle and forgets
    cells after `CFG.bev.max_age_frames`.  Planning to a Point B that may be tens of
    metres away needs memory of where the vehicle has *already found a dead end*,
    so observations are folded into a coarser world-frame grid as well.
    """
    res_m: float = 0.12                 # 2 x 2 BEV cells per global cell
    size_m: float = 60.0                # square, centred on the start pose
    #: traversal cost per metre, by evidence.  Unseen ground COSTS, it is never free.
    cost_safe: float = 1.0
    cost_risky: float = 2.5
    cost_unknown: float = 4.0           # never observed, or observed with low confidence
    #: OBSTACLE and inflated-dynamic cells are impassable (infinite cost)
    obstacle_p: float = 0.55            # p_obstacle above this -> blocked
    risky_p: float = 0.45               # p_risky above this -> risky cost
    min_conf: float = 0.35              # below this an observation counts as unknown
    #: blocked cells are dilated by the vehicle half-width so the search plans for
    #: the footprint centre, not a point robot
    inflate_m: float = 0.16
    #: beyond the hard inflation, cost per metre rises linearly to (1 + gain) x
    #: over this distance from a blocked cell, so routes stay centred in corridors
    soft_inflate_m: float = 0.50
    soft_inflate_gain: float = 3.0


@dataclass
class DynamicConfig:
    """Moving obstacles (people, animals, vehicles: prism-7 class 6)."""
    ttl_s: float = 0.8                  # a dynamic cell expires in under 1 s
    inflate_m: float = 0.30             # extra clearance around a moving thing
    max_range_m: float = 6.0            # ignore far, unreliable dynamic pixels
    min_pixels: int = 25                # fewer dynamic pixels than this is noise
    min_cells: int = 2                  # fewer occupied BEV cells than this is noise


@dataclass
class ImuConfig:
    """Rover IMU fusion (not exercised on the recorded footage: it has no IMU)."""
    gyro_weight: float = 0.98           # complementary filter: gyro for short-term yaw
    gyro_bias_alpha: float = 0.01       # rate the gyro-minus-VO residual is learned at
    scale_window_s: float = 2.0         # window for the VO-vs-IMU scale estimate
    scale_min_motion_m: float = 0.15    # do not estimate scale while nearly stationary
    scale_alpha: float = 0.05           # EMA rate for the scale correction
    scale_clip: tuple = (0.4, 2.5)      # never trust a correction outside this band
    stationary_acc_tol: float = 0.15    # m/s^2 around gravity -> zero-velocity update
    stationary_gyro_tol: float = 0.03   # rad/s
    zupt_vo_speed: float = 0.03         # m/s: VO must also read ~still for a ZUPT
    zupt_min_frames: int = 6            # ...for this many VO updates in a row
    bias_min_frames: int = 45           # 1.5 s quiet before biases are learned (no command info)


@dataclass
class NavConfig:
    goal: GoalConfig = field(default_factory=GoalConfig)
    grid: GlobalGridConfig = field(default_factory=GlobalGridConfig)
    dynamic: DynamicConfig = field(default_factory=DynamicConfig)
    imu: ImuConfig = field(default_factory=ImuConfig)
    fps: float = 30.0                   # control/perception rate the TTLs are counted at


NAV = NavConfig()
