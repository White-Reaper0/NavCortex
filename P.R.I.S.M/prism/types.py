"""Data contracts shared by every prism module.

Dense maps are all at (PROC_H, PROC_W) unless noted. BEV grids are (bev.H, bev.W)
with row 0 = farthest forward, row H-1 = ego, column W/2 = ego centreline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .config import CFG, N_TERRAIN, N_TRAV, N_ACTIONS


@dataclass
class DepthResult:
    """Output of the monocular depth stage."""
    rel_inv: np.ndarray                 # (H,W) float32 raw relative inverse depth from the network
    depth_m: np.ndarray                 # (H,W) float32 metric-aligned optical-axis depth, metres
    scale: float = 1.0                  # a in 1/D = a*q + b
    shift: float = 0.0                  # b
    align_residual: float = 0.0         # RMS residual of the inverse-depth fit
    align_inliers: int = 0              # number of sparse references used
    valid: Optional[np.ndarray] = None  # (H,W) bool, False where depth must not be trusted


@dataclass
class SegResult:
    """Output of the terrain semantic stage (prism-7 taxonomy)."""
    logits: Optional[np.ndarray] = None      # (C,H,W) float16, optional (memory)
    label: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), np.uint8))   # (H,W) uint8
    prob_max: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), np.float32))  # (H,W)
    entropy: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), np.float32))   # (H,W) normalised 0..1


@dataclass
class GeometryResult:
    """Ground-plane geometry derived from metric depth."""
    points_cam: np.ndarray              # (H,W,3) float32 XYZ in camera frame (right, down, forward)
    points_veh: np.ndarray              # (H,W,3) float32 XYZ in vehicle frame (right, forward, up)
    height_above_ground: np.ndarray     # (H,W) float32 metres, + above local ground
    slope_deg: np.ndarray               # (H,W) float32 surface slope
    roughness: np.ndarray               # (H,W) float32 local height std, metres
    ground_normal: np.ndarray           # (3,) float32
    ground_d: float = 0.0
    ground_inlier_frac: float = 0.0


@dataclass
class TraversabilityResult:
    """Per-pixel traversability, 4 classes: safe / risky / obstacle / unknown."""
    prob: np.ndarray                    # (4,H,W) float32
    label: np.ndarray                   # (H,W) uint8
    risk: np.ndarray                    # (H,W) float32 continuous risk in [0,1]


@dataclass
class UncertaintyResult:
    """Learned confidence in the depth and terrain predictions."""
    depth_conf: np.ndarray              # (H,W) float32 in [0,1]
    seg_conf: np.ndarray                # (H,W) float32 in [0,1]
    fused_conf: np.ndarray              # (H,W) float32 in [0,1]
    mean_conf: float = 0.0


@dataclass
class Pose:
    """Vehicle pose in the local map frame: x right, y forward, yaw CCW from +y."""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw], np.float32)


@dataclass
class OdometryResult:
    pose: Pose = field(default_factory=Pose)
    d_trans: float = 0.0                # metres travelled since previous frame
    d_yaw: float = 0.0                  # radians turned since previous frame
    speed_mps: float = 0.0
    n_matches: int = 0
    n_inliers: int = 0
    tracking_ok: bool = True
    track_quality: float = 1.0          # 0..1
    keypoints: Optional[np.ndarray] = None   # (N,2) float32 image coords
    flow: Optional[np.ndarray] = None        # (N,4) float32 x0,y0,x1,y1 inlier matches
    trajectory: Optional[np.ndarray] = None  # (T,2) float32 xy history


@dataclass
class PlaceResult:
    descriptor: np.ndarray              # (D,) float32 L2-normalised global descriptor
    best_match_idx: int = -1            # index into the visited database, -1 = none
    best_score: float = 0.0             # cosine similarity of the best match
    is_revisit: bool = False
    loop_gap_frames: int = 0
    db_size: int = 0


@dataclass
class BEVMap:
    """Rolling top-down 2.5D local map. All grids (H, W)."""
    height: np.ndarray                  # float32 metres above local ground, NaN = unobserved
    trav_prob: np.ndarray               # (4,H,W) float32
    trav: np.ndarray                    # uint8 argmax class
    conf: np.ndarray                    # float32 0..1
    age: np.ndarray                     # float32 frames since last observation
    hits: np.ndarray                    # float32 accumulated observation count
    terrain: np.ndarray                 # uint8 dominant prism-7 terrain class per cell

    @staticmethod
    def empty() -> "BEVMap":
        H, W = CFG.bev.H, CFG.bev.W
        return BEVMap(
            height=np.full((H, W), np.nan, np.float32),
            trav_prob=np.zeros((N_TRAV, H, W), np.float32),
            trav=np.full((H, W), 3, np.uint8),          # UNKNOWN
            conf=np.zeros((H, W), np.float32),
            age=np.full((H, W), 1e4, np.float32),
            hits=np.zeros((H, W), np.float32),
            terrain=np.zeros((H, W), np.uint8),
        )


@dataclass
class Trajectory:
    """A candidate motion rollout evaluated by the planner."""
    action: int                         # index into config.ACTIONS
    xy: np.ndarray                      # (T,2) float32 path in vehicle frame, metres
    yaw: np.ndarray                     # (T,) float32
    clearance: np.ndarray               # (T,) float32 min lateral clearance, metres
    max_step: float = 0.0               # worst height step under the footprint
    unknown_frac: float = 0.0
    mean_conf: float = 1.0
    collision_risk: float = 0.0         # world-model predicted risk in [0,1]
    cost: float = 0.0
    feasible: bool = True
    reject_reason: str = ""


@dataclass
class WorldModelPrediction:
    """One rollout of the tiny neural world model."""
    action: int
    next_states: np.ndarray             # (T, state_dim) float32
    occ_forecast: np.ndarray            # (T, 16, 16) float32 predicted obstacle occupancy
    trav_forecast: np.ndarray           # (T,) float32 predicted mean traversability
    collision_risk: np.ndarray          # (T,) float32 per-step collision risk
    risk_total: float = 0.0


@dataclass
class Decision:
    kind: int = 0                       # index into config.DECISIONS
    action: int = 0                     # index into config.ACTIONS
    speed_mps: float = 0.0
    reason: str = ""
    risk: float = 0.0
    confidence: float = 1.0
    unknown_frac: float = 0.0
    policy_source: str = "supervisor"   # "rl" | "supervisor" | "rl+supervisor"


@dataclass
class FramePacket:
    """Everything prism knows about one frame. Written to the per-clip cache."""
    clip_id: str
    idx: int
    t: float
    rgb: Optional[np.ndarray] = None            # (H,W,3) uint8 BGR at PROC resolution
    depth: Optional[DepthResult] = None
    seg: Optional[SegResult] = None
    geom: Optional[GeometryResult] = None
    trav: Optional[TraversabilityResult] = None
    unc: Optional[UncertaintyResult] = None
    odom: Optional[OdometryResult] = None
    place: Optional[PlaceResult] = None
    bev: Optional[BEVMap] = None
    trajectories: list = field(default_factory=list)      # list[Trajectory]
    wm_preds: list = field(default_factory=list)          # list[WorldModelPrediction]
    decision: Optional[Decision] = None
    timings_ms: dict = field(default_factory=dict)
