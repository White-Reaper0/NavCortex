"""prism global configuration - the single source of truth for every module.

    from prism.config import CFG
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VIDEO_IN = ROOT / "video" / "input.mp4"
CLIP_DIR = ROOT / "clips"
WORK_DIR = ROOT / "work"
CACHE_DIR = WORK_DIR / "cache"          # per-clip perception artifacts (.npz)
DATA_DIR = WORK_DIR / "dataset"         # training frames + pseudo labels
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "output"
LOG_DIR = ROOT / "logs"

# ---------------------------------------------------------------- clips
N_CLIPS = 5
CLIP_SECONDS = 10.0
CLIP_FPS = 30                 # processing / output fps
CLIP_W, CLIP_H = 1280, 720    # clip resolution on disk
CLIP_IDS = [f"clip_{i:02d}" for i in range(1, N_CLIPS + 1)]

# ---------------------------------------------------------------- inference sizes
DEPTH_INPUT = 518             # Depth Anything V2 ViT-S input side
SEG_INPUT = (512, 288)        # (w, h) segmentation student input
PROC_W, PROC_H = 640, 360     # canonical resolution shared by all dense maps

# ---------------------------------------------------------------- terrain taxonomy
# prism-7: off-road taxonomy (GOOSE / RELLIS-3D / ORFD inspired), distilled from
# an ADE20K teacher and fine-tuned on frames from this UGV camera.
TERRAIN_CLASSES = [
    "sky",          # 0
    "trail",        # 1 drivable ground: road, dirt track, path, earth, sand, gravel
    "grass",        # 2 low vegetation: traversable but support is uncertain
    "rough_veg",    # 3 bush / high vegetation / hedge
    "obstacle",     # 4 tree trunk, rock, wall, fence, pole, building
    "water",        # 5 puddle / river / lake
    "dynamic",      # 6 person, animal, vehicle
]
N_TERRAIN = len(TERRAIN_CLASSES)
IGNORE_INDEX = 255
TERRAIN_COLORS_BGR = np.array([          # OpenCV BGR order
    (235, 206, 135),   # sky        light blue
    (90, 190, 255),    # trail      tan/orange
    (95, 215, 120),    # grass      green
    (45, 130, 60),     # rough_veg  dark green
    (60, 60, 235),     # obstacle   red
    (235, 175, 60),    # water      cyan-blue
    (225, 80, 225),    # dynamic    magenta
], dtype=np.uint8)

# Prior on "a wheeled UGV can drive on this surface" in [0,1]. A cue, not a guarantee.
TERRAIN_DRIVE_PRIOR = np.array([0.00, 0.95, 0.62, 0.12, 0.00, 0.05, 0.00], dtype=np.float32)

# ---------------------------------------------------------------- traversability
TRAV_CLASSES = ["safe", "risky", "obstacle", "unknown"]
SAFE, RISKY, OBSTACLE, UNKNOWN = 0, 1, 2, 3
N_TRAV = 4
TRAV_COLORS_BGR = np.array([
    (110, 230, 120),   # safe     green
    (60, 200, 255),    # risky    amber
    (60, 60, 235),     # obstacle red
    (150, 150, 150),   # unknown  grey
], dtype=np.uint8)

# ---------------------------------------------------------------- decisions
DECISIONS = ["GO", "SLOW", "REROUTE", "STOP"]
GO, SLOW, REROUTE, STOP = 0, 1, 2, 3
DECISION_COLORS_BGR = [(110, 230, 120), (60, 200, 255), (255, 190, 70), (60, 60, 235)]

# ---------------------------------------------------------------- actions
ACTIONS = ["FORWARD", "LEFT", "RIGHT", "SLOW", "REROUTE", "STOP"]
N_ACTIONS = len(ACTIONS)
# (linear m/s, angular rad/s) command prototypes at the vehicle scale below
ACTION_CMD = np.array([
    [1.00, 0.00],    # FORWARD
    [0.70, +0.90],   # LEFT
    [0.70, -0.90],   # RIGHT
    [0.35, 0.00],    # SLOW
    [0.45, +1.60],   # REROUTE (evasive arc; turn sign chosen by the planner)
    [0.00, 0.00],    # STOP
], dtype=np.float32)


@dataclass
class CameraConfig:
    """Pinhole intrinsics at PROC_W x PROC_H.

    The source is an uncalibrated action-cam POV, so the field of view is a documented
    assumption, not a calibration result. Camera height above ground is the anchor used
    to resolve monocular scale (see perception/geometry.py).
    """
    width: int = PROC_W
    height: int = PROC_H
    hfov_deg: float = 92.0
    height_above_ground_m: float = 0.12
    pitch_deg: float = -6.0            # nose-down tilt, negative looks down

    @property
    def fx(self) -> float:
        return (self.width / 2.0) / float(np.tan(np.deg2rad(self.hfov_deg) / 2.0))

    @property
    def fy(self) -> float:
        return self.fx

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)


@dataclass
class UGVConfig:
    """Vehicle envelope, expressed at the scale implied by camera height above ground."""
    width_m: float = 0.22
    length_m: float = 0.34
    clearance_m: float = 0.045      # chassis clearance
    max_step_m: float = 0.030       # height over local ground that becomes non-traversable
    max_slope_deg: float = 22.0
    max_speed_mps: float = 1.2
    brake_distance_m: float = 0.45  # stopping space required at max speed
    full_scale_factor: float = 6.0  # RC POV -> nominal full-size UGV, reporting only


@dataclass
class BEVConfig:
    """Top-down 2.5D local map. Ego at bottom-centre, +X right, +Y forward."""
    res_m: float = 0.06
    n_forward: int = 128
    n_lateral: int = 96
    z_min: float = -0.45            # below this -> negative obstacle / drop-off
    z_max: float = 0.90
    decay_per_frame: float = 0.965  # observation-age confidence decay
    max_age_frames: int = 45        # older than this -> stale -> unknown

    @property
    def H(self) -> int:
        return self.n_forward

    @property
    def W(self) -> int:
        return 2 * self.n_lateral

    @property
    def range_forward_m(self) -> float:
        return self.n_forward * self.res_m

    @property
    def range_lateral_m(self) -> float:
        return self.n_lateral * self.res_m


@dataclass
class SafetyConfig:
    """Supervisor thresholds. Policy thresholds, not calibrated collision probabilities."""
    conf_unknown: float = 0.35
    conf_slow: float = 0.55
    risk_slow: float = 0.30
    risk_reroute: float = 0.55
    risk_stop: float = 0.78
    unknown_frac_slow: float = 0.30
    unknown_frac_stop: float = 0.62
    corridor_margin_m: float = 0.05
    horizon_s: float = 2.0
    n_rollout_steps: int = 12


@dataclass
class WorldModelConfig:
    state_dim: int = 96             # compact latent of the BEV world state
    action_dim: int = N_ACTIONS
    hidden: int = 192
    n_layers: int = 2
    pred_horizon: int = 6


@dataclass
class Config:
    cam: CameraConfig = field(default_factory=CameraConfig)
    ugv: UGVConfig = field(default_factory=UGVConfig)
    bev: BEVConfig = field(default_factory=BEVConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    wm: WorldModelConfig = field(default_factory=WorldModelConfig)
    device: str = "cuda"
    fp16: bool = True
    seed: int = 1337

    def ensure_dirs(self) -> None:
        for p in (CLIP_DIR, WORK_DIR, CACHE_DIR, DATA_DIR, CKPT_DIR, OUT_DIR, LOG_DIR):
            p.mkdir(parents=True, exist_ok=True)


CFG = Config()

# ---------------------------------------------------------------- ego mask
# The RC chassis and wheels occupy the bottom of every source frame, and a channel
# watermark sits top-left. Those pixels carry no scene geometry and are excluded from
# depth back-projection, VO features and BEV accumulation.
EGO_MASK_BOTTOM_FRAC = 0.185   # bottom band covered by the vehicle body
EGO_MASK_WATERMARK = (0.0, 0.0, 0.085, 0.075)  # (x0, y0, x1, y1) fractions

# ---------------------------------------------------------------- output layout
# Each entry becomes output/<dir>/clip_XX.mp4 (5 clips per folder).
OUTPUT_STAGES = [
    ("01_depth_anything_v2", "Depth Anything V2-Small monocular depth"),
    ("02_terrain_segmentation", "PIDNet-S terrain semantics (distilled)"),
    ("03_traversability", "Off-road traversability / risk head"),
    ("04_visual_odometry", "ORB monocular VO (ORB-SLAM3-style front end)"),
    ("05_place_recognition", "Lightweight visual place recognition"),
    ("06_uncertainty", "Depth / terrain confidence estimation"),
    ("07_lidar_like_pointcloud", "LiDAR-like 3D reconstruction from mono depth"),
    ("07b_lidar_3d_view", "Immersive 3D LiDAR view - pose-accumulated point cloud"),
    ("08_bev_25d_map", "Top-down 2.5D coloured-cell local map"),
    ("09_world_model", "Tiny neural world model rollouts"),
    ("10_rl_policy", "PPO policy trained inside the world model"),
    ("11_final_dashboard", "Full prism synchronized dashboard"),
]
