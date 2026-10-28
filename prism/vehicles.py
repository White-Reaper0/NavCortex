"""Vehicle geometry as parameters: swap the chassis, keep the models.

prism's networks never see the vehicle.  They output relative depth, terrain
classes and a traversability/risk map; *whether the vehicle can drive there* is
decided afterwards against four numbers - camera height (which fixes metric
scale), chassis clearance, step limit and footprint.  So a different vehicle is
a different JSON file in `configs/vehicles/`, not a retraining run.

    from prism.vehicles import apply_vehicle_profile
    apply_vehicle_profile("wave_rover")          # before any stage is constructed

or set the environment variable ``PRISM_VEHICLE=wave_rover``; the runtime and
the ROS 2 node apply it at start-up.

`prism/config.py` is frozen contract, so this module does not edit it: it
updates the live `CFG.cam` / `CFG.ugv` dataclass instances in place, which every
module reads at call time.  Apply a profile *before* constructing stages - a few
helpers cache footprint-sized kernels at construction, and the BEV footprint
cache is cleared here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Union

from .config import CFG, ROOT

PROFILE_DIR = ROOT / "configs" / "vehicles"
_CAM_KEYS = {"height_above_ground_m", "hfov_deg", "pitch_deg"}
_UGV_KEYS = {"width_m", "length_m", "clearance_m", "max_step_m", "max_slope_deg",
             "max_speed_mps", "brake_distance_m"}


def list_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))


def load_profile(name_or_path: Union[str, Path]) -> dict:
    p = Path(name_or_path)
    if not p.suffix:
        p = PROFILE_DIR / f"{name_or_path}.json"
    if not p.exists():
        raise FileNotFoundError(f"vehicle profile {name_or_path!r} not found; "
                                f"available: {', '.join(list_profiles())}")
    prof = json.loads(p.read_text())
    unknown = (set(prof.get("camera", {})) - _CAM_KEYS) | (set(prof.get("vehicle", {})) - _UGV_KEYS)
    if unknown:
        raise ValueError(f"profile {p.name}: unknown keys {sorted(unknown)}")
    return prof


def validate(prof: dict) -> None:
    v = {**_current()["vehicle"], **prof.get("vehicle", {})}
    c = {**_current()["camera"], **prof.get("camera", {})}
    if not c["height_above_ground_m"] > 0:
        raise ValueError("camera height must be positive: it is the metric anchor")
    if not 0 < v["max_step_m"] <= v["clearance_m"]:
        raise ValueError("need 0 < max_step_m <= clearance_m")
    if not (v["width_m"] > 0 and v["length_m"] > 0 and v["max_speed_mps"] > 0
            and v["brake_distance_m"] > 0):
        raise ValueError("footprint, speed and braking distance must be positive")


def _current() -> dict:
    return {"camera": {k: getattr(CFG.cam, k) for k in _CAM_KEYS},
            "vehicle": {k: getattr(CFG.ugv, k) for k in _UGV_KEYS}}


def apply_vehicle_profile(name_or_path: Union[str, Path, None] = None) -> dict:
    """Load a profile and write it into the live config.  Returns the profile.

    With no argument, uses ``$PRISM_VEHICLE`` if set, else does nothing.
    """
    if name_or_path is None:
        name_or_path = os.environ.get("PRISM_VEHICLE")
        if not name_or_path:
            return {"name": "default", **_current()}
    prof = load_profile(name_or_path)
    validate(prof)
    for k, val in prof.get("camera", {}).items():
        setattr(CFG.cam, k, float(val))
    for k, val in prof.get("vehicle", {}).items():
        setattr(CFG.ugv, k, float(val))
    try:                                   # footprint offsets are cached per margin
        from .nav import bev_utils
        bev_utils._FP_CACHE.clear()
    except Exception:
        pass
    return prof


def describe() -> str:
    c, u = CFG.cam, CFG.ugv
    return (f"camera {c.height_above_ground_m*100:.0f} cm high, {c.hfov_deg:.0f} deg HFOV | "
            f"chassis {u.width_m*100:.0f}x{u.length_m*100:.0f} cm, clearance "
            f"{u.clearance_m*100:.1f} cm, step limit {u.max_step_m*100:.1f} cm, "
            f"max {u.max_speed_mps:.2f} m/s")


if __name__ == "__main__":
    import copy
    snap = copy.deepcopy(_current())
    for name in list_profiles():
        prof = apply_vehicle_profile(name)
        print(f"{name:18s} {describe()}")
    # restore
    for k, v in snap["camera"].items():
        setattr(CFG.cam, k, v)
    for k, v in snap["vehicle"].items():
        setattr(CFG.ugv, k, v)
