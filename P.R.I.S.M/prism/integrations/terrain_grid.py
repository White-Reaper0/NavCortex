"""ROS-free layered terrain grid inspired by ANYbotics grid_map (BSD-3-Clause).

No grid_map source code is copied or vendored; only the multilayer concept is
reused.  Each cell carries named layers with an explicit unknown marker (NaN /
``UNKNOWN`` class), so unobserved space is never reported as traversable.

Layers produced from the existing prototype ``BEVMap``:

* ``elevation_m``  - robust top surface (NaN where unobserved)
* ``support_m``    - local supporting surface
* ``height_step_m``- height above the local ground
* ``slope_deg``    - local gradient magnitude of the elevation layer
* ``roughness_m``  - local elevation standard deviation (windowed)
* ``trav_prob``    - (4, H, W) per-class probability
* ``trav_class``   - SAFE/RISKY/OBSTACLE/UNKNOWN
* ``confidence``, ``hits``, ``age_frames``, ``observed``

Layers deliberately **not** produced: per-cell variance and surface normals.
``BEVMap`` does not retain a variance estimate or a normal fit, so synthesising
them would fabricate measurement semantics.  They are recorded as gaps rather
than approximated.

The grid is metre-based because ``BEVMap`` is metre-based, but the metre scale
inherits whatever calibration the mapping stage used.  ``metric_status`` states
that provenance; a ``REFERENCE_INPUT`` grid is an interface test, not a
validated terrain map.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..config import CFG


LAYER_NAMES = (
    "elevation_m",
    "support_m",
    "height_step_m",
    "slope_deg",
    "roughness_m",
    "trav_prob",
    "trav_class",
    "confidence",
    "hits",
    "age_frames",
    "observed",
)

# Mirrors prism.config constants without importing the frozen config module.
_UNKNOWN = 3


@dataclass(frozen=True)
class TerrainGrid:
    """Immutable multilayer grid; every layer is indexed ``[forward, lateral]``."""

    resolution_m: float
    layers: dict[str, np.ndarray]
    metric_status: str
    origin_description: str

    def layer(self, name: str) -> np.ndarray:
        return self.layers[name]

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.layers["elevation_m"].shape)  # type: ignore[return-value]

    def summary(self) -> dict[str, Any]:
        unknown = self.layers["trav_class"] == _UNKNOWN
        total = int(unknown.size)
        observed = ~unknown
        known = np.isfinite(self.layers["elevation_m"]) & ~unknown
        slope = self.layers["slope_deg"]
        rough = self.layers["roughness_m"]
        return {
            "status": "EXPERIMENTAL",
            "metric_status": self.metric_status,
            "origin": self.origin_description,
            "resolution_m": self.resolution_m,
            "shape": list(self.shape),
            "layers": list(self.layers),
            "unsupported_layers": {
                "variance": "BEVMap retains no per-cell variance estimate",
                "normals": "BEVMap retains no surface-normal fit",
            },
            "fractions": {
                "unknown": unknown.sum() / total if total else 0.0,
                "observed": observed.sum() / total if total else 0.0,
            },
            "known_cells": int(known.sum()),
            "slope_deg_median": _finite_median(slope),
            "roughness_m_median": _finite_median(rough),
            "unknown_is_traversable": False,
            "note": (
                "Unknown cells are retained as UNKNOWN with NaN geometry; they are "
                "absence of measurement, not evidence of free space."
            ),
        }


def terrain_grid_from_bev(
    bev: Any,
    *,
    resolution_m: float | None = None,
    metric_status: str = "EXPERIMENTAL_UNVERIFIED",
    origin_description: str = "vehicle frame, BEV prototype grid",
) -> TerrainGrid:
    """Convert an existing ``BEVMap`` into explicit terrain layers.

    The function is read-only and duck-typed on the published BEVMap contract
    (``height``, ``trav``, ``trav_prob``, ``conf``, ``age``, ``hits``) plus the
    published derived attributes ``height_step``, ``local_ground`` and
    ``observed``.  Missing derived attributes are recomputed, never guessed.
    """
    height = np.asarray(bev.height, dtype=np.float64)
    if height.ndim != 2:
        raise ValueError("BEVMap.height must be a 2-D array")
    resolution = float(CFG.bev.res_m if resolution_m is None else resolution_m)
    _require_resolution(resolution)

    trav_class = np.asarray(bev.trav, dtype=np.uint8)
    if trav_class.shape != height.shape:
        raise ValueError("BEVMap.trav shape must match height")
    observed = np.asarray(getattr(bev, "observed", np.isfinite(height)), dtype=bool)
    if observed.shape != height.shape:
        raise ValueError("BEVMap.observed shape must match height")

    elevation = np.where(observed, height, np.nan)
    step_source = getattr(bev, "height_step", None)
    step = (np.asarray(step_source, dtype=np.float64) if step_source is not None
            else _height_step(elevation))
    if step.shape != height.shape:
        raise ValueError("BEVMap.height_step shape must match height")
    support_source = getattr(bev, "local_ground", None)
    support = (np.asarray(support_source, dtype=np.float64) if support_source is not None
               else elevation - step)

    slope = slope_degrees(elevation, resolution)
    roughness = terrain_roughness(elevation, resolution)
    probability = np.asarray(bev.trav_prob, dtype=np.float32)
    if probability.ndim != 3 or probability.shape[0] != 4 or probability.shape[1:] != height.shape:
        raise ValueError("BEVMap.trav_prob must have shape (4, H, W)")

    layers = {
        "elevation_m": elevation.astype(np.float32),
        "support_m": support.astype(np.float32),
        "height_step_m": step.astype(np.float32),
        "slope_deg": slope.astype(np.float32),
        "roughness_m": roughness.astype(np.float32),
        "trav_prob": probability.astype(np.float32),
        "trav_class": trav_class,
        "confidence": np.asarray(bev.conf, dtype=np.float32),
        "hits": np.asarray(bev.hits, dtype=np.float32),
        "age_frames": np.asarray(bev.age, dtype=np.float32),
        "observed": observed,
    }
    return TerrainGrid(
        resolution_m=resolution,
        layers=layers,
        metric_status=metric_status,
        origin_description=origin_description,
    )


def slope_degrees(elevation: np.ndarray, resolution_m: float) -> np.ndarray:
    """Slope magnitude in degrees from central differences; NaN where undefined.

    Cells with any non-finite 3x3 neighbourhood are NaN: a slope computed across
    an unknown cell would invent a surface that was never measured.
    """
    _require_resolution(resolution_m)
    elevation = np.asarray(elevation, dtype=np.float64)
    filled = np.nan_to_num(elevation, nan=0.0)
    grad_forward, grad_lateral = np.gradient(filled, resolution_m, resolution_m)
    magnitude = np.hypot(grad_forward, grad_lateral)
    angle = np.degrees(np.arctan(magnitude))
    neighbourhood_valid = _neighbourhood_all_finite(elevation, k=3)
    return np.where(neighbourhood_valid, angle, np.nan)


def terrain_roughness(elevation: np.ndarray, resolution_m: float,
                      window_m: float = 0.24) -> np.ndarray:
    """Windowed elevation standard deviation in metres; NaN where undefined."""
    _require_resolution(resolution_m)
    elevation = np.asarray(elevation, dtype=np.float64)
    k = max(3, int(round(window_m / resolution_m)) | 1)
    valid = np.isfinite(elevation)
    values = np.where(valid, elevation, 0.0)
    kernel = np.ones((k, k), np.float32)
    count = cv2.filter2D(valid.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    total = cv2.filter2D(values.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    total_sq = cv2.filter2D((values ** 2).astype(np.float32), -1, kernel,
                            borderType=cv2.BORDER_CONSTANT)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        variance = np.maximum(total_sq / count - mean ** 2, 0.0)
    roughness = np.sqrt(variance)
    # A window with fewer than 4 measured cells cannot support a spread estimate.
    roughness = np.where(count >= 4, roughness, np.nan)
    return np.where(valid, roughness, np.nan)


def _height_step(elevation: np.ndarray) -> np.ndarray:
    if not np.isfinite(elevation).any():
        return np.full(elevation.shape, np.nan)
    filled = np.nan_to_num(elevation, nan=0.0).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    ground = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel).astype(np.float64)
    return np.where(np.isfinite(elevation), elevation - ground, np.nan)


def _neighbourhood_all_finite(values: np.ndarray, *, k: int) -> np.ndarray:
    valid = np.isfinite(values).astype(np.float32)
    kernel = np.ones((k, k), np.float32)
    count = cv2.filter2D(valid, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return count >= k * k


def _finite_median(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else None


def _require_resolution(resolution_m: float) -> None:
    if not np.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("resolution_m must be a positive finite number")
