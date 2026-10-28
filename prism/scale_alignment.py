"""Explicit, auditable scale alignment for an immutable COLMAP sparse model.

One reviewed distance reference determines only a global similarity scale. It
does not validate calibration, reconstruction shape, trajectory accuracy, or
localization. The native COLMAP model is never modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .colmap_inspection import ColmapPoint, SparseModel, trajectory_rows, write_ply


SCALE_REFERENCE_SCHEMA = "prism.colmap_scale_reference/v1"
SCALE_ALIGNMENT_SCHEMA = "prism.colmap_scale_alignment/v1"
_UNIT_TO_METRES = {"m": 1.0, "cm": 0.01, "mm": 0.001}


class ScaleAlignmentError(ValueError):
    """The supplied physical reference is incomplete or inconsistent."""


class ScaleStatus(str, Enum):
    METRIC_SCALE_UNVERIFIED = "METRIC_SCALE_UNVERIFIED"
    METRIC_SCALE_ALIGNED = "METRIC_SCALE_ALIGNED"


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScaleAlignmentError(f"{name} must be a non-empty string")
    return value.strip()


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ScaleAlignmentError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ScaleAlignmentError(f"{name} must be finite") from error
    if not math.isfinite(result) or (positive and result <= 0):
        raise ScaleAlignmentError(f"{name} must be a positive finite value" if positive else f"{name} must be finite")
    return result


def _xyz(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ScaleAlignmentError(f"{name} must contain exactly three coordinates")
    return tuple(_number(item, name) for item in value)


@dataclass(frozen=True, slots=True)
class LandmarkReference:
    landmark_id: str
    point3d_id: int
    reconstruction_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "landmark_id", _text(self.landmark_id, "landmark_id"))
        if isinstance(self.point3d_id, bool) or not isinstance(self.point3d_id, int) or self.point3d_id < 0:
            raise ScaleAlignmentError("point3d_id must be a non-negative integer")
        object.__setattr__(self, "reconstruction_xyz", _xyz(self.reconstruction_xyz, "reconstruction_xyz"))


@dataclass(frozen=True, slots=True)
class ScaleReference:
    first: LandmarkReference
    second: LandmarkReference
    measured_distance: float
    units: str
    measurement_provenance: Mapping[str, Any]
    reviewed: bool
    uncertainty: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.first, LandmarkReference) or not isinstance(self.second, LandmarkReference):
            raise ScaleAlignmentError("reference requires two landmark references")
        if self.first.landmark_id == self.second.landmark_id or self.first.point3d_id == self.second.point3d_id:
            raise ScaleAlignmentError("landmarks must be distinct")
        object.__setattr__(self, "measured_distance", _number(self.measured_distance, "measured_distance", positive=True))
        units = _text(self.units, "units")
        if units not in _UNIT_TO_METRES:
            raise ScaleAlignmentError("units must be one of: m, cm, mm")
        object.__setattr__(self, "units", units)
        if not isinstance(self.measurement_provenance, Mapping) or not self.measurement_provenance:
            raise ScaleAlignmentError("measurement_provenance must be a non-empty mapping")
        if not isinstance(self.reviewed, bool):
            raise ScaleAlignmentError("reviewed must be bool")
        if self.uncertainty is not None:
            uncertainty = _number(self.uncertainty, "uncertainty")
            if uncertainty < 0:
                raise ScaleAlignmentError("uncertainty must not be negative")
            object.__setattr__(self, "uncertainty", uncertainty)

    @property
    def reconstruction_distance(self) -> float:
        return float(np.linalg.norm(np.subtract(self.first.reconstruction_xyz, self.second.reconstruction_xyz)))

    @property
    def scale_supplied_units_per_colmap_unit(self) -> float:
        distance = self.reconstruction_distance
        if distance <= 0 or not math.isfinite(distance):
            raise ScaleAlignmentError("reconstruction landmark baseline must be finite and non-zero")
        return self.measured_distance / distance

    @property
    def scale_metres_per_colmap_unit(self) -> float:
        return self.scale_supplied_units_per_colmap_unit * _UNIT_TO_METRES[self.units]

    def as_dict(self) -> dict:
        def landmark(item): return {"landmark_id": item.landmark_id, "point3d_id": item.point3d_id,
                                    "reconstruction_xyz": list(item.reconstruction_xyz)}
        return {"first": landmark(self.first), "second": landmark(self.second),
                "measured_distance": self.measured_distance, "units": self.units,
                "measurement_provenance": dict(self.measurement_provenance), "reviewed": self.reviewed,
                "uncertainty": self.uncertainty}


def load_scale_reference(path: str | Path) -> ScaleReference:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema") != SCALE_REFERENCE_SCHEMA:
        raise ScaleAlignmentError("unsupported scale reference schema")
    landmarks = raw.get("landmarks")
    if not isinstance(landmarks, list) or len(landmarks) != 2:
        raise ScaleAlignmentError("scale reference must contain exactly two landmarks")
    items = [LandmarkReference(item.get("landmark_id"), item.get("point3d_id"), item.get("reconstruction_xyz"))
             for item in landmarks]
    return ScaleReference(items[0], items[1], raw.get("measured_distance"), raw.get("units"),
                          raw.get("measurement_provenance"), raw.get("reviewed"), raw.get("uncertainty"))


def verify_reference_against_model(reference: ScaleReference, model: SparseModel, *, tolerance: float = 1e-8) -> None:
    """Prevent scaling a model using invented or mismatched reconstruction points."""
    points = {point.point_id: point for point in model.points}
    for landmark in (reference.first, reference.second):
        point = points.get(landmark.point3d_id)
        if point is None:
            raise ScaleAlignmentError(f"landmark {landmark.landmark_id} references missing point3d_id {landmark.point3d_id}")
        if not np.allclose(point.xyz, landmark.reconstruction_xyz, rtol=0, atol=tolerance):
            raise ScaleAlignmentError(f"landmark {landmark.landmark_id} coordinates do not match point3d_id {landmark.point3d_id}")
    _ = reference.scale_supplied_units_per_colmap_unit


def scale_points(points: tuple[ColmapPoint, ...], scale_metres_per_unit: float) -> tuple[ColmapPoint, ...]:
    scale = _number(scale_metres_per_unit, "scale_metres_per_unit", positive=True)
    return tuple(ColmapPoint(point.point_id, tuple(float(value * scale) for value in point.xyz), point.rgb,
                             point.error_px, point.track) for point in points)


def aligned_report(reference: Optional[ScaleReference], *, calibration: Mapping[str, Any]) -> dict:
    """Status report that remains unverified until a reviewed reference exists."""
    base = {"schema": SCALE_ALIGNMENT_SCHEMA, "calibration": dict(calibration),
            "limitations": ["A single scale reference does not validate camera calibration, reconstruction shape, trajectory, or localization.",
                            "Scaled exports are not a validated terrain map and are separate from the P.R.I.S.M. mapper."]}
    if reference is None:
        return {**base, "status": ScaleStatus.METRIC_SCALE_UNVERIFIED.value,
                "reason": "no physical scale reference was supplied", "reference": None}
    # Validate even an unreviewed reference.  It may be retained for audit, but
    # a malformed or degenerate candidate must never look usable in a report.
    _ = reference.scale_supplied_units_per_colmap_unit
    payload = reference.as_dict()
    if not reference.reviewed:
        return {**base, "status": ScaleStatus.METRIC_SCALE_UNVERIFIED.value,
                "reason": "physical scale reference is supplied but not marked reviewed", "reference": payload}
    return {**base, "status": ScaleStatus.METRIC_SCALE_ALIGNED.value, "reference": payload,
            "reconstruction_baseline_arbitrary_units": reference.reconstruction_distance,
            "scale_supplied_units_per_colmap_unit": reference.scale_supplied_units_per_colmap_unit,
            "scale_metres_per_colmap_unit": reference.scale_metres_per_colmap_unit,
            "relative_reference_uncertainty": (None if reference.uncertainty is None else reference.uncertainty / reference.measured_distance)}


def write_scaled_exports(model: SparseModel, manifest: Mapping[str, object], output_dir: str | Path,
                         reference: ScaleReference) -> dict:
    if not reference.reviewed:
        raise ScaleAlignmentError("scaled exports require a reviewed physical reference")
    verify_reference_against_model(reference, model)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    scale = reference.scale_metres_per_colmap_unit
    write_ply(scale_points(model.points, scale), output / "sparse_points_scaled_m.ply")
    rows = trajectory_rows(model, manifest)
    trajectory = output / "camera_trajectory_scaled_m.csv"
    headers = ("image_id", "file", "source_frame_index", "source_time_s", "q_w", "q_x", "q_y", "q_z",
               "t_x_world_to_camera_m", "t_y_world_to_camera_m", "t_z_world_to_camera_m",
               "center_x_m", "center_y_m", "center_z_m")
    with trajectory.open("w", encoding="utf-8") as file:
        file.write(",".join(headers) + "\n")
        for row in rows:
            file.write(",".join(str(item) for item in (row["image_id"], row["file"], row["source_frame_index"], row["source_time_s"],
                *row["qvec_wxyz"], *(value * scale for value in row["tvec_world_to_camera"]),
                *(value * scale for value in row["camera_center_arbitrary_units"]))) + "\n")
    return {"scaled_point_cloud": "sparse_points_scaled_m.ply", "scaled_trajectory": "camera_trajectory_scaled_m.csv"}
