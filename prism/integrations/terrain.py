"""Conservative terrain-grid boundary for sparse or missing spatial inputs.

The layered grid itself lives in :mod:`prism.integrations.terrain_grid` and is
ROS-free.  This module only decides whether producing one is justified: a
metre-based grid inherits the calibration of whatever built it, so arbitrary
COLMAP scale must not be presented as metric terrain.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from .terrain_grid import terrain_grid_from_bev


def terrain_status(*, geometry_report: Mapping[str, object],
                   bev: Optional[Any] = None) -> dict:
    """Report why a terrain grid is or is not produced.

    ``bev`` is an optional prototype ``BEVMap``.  When it is supplied the
    layered summary is returned alongside the status; the summary keeps
    unobserved cells at ``UNKNOWN`` and never converts them to free space.
    """
    scale = geometry_report.get("physical_scale")
    if scale != "METRIC_SCALE_ALIGNED":
        return {
            "status": "BLOCKED_MISSING_INPUT",
            "reason": "sparse COLMAP geometry has arbitrary/unverified scale; metric terrain descriptors are not justified",
            "grid": None,
            "layered_adapter": "prism.integrations.terrain_grid (available only with metre-verified input)",
            "bridge": "future export may map validated cells to Grid Map layers after metric pose, calibration, and spatial measurements exist",
        }
    if bev is None:
        return {
            "status": "UNVALIDATED",
            "reason": "metric scale alone does not establish dense terrain coverage, vehicle frame alignment, or traversability",
            "grid": None,
            "layered_adapter": "prism.integrations.terrain_grid (needs a BEVMap to read)",
        }
    grid = terrain_grid_from_bev(
        bev,
        metric_status="METRIC_SCALE_ALIGNED_UNVALIDATED",
        origin_description="vehicle frame, prototype BEV grid",
    )
    summary = grid.summary()
    summary["status"] = "UNVALIDATED"
    summary["reason"] = (
        "layers were built from the prototype BEV grid; terrain accuracy, "
        "vehicle-frame alignment, and traversability are not measured"
    )
    summary["grid"] = grid
    return summary
