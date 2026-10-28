"""Diagnostic mapping boundary for the integrated offline replay.

The legacy navigator map is not updated from 2-D detections alone.  This module
records why no metric terrain map artifact was produced during replay.
"""
from __future__ import annotations

from typing import Mapping


def mapping_diagnostic_status(*, geometry_report: Mapping[str, object],
                              visual_motion_summary: Mapping[str, object]) -> dict:
    scale = geometry_report.get("physical_scale")
    tracking_only = bool(visual_motion_summary.get("tracking_only"))
    trajectory = bool(visual_motion_summary.get("trajectory_produced"))
    reasons = []
    if scale != "METRIC_SCALE_ALIGNED":
        reasons.append("COLMAP/world coordinates remain arbitrary or unverified; metric map cells are not justified.")
    if tracking_only:
        reasons.append("Visual motion is tracking-only without measured calibration; no validated vehicle-frame trajectory.")
    if not trajectory:
        reasons.append("No metric trajectory was produced for map fusion.")
    reasons.append("2-D object detections were not written into the active navigator map.")

    return {
        "status": "DIAGNOSTIC_ONLY",
        "active_map_updated": False,
        "artifact": "diagnostic status only; not a metric terrain map",
        "reasons": reasons,
        "bridge": "Validated depth/segmentation/traversability observations may enter RuntimeAdapter when coordinate frames and calibration exist.",
    }
