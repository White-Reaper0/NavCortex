"""Read-only adapters around the existing inspected COLMAP sparse model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from ..colmap_inspection import SparseModel, build_qa_report, read_sparse_model, trajectory_rows


def sparse_geometry_report(model_dir: str | Path, manifest_path: str | Path,
                           requested_source_frames: Iterable[int],
                           *, export_open3d_ply: Path | None = None) -> dict:
    """Return QA and only exact manifest-to-COLMAP source-frame pose matches."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    model = read_sparse_model(model_dir)
    rows = trajectory_rows(model, manifest)
    by_source = {row["source_frame_index"]: row for row in rows
                 if isinstance(row["source_frame_index"], int)}
    requested = tuple(sorted(set(int(value) for value in requested_source_frames)))
    exact = [by_source[index] for index in requested if index in by_source]
    missing = [index for index in requested if index not in by_source]
    open3d = open3d_sparse_inspection(model, export_ply_path=export_open3d_ply)
    return {
        "status": "RELATIVE_ONLY",
        "coordinate_frame": "COLMAP reconstruction world; arbitrary units",
        "physical_scale": "METRIC_SCALE_UNVERIFIED",
        "calibration": "COLMAP self-calibrated intrinsics only; no measured checkerboard artifact",
        "qa": build_qa_report(model, manifest),
        "requested_source_frame_indices": list(requested),
        "exact_pose_matches": exact,
        "unmatched_source_frame_indices": missing,
        "association_policy": "exact source-frame identity only; nearest-frame association is prohibited",
        "open3d": open3d,
    }


def open3d_sparse_inspection(model: SparseModel, *, export_ply_path: Path | None = None) -> dict:
    """Load native COLMAP XYZ/RGB into Open3D without rescaling or unit conversion."""
    try:
        import open3d as o3d  # type: ignore
    except Exception as error:
        return {
            "status": "UNAVAILABLE",
            "reason": f"Open3D import failed: {type(error).__name__}: {error}",
            "visualization": "use inspection_v2/sparse_points.ply in a local PLY viewer",
        }

    if not model.points:
        return {
            "status": "AVAILABLE",
            "version": getattr(o3d, "__version__", "unknown"),
            "point_count": 0,
            "coordinate_frame": "COLMAP reconstruction world; arbitrary units; not rescaled",
            "visualization": "empty sparse model",
        }

    xyz = np.asarray([item.xyz for item in model.points], dtype=np.float64)
    rgb = np.asarray([item.rgb for item in model.points], dtype=np.float64) / 255.0
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.colors = o3d.utility.Vector3dVector(rgb)
    bounds_min = xyz.min(axis=0).tolist()
    bounds_max = xyz.max(axis=0).tolist()
    exported = None
    if export_ply_path is not None:
        export_ply_path.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_point_cloud(str(export_ply_path), cloud, write_ascii=False):
            raise OSError(f"Open3D failed to write point cloud: {export_ply_path}")
        exported = str(export_ply_path)

    return {
        "status": "LOADED",
        "version": getattr(o3d, "__version__", "unknown"),
        "point_count": int(len(model.points)),
        "coordinate_frame": "COLMAP reconstruction world; arbitrary units; not rescaled",
        "axis_aligned_bounds_arbitrary_units": {"min": bounds_min, "max": bounds_max},
        "mean_reprojection_error_px": float(np.mean([item.error_px for item in model.points])),
        "exported_ply": exported,
        "visualization": "optional Open3D viewer or exported PLY; CPU-only; no CUDA required",
    }
