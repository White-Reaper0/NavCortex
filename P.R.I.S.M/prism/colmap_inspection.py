"""Local, read-only inspection of COLMAP sparse binary models.

Exports are diagnostic artifacts only.  COLMAP coordinates have arbitrary
monocular scale and self-calibrated intrinsics are not a measured calibration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import struct
from typing import Iterable, Mapping

import numpy as np


class ColmapModelError(ValueError):
    """A native sparse model cannot be read or does not satisfy QA invariants."""


_CAMERA_MODEL_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}


@dataclass(frozen=True, slots=True)
class ColmapCamera:
    camera_id: int
    model_id: int
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ColmapImage:
    image_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str
    point3d_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ColmapPoint:
    point_id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]
    error_px: float
    track: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class SparseModel:
    cameras: tuple[ColmapCamera, ...]
    images: tuple[ColmapImage, ...]
    points: tuple[ColmapPoint, ...]


def _read_exact(handle, count: int) -> bytes:
    value = handle.read(count)
    if len(value) != count:
        raise ColmapModelError("truncated COLMAP binary model")
    return value


def _unpack(handle, pattern: str):
    return struct.unpack("<" + pattern, _read_exact(handle, struct.calcsize("<" + pattern)))


def _read_name(handle) -> str:
    chunks = bytearray()
    while True:
        value = _read_exact(handle, 1)
        if value == b"\x00":
            try:
                return chunks.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ColmapModelError("image name is not UTF-8") from error
        chunks.extend(value)


def read_sparse_model(model_dir: str | Path) -> SparseModel:
    """Read native ``cameras.bin``, ``images.bin``, and ``points3D.bin``."""
    model_dir = Path(model_dir)
    paths = {name: model_dir / name for name in ("cameras.bin", "images.bin", "points3D.bin")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ColmapModelError("missing native model file(s): " + ", ".join(missing))
    cameras: list[ColmapCamera] = []
    with paths["cameras.bin"].open("rb") as handle:
        for _ in range(_unpack(handle, "Q")[0]):
            camera_id, model_id, width, height = _unpack(handle, "iiQQ")
            if model_id not in _CAMERA_MODEL_PARAMS:
                raise ColmapModelError(f"unsupported COLMAP camera model id {model_id}")
            params = _unpack(handle, "d" * _CAMERA_MODEL_PARAMS[model_id])
            cameras.append(ColmapCamera(camera_id, model_id, width, height, params))
    images: list[ColmapImage] = []
    with paths["images.bin"].open("rb") as handle:
        for _ in range(_unpack(handle, "Q")[0]):
            image_id, *pose, camera_id = _unpack(handle, "i" + "d" * 7 + "i")
            name = _read_name(handle)
            count = _unpack(handle, "Q")[0]
            point_ids = []
            for _ in range(count):
                _, _, point_id = _unpack(handle, "ddq")
                point_ids.append(point_id)
            images.append(ColmapImage(image_id, tuple(pose[:4]), tuple(pose[4:]), camera_id,
                                      name, tuple(point_ids)))
    points: list[ColmapPoint] = []
    with paths["points3D.bin"].open("rb") as handle:
        for _ in range(_unpack(handle, "Q")[0]):
            point_id, x, y, z, red, green, blue, error = _unpack(handle, "QdddBBBd")
            track = tuple(_unpack(handle, "II") for _ in range(_unpack(handle, "Q")[0]))
            points.append(ColmapPoint(point_id, (x, y, z), (red, green, blue), error, track))
    return SparseModel(tuple(cameras), tuple(images), tuple(points))


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    values = np.asarray(tuple(values), dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"count": int(len(values)), "finite_count": 0, "min": None, "median": None,
                "mean": None, "p95": None, "max": None}
    return {"count": int(len(values)), "finite_count": int(len(finite)), "min": float(finite.min()),
            "median": float(np.median(finite)), "mean": float(finite.mean()),
            "p95": float(np.percentile(finite, 95)), "max": float(finite.max())}


def camera_center(image: ColmapImage) -> tuple[float, float, float]:
    """Return COLMAP world-space camera center ``-R(q)^T t`` (arbitrary units)."""
    q = np.asarray(image.qvec, dtype=float)
    norm = np.linalg.norm(q)
    if not math.isfinite(norm) or norm == 0:
        raise ColmapModelError(f"image {image.name} has invalid quaternion")
    q /= norm
    w, x, y, z = q
    rotation = np.array(((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
                         (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
                         (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y))))
    return tuple((-rotation.T @ np.asarray(image.tvec, dtype=float)).tolist())


def build_qa_report(model: SparseModel, frame_manifest: Mapping[str, object] | None = None) -> dict:
    """Create deterministic integrity/geometry QA without assigning metric scale."""
    camera_ids = {camera.camera_id for camera in model.cameras}
    names = set()
    errors: list[str] = []
    centers = []
    image_rows = []
    manifest = {item["file"]: item for item in frame_manifest.get("frames", [])} if frame_manifest else {}
    for image in sorted(model.images, key=lambda item: item.name):
        if image.name in names: errors.append(f"duplicate image name: {image.name}")
        names.add(image.name)
        if image.camera_id not in camera_ids: errors.append(f"{image.name}: missing camera {image.camera_id}")
        try:
            center = camera_center(image)
            if not np.isfinite(center).all(): raise ColmapModelError("non-finite camera center")
            centers.append((image, center))
        except ColmapModelError as error:
            errors.append(str(error)); center = (None, None, None)
        source = manifest.get(image.name, {})
        image_rows.append({"image_id": image.image_id, "file": image.name,
                           "source_frame_index": source.get("source_frame_index"),
                           "source_time_s": source.get("source_time_s"), "camera_id": image.camera_id,
                           "camera_center_arbitrary_units": center})
    invalid_points = []
    for point in model.points:
        if not np.isfinite(point.xyz).all() or not math.isfinite(point.error_px) or point.error_px < 0:
            invalid_points.append(point.point_id)
    tracks = [len(point.track) for point in model.points]
    reprojection = [point.error_px for point in model.points]
    ordered = sorted(centers, key=lambda item: (manifest.get(item[0].name, {}).get("source_frame_index", 10**12), item[0].name))
    step_rows = [{"from_file": ordered[index - 1][0].name, "to_file": ordered[index][0].name,
                  "distance_arbitrary_units": float(np.linalg.norm(np.subtract(ordered[index][1], ordered[index - 1][1])))}
                 for index in range(1, len(ordered))]
    steps = [row["distance_arbitrary_units"] for row in step_rows]
    source_indices = [manifest.get(image.name, {}).get("source_frame_index") for image, _ in ordered]
    source_gaps = [right - left for left, right in zip(source_indices, source_indices[1:])
                   if isinstance(left, int) and isinstance(right, int)]
    components = _image_components(model.images, model.points)
    xyz = np.asarray([point.xyz for point in model.points if np.isfinite(point.xyz).all()], dtype=float)
    extent = (xyz.max(axis=0) - xyz.min(axis=0)).tolist() if len(xyz) else None
    rank = int(np.linalg.matrix_rank(xyz - xyz.mean(axis=0))) if len(xyz) else 0
    q95 = _summary(reprojection)["p95"]
    high_error = sum(error > max(5.0, float(q95 or 0.0)) for error in reprojection if math.isfinite(error))
    step_median = float(np.median(steps)) if steps else 0.0
    step_mad = float(np.median(np.abs(np.asarray(steps) - step_median))) if steps else 0.0
    step_limit = step_median + 3 * step_mad if step_mad else None
    step_outliers = [row for row in step_rows if step_limit is not None and row["distance_arbitrary_units"] > step_limit]
    return {"format": "prism-colmap-sparse-qa/v1", "physical_scale": "unverified_arbitrary_colmap_units",
            "calibration": "COLMAP self-calibrated/refined intrinsics; not measured checkerboard calibration",
            "registered_images": len(model.images), "cameras": len(model.cameras), "points3D": len(model.points),
            "camera_records": [asdict(camera) for camera in model.cameras],
            "source_manifest_images": len(manifest), "image_rows": image_rows,
            "point_track_length": _summary(tracks), "reprojection_error_px": _summary(reprojection),
            "high_reprojection_error_points": high_error, "invalid_point_ids": invalid_points,
            "trajectory": {"camera_pose_count": len(centers), "step_distance_arbitrary_units": _summary(steps),
                           "source_frame_gaps": _summary(source_gaps), "image_graph_components": components,
                           "step_outlier_limit_arbitrary_units": step_limit, "step_outliers": step_outliers,
                           "continuity_warning": components > 1 or any(step == 0 for step in steps)},
            "geometry": {"point_extent_arbitrary_units": extent, "point_cloud_rank": rank,
                         "degenerate_warning": rank < 2 or not len(model.points)},
            "issues": errors}


def _image_components(images: Iterable[ColmapImage], points: Iterable[ColmapPoint]) -> int:
    ids = {image.image_id for image in images}; parent = {item: item for item in ids}
    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]; item = parent[item]
        return item
    def join(left, right):
        left, right = find(left), find(right)
        if left != right: parent[right] = left
    for point in points:
        observed = [image_id for image_id, _ in point.track if image_id in parent]
        for image_id in observed[1:]: join(observed[0], image_id)
    return len({find(item) for item in ids})


def write_ply(points: Iterable[ColmapPoint], path: str | Path) -> None:
    points = tuple(points)
    if any(not np.isfinite(point.xyz).all() for point in points):
        raise ColmapModelError("cannot export non-finite sparse points")
    lines = ["ply", "format ascii 1.0", f"element vertex {len(points)}", "property float x", "property float y",
             "property float z", "property uchar red", "property uchar green", "property uchar blue", "end_header"]
    lines.extend(f"{point.xyz[0]:.9g} {point.xyz[1]:.9g} {point.xyz[2]:.9g} "
                 f"{point.rgb[0]} {point.rgb[1]} {point.rgb[2]}" for point in points)
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def trajectory_rows(model: SparseModel, manifest: Mapping[str, object] | None = None) -> list[dict]:
    lookup = {item["file"]: item for item in manifest.get("frames", [])} if manifest else {}
    rows = []
    for image in model.images:
        source = lookup.get(image.name, {})
        rows.append({"image_id": image.image_id, "file": image.name,
                     "source_frame_index": source.get("source_frame_index"),
                     "source_time_s": source.get("source_time_s"), "qvec_wxyz": image.qvec,
                     "tvec_world_to_camera": image.tvec, "camera_center_arbitrary_units": camera_center(image)})
    return sorted(rows, key=lambda row: (row["source_frame_index"] is None, row["source_frame_index"], row["file"]))
