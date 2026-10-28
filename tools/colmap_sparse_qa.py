"""Read-only QA tooling for COLMAP sparse reconstruction models.

This module is intentionally independent of the P.R.I.S.M. mapper. It only
reads COLMAP native binary outputs (cameras.bin / images.bin / points3D.bin)
and writes PLY, CSV and JSON artefacts. It never runs COLMAP, never modifies
the mapper, and never infers metric scale from arbitrary reconstruction units.
"""

from __future__ import annotations

import csv
import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

COLMAP_MODELS = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}

# Number of intrinsic parameters per COLMAP camera model. cameras.bin does not
# store a parameter count; it is implied by the model id.
PARAM_COUNTS = {
    0: 3,   # SIMPLE_PINHOLE: f, cx, cy
    1: 4,   # PINHOLE: fx, fy, cx, cy
    2: 4,   # SIMPLE_RADIAL: f, cx, cy, k
    3: 4,   # RADIAL: f, cx, cy, k1
    4: 8,   # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
    5: 8,   # OPENCV_FISHEYE
    6: 12,  # FULL_OPENCV
    7: 5,   # FOV: fx, fy, cx, cy, omega
    8: 4,   # SIMPLE_RADIAL_FISHEYE
    9: 4,   # RADIAL_FISHEYE
    10: 12, # THIN_PRISM_FISHEYE
}


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float
    point3D_id: int


@dataclass(frozen=True)
class Image:
    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str
    points2D: tuple[Point2D, ...] = ()


@dataclass(frozen=True)
class Point3D:
    point3D_id: int
    xyz: tuple[float, float, float]
    error: float
    track: tuple[tuple[int, int], ...]  # (image_id, point2D_idx)
    color: tuple[int, int, int]


@dataclass(frozen=True)
class Model:
    cameras: dict[int, Camera]
    images: dict[int, Image]
    points: dict[int, Point3D]


@dataclass
class QAReport:
    source_model: str
    registered_images: int
    total_points: int
    total_observations: int
    mean_track_length: float
    mean_observations_per_image: float
    mean_reprojection_error_px: Optional[float]
    reprojection_error_percentiles_px: dict[str, float]
    camera_model: Optional[str]
    camera_params: Optional[tuple[float, ...]]
    image_size: Optional[tuple[int, int]]
    trajectory: list[dict] = field(default_factory=list)
    outliers: list[dict] = field(default_factory=list)
    connectivity: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_model": self.source_model,
            "registered_images": self.registered_images,
            "total_points": self.total_points,
            "total_observations": self.total_observations,
            "mean_track_length": self.mean_track_length,
            "mean_observations_per_image": self.mean_observations_per_image,
            "mean_reprojection_error_px": self.mean_reprojection_error_px,
            "reprojection_error_percentiles_px": self.reprojection_error_percentiles_px,
            "camera_model": self.camera_model,
            "camera_params": list(self.camera_params) if self.camera_params else None,
            "image_size": list(self.image_size) if self.image_size else None,
            "trajectory": self.trajectory,
            "outliers": self.outliers,
            "connectivity": self.connectivity,
            "notes": self.notes,
        }


def _read_uint64(handle) -> int:
    return struct.unpack("<Q", handle.read(8))[0]


def _read_uint32(handle) -> int:
    return struct.unpack("<I", handle.read(4))[0]


def _read_double(handle) -> float:
    return struct.unpack("<d", handle.read(8))[0]


def _read_cstring(handle) -> str:
    chars = []
    while True:
        byte = handle.read(1)
        if not byte or byte == b"\x00":
            break
        chars.append(byte)
    return b"".join(chars).decode("utf-8", errors="replace")


def load_colmap_model(model_dir: str | Path) -> Model:
    """Parse COLMAP native sparse model binary files.

    Field sizes observed empirically from COLMAP 4.2.1 output:
    * cameras.bin: count u64; per camera id u32, model u32, width u32,
      height u32, num_params u64, params double[].
    * images.bin: count u64; per image id u32, qvec 4x double, tvec 3x double,
      camera_id u32, name NUL-terminated, num_points2D u64, then per point2D
      x double, y double, point3D_id u32 (0xFFFFFFFF = unmatched).
    * points3D.bin: count u64; per point id u64, xyz 3x double, error double,
      track_length u64, then per track image_id u32, point2D_idx u32, and
      3 uchar RGB bytes.
    All integers are little-endian.
    """
    model_dir = Path(model_dir)
    cameras_path = model_dir / "cameras.bin"
    images_path = model_dir / "images.bin"
    points_path = model_dir / "points3D.bin"

    if not cameras_path.is_file():
        raise FileNotFoundError(f"cameras.bin not found in {model_dir}")
    if not images_path.is_file():
        raise FileNotFoundError(f"images.bin not found in {model_dir}")
    if not points_path.is_file():
        raise FileNotFoundError(f"points3D.bin not found in {model_dir}")

    cameras: dict[int, Camera] = {}
    with cameras_path.open("rb") as handle:
        start = handle.tell()
        num_cameras = _read_uint64(handle)
        for _ in range(num_cameras):
            camera_id = _read_uint32(handle)
            model_id = _read_uint32(handle)
            width = _read_uint64(handle)
            height = _read_uint64(handle)
            num_params = PARAM_COUNTS.get(int(model_id))
            if num_params is None:
                raise ValueError(
                    f"unknown COLMAP camera model id {model_id}; cannot infer "
                    "parameter count from cameras.bin"
                )
            params = tuple(_read_double(handle) for _ in range(num_params))
            cameras[int(camera_id)] = Camera(
                camera_id=int(camera_id),
                model=COLMAP_MODELS.get(int(model_id), f"model_{int(model_id)}"),
                width=int(width),
                height=int(height),
                params=params,
            )
        _expect_end(handle, cameras_path, start)

    images: dict[int, Image] = {}
    with images_path.open("rb") as handle:
        start = handle.tell()
        num_images = _read_uint64(handle)
        for _ in range(num_images):
            image_id = _read_uint32(handle)
            qw = _read_double(handle)
            qx = _read_double(handle)
            qy = _read_double(handle)
            qz = _read_double(handle)
            tx = _read_double(handle)
            ty = _read_double(handle)
            tz = _read_double(handle)
            camera_id = _read_uint32(handle)
            name = _read_cstring(handle)
            num_points2d = _read_uint64(handle)
            points2D = []
            for _ in range(num_points2d):
                x = _read_double(handle)
                y = _read_double(handle)
                point3D_id_raw = _read_uint64(handle)
                point3D_id = (
                    -1
                    if point3D_id_raw == 0xFFFFFFFFFFFFFFFF
                    else int(point3D_id_raw)
                )
                points2D.append(
                    Point2D(x=x, y=y, point3D_id=point3D_id)
                )
            images[int(image_id)] = Image(
                image_id=int(image_id),
                qw=qw,
                qx=qx,
                qy=qy,
                qz=qz,
                tx=tx,
                ty=ty,
                tz=tz,
                camera_id=int(camera_id),
                name=name,
                points2D=tuple(points2D),
            )
        _expect_end(handle, images_path, start)

    points: dict[int, Point3D] = {}
    with points_path.open("rb") as handle:
        start = handle.tell()
        num_points = _read_uint64(handle)
        for _ in range(num_points):
            point3D_id = _read_uint64(handle)
            x = _read_double(handle)
            y = _read_double(handle)
            z = _read_double(handle)
            error = _read_double(handle)
            track_length = _read_uint64(handle)
            track = []
            for _ in range(track_length):
                image_id = _read_uint32(handle)
                point2D_idx = _read_uint32(handle)
                track.append((int(image_id), int(point2D_idx)))
            red = handle.read(1)[0]
            green = handle.read(1)[0]
            blue = handle.read(1)[0]
            points[int(point3D_id)] = Point3D(
                point3D_id=int(point3D_id),
                xyz=(x, y, z),
                error=error,
                track=tuple(track),
                color=(red, green, blue),
            )
        _expect_end(handle, points_path, start)

    return Model(cameras=cameras, images=images, points=points)


def _expect_end(handle, path: Path, start: int) -> None:
    end = handle.tell()
    actual = end - start
    handle.seek(0, 2)
    file_size = handle.tell()
    if actual != file_size:
        raise ValueError(
            f"{path.name}: consumed {actual} bytes but file size is {file_size}; "
            "binary layout mismatch"
        )


def _quaternion_to_rotation(qw, qx, qy, qz) -> np.ndarray:
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0.0:
        return np.eye(3)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qw * qz),
                2 * (qx * qz + qw * qy),
            ],
            [
                2 * (qx * qy + qw * qz),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qw * qx),
            ],
            [
                2 * (qx * qz - qw * qy),
                2 * (qy * qz + qw * qx),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def _camera_center(image: Image) -> tuple[float, float, float]:
    R = _quaternion_to_rotation(image.qw, image.qx, image.qy, image.qz)
    t = np.array([image.tx, image.ty, image.tz], dtype=np.float64)
    center = -R.T @ t
    return float(center[0]), float(center[1]), float(center[2])


def export_trajectory_csv(
    model: Model,
    output: str | Path,
    frame_lookup: Optional[dict[str, dict]] = None,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image_id",
                "file",
                "source_frame_index",
                "source_time_s",
                "qw",
                "qx",
                "qy",
                "qz",
                "tx",
                "ty",
                "tz",
                "camera_center_x",
                "camera_center_y",
                "camera_center_z",
                "camera_id",
                "num_points2D",
            ]
        )
        for image in sorted(model.images.values(), key=lambda im: im.image_id):
            cx, cy, cz = _camera_center(image)
            meta = {}
            if frame_lookup:
                meta = frame_lookup.get(image.name, {})
            writer.writerow(
                [
                    image.image_id,
                    image.name,
                    meta.get("source_frame_index", ""),
                    meta.get("source_time_s", ""),
                    f"{image.qw:.10f}",
                    f"{image.qx:.10f}",
                    f"{image.qy:.10f}",
                    f"{image.qz:.10f}",
                    f"{image.tx:.10f}",
                    f"{image.ty:.10f}",
                    f"{image.tz:.10f}",
                    f"{cx:.10f}",
                    f"{cy:.10f}",
                    f"{cz:.10f}",
                    image.camera_id,
                    len(image.points2D),
                ]
            )
    return output


def export_point_cloud_ply(
    model: Model,
    output: str | Path,
    include_invalid: bool = False,
) -> Path:
    """Write an ASCII PLY point cloud.

    PLY vertex elements: x,y,z (float), red,green,blue (uchar),
    reprojection_error_px (float), track_length (uint).
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    vertices = []
    for point in model.points.values():
        x, y, z = point.xyz
        if not include_invalid and not np.all(
            np.isfinite([x, y, z, point.error])
        ):
            continue
        vertices.append((x, y, z, *point.color, float(point.error), len(point.track)))

    with output.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("property float reprojection_error_px\n")
        handle.write("property uint track_length\n")
        handle.write("end_header\n")
        for x, y, z, r, g, b, err, track_len in vertices:
            handle.write(
                f"{x:.10f} {y:.10f} {z:.10f} {r} {g} {b} {err:.10f} {track_len}\n"
            )
    return output


def _percentiles(values: list[float], probs: tuple[float, ...]) -> list[float]:
    if not values:
        return [float("nan")] * len(probs)
    arr = np.array(values, dtype=np.float64)
    return [float(np.percentile(arr, p)) for p in probs]


def build_qa_report(
    model: Model,
    source_model: str,
    frame_lookup: Optional[dict[str, dict]] = None,
) -> QAReport:
    images = sorted(model.images.values(), key=lambda im: im.image_id)
    points = list(model.points.values())
    cameras = list(model.cameras.values())

    track_lengths = [len(pt.track) for pt in points]
    # A sparse 3D observation is a verified image-point association in a
    # triangulated point track.  ``Image.points2D`` includes unmatched feature
    # keypoints and must not inflate sparse-model observation statistics.
    observations = sum(track_lengths)
    errors = [pt.error for pt in points if np.isfinite(pt.error)]

    p25, p50, p75, p95, p99 = _percentiles(errors, (25, 50, 75, 95, 99))
    mean_error = float(np.mean(errors)) if errors else None

    # Trajectory rows + continuity checks.
    trajectory = []
    centers = []
    for image in images:
        cx, cy, cz = _camera_center(image)
        centers.append((cx, cy, cz))
        meta = frame_lookup.get(image.name, {}) if frame_lookup else {}
        trajectory.append(
            {
                "image_id": image.image_id,
                "file": image.name,
                "source_frame_index": meta.get("source_frame_index"),
                "source_time_s": meta.get("source_time_s"),
                "camera_center_x": cx,
                "camera_center_y": cy,
                "camera_center_z": cz,
                "tx": image.tx,
                "ty": image.ty,
                "tz": image.tz,
                "qw": image.qw,
                "qx": image.qx,
                "qy": image.qy,
                "qz": image.qz,
                "camera_id": image.camera_id,
                "num_points2D": len(image.points2D),
            }
        )

    connectivity = _connectivity(model)
    outliers = _detect_outliers(model, errors)

    notes: list[str] = []
    if not cameras:
        notes.append("No cameras found in model.")
    else:
        cam = cameras[0]
        notes.append(
            f"Camera model {cam.model} ({cam.width}x{cam.height}) is "
            "COLMAP self-calibrated; not a measured checkerboard calibration."
        )
    if model.points and not any(np.isfinite(pt.error) for pt in points):
        notes.append("No finite reprojection errors available.")
    if observations == 0:
        notes.append("No observations; model is empty/degenerate.")
    if connectivity.get("largest_component_images", 0) < len(images):
        notes.append(
            "Visibility graph is disconnected; reconstruction may contain "
            "independent sub-models."
        )

    return QAReport(
        source_model=source_model,
        registered_images=len(images),
        total_points=len(points),
        total_observations=observations,
        mean_track_length=(
            float(np.mean(track_lengths)) if track_lengths else 0.0
        ),
        mean_observations_per_image=(
            observations / len(images) if images else 0.0
        ),
        mean_reprojection_error_px=mean_error,
        reprojection_error_percentiles_px={
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p95": p95,
            "p99": p99,
        },
        camera_model=cameras[0].model if cameras else None,
        camera_params=cameras[0].params if cameras else None,
        image_size=(cameras[0].width, cameras[0].height) if cameras else None,
        trajectory=trajectory,
        outliers=outliers,
        connectivity=connectivity,
        notes=notes,
    )


def _connectivity(model: Model) -> dict:
    images = list(model.images.values())
    if not images:
        return {
            "image_count": 0,
            "largest_component_images": 0,
            "connected_components": 0,
            "fully_connected": True,
        }

    adj = {im.image_id: set() for im in images}
    point_to_images: dict[int, set[int]] = {}
    for point in model.points.values():
        for image_id, _ in point.track:
            point_to_images.setdefault(point.point3D_id, set()).add(image_id)
    for image_ids in point_to_images.values():
        id_list = sorted(image_ids)
        for i in range(len(id_list)):
            for j in range(i + 1, len(id_list)):
                adj[id_list[i]].add(id_list[j])
                adj[id_list[j]].add(id_list[i])

    seen = set()
    components = []
    for image in images:
        if image.image_id in seen:
            continue
        stack = [image.image_id]
        comp = []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            stack.extend(adj[cur] - seen)
        components.append(comp)
    components.sort(key=len, reverse=True)
    largest = len(components[0]) if components else 0
    return {
        "image_count": len(images),
        "largest_component_images": largest,
        "connected_components": len(components),
        "component_sizes": [len(c) for c in components],
        "fully_connected": largest == len(images),
    }


def _detect_outliers(model: Model, errors: list[float]) -> list[dict]:
    outliers = []
    points = list(model.points.values())
    if not points:
        return outliers

    median_error = float(np.median(errors)) if errors else 0.0
    error_p95 = float(np.percentile(errors, 95)) if errors else 0.0
    error_threshold = max(5.0, 3.0 * median_error, error_p95 + 1.0)

    centers = np.array(
        [_camera_center(im) for im in model.images.values()], dtype=np.float64
    )
    if len(centers):
        centroid = centers.mean(axis=0)
        scale = float(np.max(np.linalg.norm(centers - centroid, axis=1))) or 1.0
    else:
        centroid = np.zeros(3)
        scale = 1.0

    for point in points:
        x, y, z = point.xyz
        flags = []
        if not np.all(np.isfinite([x, y, z])):
            flags.append("non_finite_position")
        if point.error > error_threshold:
            flags.append("high_reprojection_error")
        if len(point.track) <= 1:
            flags.append("singleton_track")
        dist = float(np.linalg.norm(np.array([x, y, z]) - centroid))
        if dist > 10.0 * scale:
            flags.append("distant_from_camera_centroid")
        if flags:
            outliers.append(
                {
                    "point3D_id": point.point3D_id,
                    "xyz": [x, y, z],
                    "error": point.error,
                    "track_length": len(point.track),
                    "flags": flags,
                }
            )
    outliers.sort(key=lambda o: (-o["error"], o["point3D_id"]))
    return outliers[:50]


def write_report(report: QAReport, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return output


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="QA inspection and export for a COLMAP sparse model.")
    parser.add_argument("--model", required=True,
                        help="COLMAP sparse model directory (sparse/0)")
    parser.add_argument("--manifest", default=None,
                        help="frames_manifest.json for source frame identities")
    parser.add_argument("--output", required=True,
                        help="output directory for PLY, CSV and JSON report")
    parser.add_argument("--include-invalid-points", action="store_true",
                        help="include non-finite points in PLY export")
    args = parser.parse_args(argv)

    model = load_colmap_model(args.model)

    frame_lookup: Optional[dict[str, dict]] = None
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        frame_lookup = {
            fr["file"]: fr for fr in manifest.get("frames", [])
        }

    report = build_qa_report(model, str(args.model), frame_lookup=frame_lookup)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_point_cloud_ply(model, out_dir / "points.ply",
                           include_invalid=args.include_invalid_points)
    export_trajectory_csv(model, out_dir / "camera_trajectory.csv",
                          frame_lookup=frame_lookup)
    write_report(report, out_dir / "qa_report.json")

    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
