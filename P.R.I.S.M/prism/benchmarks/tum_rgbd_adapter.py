"""TUM RGB-D benchmark adapter for offline, reference-input validation.

The adapter is deliberately separate from the ``clips/clip_01.mp4`` workflow.
It reads the official sequence index files, performs one-to-one timestamp
association, exposes the published calibration, and provides bounded depth and
ground-truth helpers.  Dataset files are never copied into source control.

Official sources
----------------
* dataset home: https://vision.in.tum.de/data/datasets/rgbd-dataset
* download page: https://vision.in.tum.de/data/datasets/rgbd-dataset/download
* file formats: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats

TUM does not publish a checksum for the archive.  The local SHA-256 is recorded
for reproducibility, and the downloaded byte count is compared with the
official ``Content-Length`` observed for this archive.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import tarfile
from pathlib import Path
from typing import Optional, Sequence
from urllib.request import urlopen

import cv2
import numpy as np

from prism.config import PROC_H, PROC_W
from prism.visual_odometry_replay import CameraCalibration


DATASET_PAGE = "https://vision.in.tum.de/data/datasets/rgbd-dataset"
DOWNLOAD_PAGE = "https://vision.in.tum.de/data/datasets/rgbd-dataset/download"
FILE_FORMATS_PAGE = "https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats"

FREIBURG1_XYZ_SEQUENCE = "rgbd_dataset_freiburg1_xyz"
FREIBURG1_XYZ_ARCHIVE = f"{FREIBURG1_XYZ_SEQUENCE}.tgz"
FREIBURG1_XYZ_URLS = (
    "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz",
    "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz",
)

FREIBURG1_RPY_SEQUENCE = "rgbd_dataset_freiburg1_rpy"
FREIBURG1_RPY_ARCHIVE = f"{FREIBURG1_RPY_SEQUENCE}.tgz"
FREIBURG1_RPY_URLS = (
    "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_rpy.tgz",
    "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_rpy.tgz",
)

FREIBURG1_360_SEQUENCE = "rgbd_dataset_freiburg1_360"
FREIBURG1_360_ARCHIVE = f"{FREIBURG1_360_SEQUENCE}.tgz"
FREIBURG1_360_URLS = (
    "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_360.tgz",
    "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_360.tgz",
)

#: Official download page states "approx. 0.42GB"; the served archive currently
#: reports this exact Content-Length (448,204,271 bytes).
OFFICIAL_ARCHIVE_BYTES = 448_204_271
OFFICIAL_ARCHIVE_SIZE_TEXT = "approx. 0.42 GB (official download page)"
OFFICIAL_TRAJECTORY_LENGTH_TEXT = "7.112 m (official download page, full sequence)"
OFFICIAL_CHECKSUM_TEXT = (
    "TUM publishes no archive checksum on the dataset or download pages; "
    "the local SHA-256 is recorded for reproducibility only."
)

#: Freiburg 1 RPY sequence (rotational motion)
FREIBURG1_RPY_ARCHIVE_BYTES = 410_268_381
FREIBURG1_RPY_SIZE_TEXT = "approx. 0.42 GB (official download page)"

#: Freiburg 1 360 sequence (360-degree rotation)
FREIBURG1_360_ARCHIVE_BYTES = 417_954_230
FREIBURG1_360_SIZE_TEXT = "approx. 0.45 GB (official download page)"

DATASET_CITATION = (
    "J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, "
    "\"A Benchmark for the Evaluation of RGB-D SLAM Systems,\" "
    "Proc. of the International Conference on Intelligent Robot Systems (IROS), "
    "Oct. 2012."
)
DATASET_LICENSE = (
    "Dataset data: Creative Commons 4.0 Attribution License (CC BY 4.0). "
    "Accompanying source code: BSD-2-Clause License. Dataset files are not "
    "redistributed with P.R.I.S.M.; attribution to the TUM RGB-D benchmark and "
    "the cited publication is required."
)

DEPTH_SCALE = 5000.0  # 16-bit PNG depth units -> metres (official file formats page)

# ROS default intrinsics, recommended by TUM for the pre-registered depth images
# because undistorting already-registered depth is not straightforward.
FREIBURG1_ROS_DEFAULT_INTRINSICS = {
    "fx": 525.0,
    "fy": 525.0,
    "cx": 319.5,
    "cy": 239.5,
    "distortion": (0.0, 0.0, 0.0, 0.0, 0.0),
    "provenance": "TUM RGB-D file formats page (ROS default; recommended for registered depth)",
}

# Freiburg 1 RGB calibration at the native 640x480 resolution.
FREIBURG1_RGB_INTRINSICS = {
    "fx": 517.3,
    "fy": 516.5,
    "cx": 318.6,
    "cy": 255.3,
    "distortion": (0.2624, -0.9531, -0.0054, 0.0026, 1.1633),
    "provenance": "TUM RGB-D file formats page (Freiburg 1 RGB calibration)",
}

NATIVE_WIDTH = 640
NATIVE_HEIGHT = 480


@dataclass(frozen=True, slots=True)
class TimestampedPath:
    timestamp_s: float
    relative_path: str


@dataclass(frozen=True, slots=True)
class TumPose:
    """TUM ground-truth camera-to-world pose (translation + quaternion xyzw)."""

    timestamp_s: float
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class TumSequenceLayout:
    sequence_root: Path
    sequence_name: str
    rgb_entries: tuple[TimestampedPath, ...]
    depth_entries: tuple[TimestampedPath, ...]
    groundtruth: tuple[TumPose, ...]
    rgb_depth_associations: tuple[tuple[TimestampedPath, TimestampedPath], ...]


@dataclass(frozen=True, slots=True)
class TumDatasetProvenance:
    archive_name: str
    source_urls: tuple[str, ...]
    dataset_page: str
    download_page: str
    file_formats_page: str
    citation: str
    license_note: str
    archive_path: Optional[str]
    archive_bytes: Optional[int]
    archive_sha256: Optional[str]
    extracted_sequence_root: Optional[str]
    download_status: str
    download_error: Optional[str]
    official_archive_bytes: Optional[int] = OFFICIAL_ARCHIVE_BYTES
    official_archive_size_text: str = OFFICIAL_ARCHIVE_SIZE_TEXT
    archive_size_matches_official: Optional[bool] = None
    checksum_status: str = OFFICIAL_CHECKSUM_TEXT
    official_trajectory_length_text: str = OFFICIAL_TRAJECTORY_LENGTH_TEXT


_LINE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+(\S+)\s*$")


def _parse_index_file(path: Path) -> tuple[TimestampedPath, ...]:
    entries: list[TimestampedPath] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _LINE_RE.match(line)
        if not match:
            continue
        entries.append(TimestampedPath(float(match.group(1)), match.group(2)))
    return tuple(entries)


def _parse_groundtruth(path: Path) -> tuple[TumPose, ...]:
    poses: list[TumPose] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        pose = TumPose(
            timestamp_s=float(parts[0]),
            position=(float(parts[1]), float(parts[2]), float(parts[3])),
            quaternion_xyzw=(float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])),
        )
        poses.append(pose)
    return tuple(poses)


def associate_timestamp_lists(
    first: Sequence[TimestampedPath],
    second: Sequence[TimestampedPath],
    *,
    max_difference_s: float = 0.02,
) -> tuple[tuple[TimestampedPath, TimestampedPath], ...]:
    """One-to-one timestamp association with TUM ``associate.py`` semantics.

    Every candidate pair within ``max_difference_s`` is ranked by absolute time
    difference, then greedily accepted if neither entry was already used.  This
    avoids the bug where one depth frame is matched to several RGB frames.
    """
    if max_difference_s < 0.0:
        raise ValueError("max_difference_s must be non-negative")
    if not first or not second:
        return ()

    order = sorted(range(len(second)), key=lambda j: second[j].timestamp_s)
    second_times = np.asarray([second[j].timestamp_s for j in order], dtype=np.float64)
    candidates: list[tuple[float, int, int]] = []
    for i, item in enumerate(first):
        lo = int(np.searchsorted(second_times, item.timestamp_s - max_difference_s, "left"))
        hi = int(np.searchsorted(second_times, item.timestamp_s + max_difference_s, "right"))
        for position in range(lo, hi):
            j = order[position]
            candidates.append((abs(second[j].timestamp_s - item.timestamp_s), i, j))
    candidates.sort()

    used_first: set[int] = set()
    used_second: set[int] = set()
    accepted: dict[int, tuple[TimestampedPath, TimestampedPath]] = {}
    for _difference, i, j in candidates:
        if i in used_first or j in used_second:
            continue
        used_first.add(i)
        used_second.add(j)
        accepted[i] = (first[i], second[j])
    return tuple(accepted[i] for i in sorted(accepted))


def intrinsics_matrix(spec: dict) -> np.ndarray:
    return np.array(
        [[spec["fx"], 0.0, spec["cx"]], [0.0, spec["fy"], spec["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def freiburg1_rgb_intrinsics_at_size(
    width: int = PROC_W, height: int = PROC_H
) -> np.ndarray:
    """Freiburg 1 RGB intrinsics scaled to an arbitrary image size.

    The replay harness resizes every input frame to 640x360.  Scaling the native
    640x480 RGB calibration anisotropically to that size matches the actual image
    resize; the distortion coefficients live in normalised coordinates and remain
    unchanged.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    scale_x = width / float(NATIVE_WIDTH)
    scale_y = height / float(NATIVE_HEIGHT)
    return np.array(
        [
            [FREIBURG1_RGB_INTRINSICS["fx"] * scale_x, 0.0,
             FREIBURG1_RGB_INTRINSICS["cx"] * scale_x],
            [0.0, FREIBURG1_RGB_INTRINSICS["fy"] * scale_y,
             FREIBURG1_RGB_INTRINSICS["cy"] * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def freiburg1_calibration(*, ros_default: bool = True) -> CameraCalibration:
    """Native-resolution Freiburg 1 calibration (reference metadata)."""
    spec = FREIBURG1_ROS_DEFAULT_INTRINSICS if ros_default else FREIBURG1_RGB_INTRINSICS
    return CameraCalibration.from_intrinsics(
        intrinsics_matrix(spec),
        spec["provenance"],
        distortion=np.asarray(spec["distortion"], dtype=np.float64),
        image_size=(NATIVE_WIDTH, NATIVE_HEIGHT),
        calibration_id=("tum-rgbd-freiburg1-ros-default" if ros_default
                        else "tum-rgbd-freiburg1-rgb"),
    )


def freiburg1_rgb_processing_calibration(
    width: int = PROC_W, height: int = PROC_H
) -> CameraCalibration:
    """RGB calibration at the replay harness's processing resolution."""
    if (width, height) == (NATIVE_WIDTH, NATIVE_HEIGHT):
        return freiburg1_calibration(ros_default=False)
    provenance = (
        f"{FREIBURG1_RGB_INTRINSICS['provenance']}; intrinsics scaled "
        f"anisotropically from {NATIVE_WIDTH}x{NATIVE_HEIGHT} to {width}x{height} "
        "to match the harness resize, distortion unchanged"
    )
    return CameraCalibration.from_intrinsics(
        freiburg1_rgb_intrinsics_at_size(width, height),
        provenance,
        distortion=np.asarray(FREIBURG1_RGB_INTRINSICS["distortion"], dtype=np.float64),
        image_size=(width, height),
        calibration_id=f"tum-rgbd-freiburg1-rgb-{width}x{height}",
    )


def discover_sequence_root(base_dir: Path, archive_stem: str) -> Optional[Path]:
    base_dir = Path(base_dir)
    direct = base_dir / archive_stem
    if _is_sequence_root(direct):
        return direct
    for candidate in base_dir.rglob("rgb.txt"):
        parent = candidate.parent
        if _is_sequence_root(parent) and archive_stem in parent.name:
            return parent
    return None


def _is_sequence_root(path: Path) -> bool:
    return all((path / name).is_file()
               for name in ("rgb.txt", "depth.txt", "groundtruth.txt"))


def load_sequence_layout(sequence_root: Path) -> TumSequenceLayout:
    sequence_root = Path(sequence_root)
    rgb = _parse_index_file(sequence_root / "rgb.txt")
    depth = _parse_index_file(sequence_root / "depth.txt")
    groundtruth = _parse_groundtruth(sequence_root / "groundtruth.txt")
    return TumSequenceLayout(
        sequence_root=sequence_root,
        sequence_name=sequence_root.name,
        rgb_entries=rgb,
        depth_entries=depth,
        groundtruth=groundtruth,
        rgb_depth_associations=associate_timestamp_lists(rgb, depth),
    )


def sequence_integrity(
    layout: TumSequenceLayout, *, path_sample_size: int = 10
) -> dict:
    """Bounded integrity summary without reading every image."""
    rgb_times = np.asarray([entry.timestamp_s for entry in layout.rgb_entries], np.float64)
    depth_times = np.asarray([entry.timestamp_s for entry in layout.depth_entries], np.float64)
    gt_times = np.asarray([pose.timestamp_s for pose in layout.groundtruth], np.float64)
    intervals = np.diff(np.sort(rgb_times)) if rgb_times.size > 1 else np.empty(0)
    median_interval = float(np.median(intervals)) if intervals.size else float("nan")

    unique_depth = len({entry.relative_path for _, entry in layout.rgb_depth_associations})
    sample: list[TimestampedPath] = []
    combined = list(layout.rgb_entries) + list(layout.depth_entries)
    if combined:
        count = min(max(1, path_sample_size), len(combined))
        positions = np.linspace(0, len(combined) - 1, count).round().astype(int)
        sample = [combined[int(i)] for i in positions]
    missing = [entry.relative_path for entry in sample
               if not (layout.sequence_root / entry.relative_path).is_file()]

    return {
        "sequence_root": str(layout.sequence_root),
        "rgb_entries": len(layout.rgb_entries),
        "depth_entries": len(layout.depth_entries),
        "groundtruth_poses": len(layout.groundtruth),
        "rgb_depth_associations": len(layout.rgb_depth_associations),
        "unique_depth_entries_used": unique_depth,
        "reused_depth_entries": len(layout.rgb_depth_associations) - unique_depth,
        "rgb_timestamps_unique": bool(rgb_times.size == np.unique(rgb_times).size),
        "depth_timestamps_unique": bool(depth_times.size == np.unique(depth_times).size),
        "rgb_span_s": float(rgb_times[-1] - rgb_times[0]) if rgb_times.size else None,
        "groundtruth_span_s": float(gt_times[-1] - gt_times[0]) if gt_times.size else None,
        "median_rgb_interval_s": median_interval if np.isfinite(median_interval) else None,
        "estimated_rgb_fps": (1.0 / median_interval) if median_interval > 0 else None,
        "sampled_paths_checked": len(sample),
        "sampled_paths_missing": missing,
        "association_semantics": "one-to-one greedy nearest within 0.02 s (TUM associate.py)",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(
    *,
    archive_path: Optional[Path],
    extracted_root: Optional[Path],
    download_status: str,
    download_error: Optional[str],
    archive_sha256: Optional[str] = None,
    archive_name: str = FREIBURG1_XYZ_ARCHIVE,
    source_urls: Sequence[str] = FREIBURG1_XYZ_URLS,
    official_bytes: Optional[int] = OFFICIAL_ARCHIVE_BYTES,
) -> TumDatasetProvenance:
    archive_bytes = archive_path.stat().st_size if archive_path is not None else None
    size_matches = (archive_bytes == official_bytes
                    if archive_bytes is not None and official_bytes is not None else None)
    return TumDatasetProvenance(
        archive_name=archive_name,
        source_urls=tuple(source_urls),
        dataset_page=DATASET_PAGE,
        download_page=DOWNLOAD_PAGE,
        file_formats_page=FILE_FORMATS_PAGE,
        citation=DATASET_CITATION,
        license_note=DATASET_LICENSE,
        archive_path=str(archive_path) if archive_path is not None else None,
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
        extracted_sequence_root=str(extracted_root) if extracted_root is not None else None,
        download_status=download_status,
        download_error=download_error,
        official_archive_bytes=official_bytes,
        archive_size_matches_official=size_matches,
    )


def download_archive(
    dest_dir: Path,
    *,
    archive_name: str = FREIBURG1_XYZ_ARCHIVE,
    urls: Sequence[str] = FREIBURG1_XYZ_URLS,
    expected_bytes: Optional[int] = OFFICIAL_ARCHIVE_BYTES,
    timeout_s: float = 300.0,
) -> TumDatasetProvenance:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / archive_name

    if archive_path.is_file() and archive_path.stat().st_size > 0:
        size = archive_path.stat().st_size
        sha = sha256_file(archive_path)
        if expected_bytes is not None and size != expected_bytes:
            return _provenance(
                archive_path=archive_path, extracted_root=None,
                download_status="present_size_mismatch",
                download_error=(
                    f"local archive is {size} bytes but the official archive is "
                    f"{expected_bytes} bytes; refusing to treat it as verified"),
                archive_sha256=sha, archive_name=archive_name, source_urls=urls,
                official_bytes=expected_bytes,
            )
        return _provenance(
            archive_path=archive_path, extracted_root=None,
            download_status="already_present", download_error=None,
            archive_sha256=sha, archive_name=archive_name, source_urls=urls,
            official_bytes=expected_bytes,
        )

    last_error: Optional[str] = None
    for url in urls:
        part_path = archive_path.with_name(archive_name + ".part")
        part_path.unlink(missing_ok=True)
        try:
            digest = hashlib.sha256()
            total = 0
            with urlopen(url, timeout=timeout_s) as response:
                with part_path.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        digest.update(chunk)
                        total += len(chunk)
            if expected_bytes is not None and total != expected_bytes:
                part_path.unlink(missing_ok=True)
                last_error = (
                    f"{url}: downloaded {total} bytes, expected {expected_bytes}")
                continue
            part_path.replace(archive_path)
            return _provenance(
                archive_path=archive_path, extracted_root=None,
                download_status="downloaded", download_error=None,
                archive_sha256=digest.hexdigest(), archive_name=archive_name,
                source_urls=urls, official_bytes=expected_bytes,
            )
        except Exception as error:  # pragma: no cover - network dependent
            last_error = f"{url}: {type(error).__name__}: {error}"
            part_path.unlink(missing_ok=True)
    return _provenance(
        archive_path=None, extracted_root=None,
        download_status="failed", download_error=last_error,
        archive_name=archive_name, source_urls=urls, official_bytes=expected_bytes,
    )


def _safe_tar_members(archive: tarfile.TarFile, dest_dir: Path) -> list[tarfile.TarInfo]:
    dest = dest_dir.resolve()
    members: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        name = member.name.replace("\\", "/")
        if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
            raise ValueError(f"unsafe absolute path in archive: {member.name}")
        target = (dest / name).resolve()
        if target != dest and dest not in target.parents:
            raise ValueError(f"archive member escapes destination: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"archive contains a link, refusing to extract: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"archive contains a non-file entry: {member.name}")
        members.append(member)
    return members


def extract_archive(archive_path: Path, dest_dir: Path,
                    *, sequence_name: str = FREIBURG1_XYZ_SEQUENCE) -> Path:
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = _safe_tar_members(archive, dest_dir)
        if hasattr(tarfile, "data_filter"):
            archive.extractall(path=dest_dir, members=members, filter="data")
        else:  # pragma: no cover - Python < 3.12
            archive.extractall(path=dest_dir, members=members)
    root = discover_sequence_root(dest_dir, sequence_name)
    if root is None:
        raise FileNotFoundError(
            f"could not locate extracted TUM sequence {sequence_name!r} under {dest_dir}")
    return root


class TumRgbSequenceCapture:
    """OpenCV-like capture over a TUM ``rgb.txt`` image list."""

    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FRAME_COUNT = 7

    def __init__(self, layout: TumSequenceLayout, *, fps: Optional[float] = None) -> None:
        self._layout = layout
        if fps is None:
            times = np.asarray([entry.timestamp_s for entry in layout.rgb_entries], np.float64)
            intervals = np.diff(times)
            fps = float(1.0 / np.median(intervals)) if intervals.size and np.median(intervals) > 0 else 30.0
        self._fps = float(fps)
        self._index = 0
        self._opened = True
        self._width = NATIVE_WIDTH
        self._height = NATIVE_HEIGHT

    @classmethod
    def from_sequence_root(
        cls, sequence_root: str | Path, *, fps: Optional[float] = None
    ) -> "TumRgbSequenceCapture":
        return cls(load_sequence_layout(Path(sequence_root)), fps=fps)

    @property
    def timestamps_s(self) -> tuple[float, ...]:
        return tuple(item.timestamp_s for item in self._layout.rgb_entries)

    @property
    def frame_index(self) -> int:
        return self._index

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop_id: int) -> float:
        if prop_id == self.CAP_PROP_FPS:
            return self._fps
        if prop_id == self.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop_id == self.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop_id == self.CAP_PROP_FRAME_COUNT:
            return float(len(self._layout.rgb_entries))
        return 0.0

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if not self._opened or self._index >= len(self._layout.rgb_entries):
            return False, None
        entry = self._layout.rgb_entries[self._index]
        image_path = self._layout.sequence_root / entry.relative_path
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        self._index += 1
        if bgr is None:
            return False, None
        self._height, self._width = bgr.shape[:2]
        return True, bgr

    def release(self) -> None:
        self._opened = False


def tum_capture_factory(
    sequence_root: str | Path, *, fps: Optional[float] = None
) -> TumRgbSequenceCapture:
    return TumRgbSequenceCapture.from_sequence_root(sequence_root, fps=fps)


def load_depth_png(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise OSError(f"could not read depth image: {path}")
    if depth.ndim != 2:
        raise ValueError(f"depth image must be single-channel: {path}")
    if depth.dtype != np.uint16:
        raise ValueError(f"TUM depth PNG must be 16-bit, got {depth.dtype}: {path}")
    return depth


def depth_to_metres(depth_u16: np.ndarray) -> np.ndarray:
    """Convert TUM 16-bit depth to optical-axis metres; missing pixels -> NaN."""
    depth = np.asarray(depth_u16)
    if depth.ndim != 2:
        raise ValueError("depth image must be two-dimensional")
    result = depth.astype(np.float64) / DEPTH_SCALE
    result[depth == 0] = np.nan
    return result


def depth_to_points(
    depth_u16: np.ndarray,
    *,
    intrinsics: dict,
    stride: int = 8,
    max_points: int = 5000,
    max_depth_m: Optional[float] = None,
) -> np.ndarray:
    """Vectorised back-projection of valid depth pixels to camera-frame metres."""
    if stride < 1:
        raise ValueError("stride must be at least 1")
    z = depth_to_metres(depth_u16)[::stride, ::stride]
    if max_depth_m is not None:
        z[z > max_depth_m] = np.nan
    rows, cols = np.nonzero(np.isfinite(z) & (z > 0.0))
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if rows.size > max_points:
        selector = np.linspace(0, rows.size - 1, max_points).round().astype(int)
        rows, cols = rows[selector], cols[selector]
    sampled = z[rows, cols]
    u = cols.astype(np.float64) * stride
    v = rows.astype(np.float64) * stride
    x = (u - intrinsics["cx"]) * sampled / intrinsics["fx"]
    y = (v - intrinsics["cy"]) * sampled / intrinsics["fy"]
    return np.stack([x, y, sampled], axis=1)


def backproject_dense(
    depth_u16: np.ndarray,
    *,
    intrinsics: dict,
    max_depth_m: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense camera-frame point cloud with NaN where depth is invalid."""
    z = depth_to_metres(depth_u16).astype(np.float32)
    if max_depth_m is not None:
        z[z > max_depth_m] = np.nan
    valid = np.isfinite(z) & (z > 0.0)
    height, width = z.shape
    u = np.arange(width, dtype=np.float32)[None, :]
    v = np.arange(height, dtype=np.float32)[:, None]
    x = (u - np.float32(intrinsics["cx"])) * z / np.float32(intrinsics["fx"])
    y = (v - np.float32(intrinsics["cy"])) * z / np.float32(intrinsics["fy"])
    points = np.stack([x, y, z], axis=-1)
    points[~valid] = np.nan
    return points.astype(np.float32), valid


def associate_groundtruth_poses(
    groundtruth: Sequence[TumPose],
    timestamps_s: Sequence[float],
    *,
    max_difference_s: float = 0.02,
) -> tuple[Optional[TumPose], ...]:
    """Nearest ground-truth pose for each timestamp, or ``None`` outside tolerance."""
    if max_difference_s < 0.0:
        raise ValueError("max_difference_s must be non-negative")
    if not groundtruth:
        return tuple(None for _ in timestamps_s)
    poses = list(groundtruth)
    times = np.asarray([pose.timestamp_s for pose in poses], np.float64)
    out: list[Optional[TumPose]] = []
    for timestamp in timestamps_s:
        index = int(np.searchsorted(times, timestamp, "left"))
        candidates = []
        if index < len(poses):
            candidates.append(index)
        if index > 0:
            candidates.append(index - 1)
        best = min(candidates, key=lambda i: (abs(times[i] - timestamp), i))
        out.append(poses[best] if abs(times[best] - timestamp) <= max_difference_s else None)
    return tuple(out)


def groundtruth_positions_at_timestamps(
    groundtruth: Sequence[TumPose],
    timestamps_s: Sequence[float],
    *,
    max_difference_s: float = 0.02,
) -> tuple[tuple[float, float, float], ...]:
    """GT positions for each timestamp; NaN triple where no pose is within tolerance."""
    associated = associate_groundtruth_poses(
        groundtruth, timestamps_s, max_difference_s=max_difference_s)
    return tuple(
        pose.position if pose is not None else (float("nan"), float("nan"), float("nan"))
        for pose in associated
    )


def ensure_sequence(
    dataset_dir: Path,
    sequence_name: str,
    archive_name: str,
    urls: tuple[str, ...],
    official_bytes: int,
    *,
    allow_download: bool = True,
) -> tuple[Optional[TumSequenceLayout], TumDatasetProvenance]:
    """Locate, or (optionally) download and extract, the official sequence."""
    dataset_dir = Path(dataset_dir)
    archive_path = dataset_dir / archive_name
    existing = discover_sequence_root(dataset_dir, sequence_name)
    if existing is not None:
        archive_sha = sha256_file(archive_path) if archive_path.is_file() else None
        status = "sequence_present"
        error = None
        if archive_path.is_file() and archive_path.stat().st_size != official_bytes:
            status = "sequence_present_archive_size_mismatch"
            error = (
                f"extracted sequence is present, but the local archive is "
                f"{archive_path.stat().st_size} bytes instead of "
                f"{official_bytes}; treating extracted data as unverified")
        return load_sequence_layout(existing), _provenance(
            archive_path=archive_path if archive_path.is_file() else None,
            extracted_root=existing,
            download_status=status,
            download_error=error,
            archive_sha256=archive_sha,
            archive_name=archive_name, source_urls=urls, official_bytes=official_bytes,
        )

    if archive_path.is_file():
        provenance = download_archive(
            dataset_dir, urls=urls, archive_name=archive_name, expected_bytes=official_bytes)
        if provenance.archive_size_matches_official is False:
            return None, provenance
        root = extract_archive(archive_path, dataset_dir, sequence_name=sequence_name)
        return load_sequence_layout(root), _provenance(
            archive_path=archive_path, extracted_root=root,
            download_status="extracted_local_archive", download_error=None,
            archive_sha256=provenance.archive_sha256,
            archive_name=archive_name, source_urls=urls, official_bytes=official_bytes,
        )

    if not allow_download:
        return None, _provenance(
            archive_path=None, extracted_root=None, download_status="missing",
            download_error="dataset not present and download disabled",
            archive_name=archive_name, source_urls=urls, official_bytes=official_bytes,
        )

    provenance = download_archive(
        dataset_dir, urls=urls, archive_name=archive_name, expected_bytes=official_bytes)
    if provenance.download_status == "failed" or provenance.archive_path is None:
        return None, provenance
    if provenance.archive_size_matches_official is False:
        return None, provenance
    root = extract_archive(Path(provenance.archive_path), dataset_dir,
                           sequence_name=sequence_name)
    return load_sequence_layout(root), _provenance(
        archive_path=Path(provenance.archive_path), extracted_root=root,
        download_status=provenance.download_status, download_error=None,
        archive_sha256=provenance.archive_sha256,
        archive_name=archive_name, source_urls=urls, official_bytes=official_bytes,
    )


def ensure_freiburg1_xyz(
    dataset_dir: Path,
    *,
    allow_download: bool = True,
) -> tuple[Optional[TumSequenceLayout], TumDatasetProvenance]:
    """Locate, or (optionally) download and extract, the Freiburg 1 XYZ sequence."""
    return ensure_sequence(
        dataset_dir,
        FREIBURG1_XYZ_SEQUENCE,
        FREIBURG1_XYZ_ARCHIVE,
        FREIBURG1_XYZ_URLS,
        OFFICIAL_ARCHIVE_BYTES,
        allow_download=allow_download,
    )


def ensure_freiburg1_rpy(
    dataset_dir: Path,
    *,
    allow_download: bool = True,
) -> tuple[Optional[TumSequenceLayout], TumDatasetProvenance]:
    """Locate, or (optionally) download and extract, the Freiburg 1 RPY sequence."""
    return ensure_sequence(
        dataset_dir,
        FREIBURG1_RPY_SEQUENCE,
        FREIBURG1_RPY_ARCHIVE,
        FREIBURG1_RPY_URLS,
        FREIBURG1_RPY_ARCHIVE_BYTES,
        allow_download=allow_download,
    )


def ensure_freiburg1_360(
    dataset_dir: Path,
    *,
    allow_download: bool = True,
) -> tuple[Optional[TumSequenceLayout], TumDatasetProvenance]:
    """Locate, or (optionally) download and extract, the Freiburg 1 360 sequence."""
    return ensure_sequence(
        dataset_dir,
        FREIBURG1_360_SEQUENCE,
        FREIBURG1_360_ARCHIVE,
        FREIBURG1_360_URLS,
        FREIBURG1_360_ARCHIVE_BYTES,
        allow_download=allow_download,
    )
