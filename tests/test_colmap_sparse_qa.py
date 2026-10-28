"""Model-free tests for the COLMAP sparse-model QA tooling.

These tests construct :class:`tools.colmap_sparse_qa.Model` objects directly
rather than depending on the real COLMAP output, so they run offline without
the external COLMAP binary and without overwriting any reconstruction output.
"""

from pathlib import Path

import numpy as np
import pytest

from tools.colmap_sparse_qa import (
    Camera,
    Image,
    Model,
    Point2D,
    Point3D,
    build_qa_report,
    export_point_cloud_ply,
    export_trajectory_csv,
    load_colmap_model,
)


def _camera(model="SIMPLE_RADIAL", width=960, height=540, params=(534.46, 480.0, 270.0, -0.15)):
    return Camera(camera_id=1, model=model, width=width, height=height, params=params)


def _image(image_id, name, q=(1.0, 0.0, 0.0, 0.0), t=(0.0, 0.0, 0.0), n_pts=0,
           camera_id=1):
    pts = tuple(
        Point2D(x=float(i), y=float(i), point3D_id=i) for i in range(n_pts)
    )
    return Image(
        image_id=image_id,
        qw=q[0],
        qx=q[1],
        qy=q[2],
        qz=q[3],
        tx=t[0],
        ty=t[1],
        tz=t[2],
        camera_id=camera_id,
        name=name,
        points2D=pts,
    )


def _point(point_id, xyz, error, track, color=(128, 128, 128)):
    return Point3D(
        point3D_id=point_id,
        xyz=tuple(xyz),
        error=error,
        track=tuple(track),
        color=color,
    )


def _complete_model():
    cam = _camera()
    images = {
        1: _image(1, "frame_0000.png", t=(0.0, 0.0, 0.0), n_pts=3),
        2: _image(2, "frame_0008.png", t=(0.1, 0.0, 0.0), n_pts=3),
        3: _image(3, "frame_0015.png", t=(0.2, 0.0, 0.0), n_pts=3),
    }
    points = {
        1: _point(1, (1.0, 2.0, 3.0), 0.5, [(1, 0), (2, 0)]),
        2: _point(2, (2.0, 3.0, 4.0), 1.5, [(2, 1), (3, 1)]),
        3: _point(3, (3.0, 4.0, 5.0), 2.5, [(1, 2), (3, 2)]),
    }
    return Model(cameras={1: cam}, images=images, points=points)


def test_build_qa_report_complete_model():
    model = _complete_model()
    report = build_qa_report(model, "synthetic_model")

    assert report.registered_images == 3
    assert report.total_points == 3
    assert report.total_observations == 6
    assert report.mean_track_length == pytest.approx(2.0)
    assert report.mean_observations_per_image == pytest.approx(2.0)
    assert report.mean_reprojection_error_px == pytest.approx(1.5)
    assert report.camera_model == "SIMPLE_RADIAL"
    assert report.camera_params == (534.46, 480.0, 270.0, -0.15)
    assert report.image_size == (960, 540)
    assert len(report.trajectory) == 3
    assert report.connectivity["fully_connected"] is True
    assert report.connectivity["connected_components"] == 1
    assert any("self-calibrated" in n for n in report.notes)


def test_build_qa_report_empty_model():
    model = Model(cameras={}, images={}, points={})
    report = build_qa_report(model, "empty_model")

    assert report.registered_images == 0
    assert report.total_points == 0
    assert report.total_observations == 0
    assert report.mean_track_length == 0.0
    assert report.mean_observations_per_image == 0.0
    assert report.mean_reprojection_error_px is None
    assert report.connectivity["fully_connected"] is True
    assert any("empty/degenerate" in n for n in report.notes)


def test_build_qa_report_incomplete_model_no_points():
    model = Model(
        cameras={1: _camera()},
        images={1: _image(1, "frame_0000.png", n_pts=0)},
        points={},
    )
    report = build_qa_report(model, "incomplete_model")

    assert report.registered_images == 1
    assert report.total_points == 0
    assert report.total_observations == 0
    assert report.mean_track_length == 0.0
    assert report.mean_reprojection_error_px is None
    assert any("No observations" in n for n in report.notes)


def test_build_qa_report_disconnected_model():
    cam = _camera()
    images = {
        1: _image(1, "frame_0000.png", n_pts=1),
        2: _image(2, "frame_0008.png", n_pts=1),
    }
    # Two images observe disjoint points: visibility graph disconnected.
    points = {
        1: _point(1, (1.0, 0.0, 0.0), 0.5, [(1, 0)]),
        2: _point(2, (2.0, 0.0, 0.0), 0.5, [(2, 0)]),
    }
    model = Model(cameras={1: cam}, images=images, points=points)
    report = build_qa_report(model, "disconnected_model")

    assert report.registered_images == 2
    assert report.connectivity["fully_connected"] is False
    assert report.connectivity["connected_components"] == 2
    assert any("disconnected" in n for n in report.notes)


def test_invalid_points_are_flagged_as_outliers():
    cam = _camera()
    images = {
        1: _image(1, "frame_0000.png", n_pts=1),
        2: _image(2, "frame_0008.png", n_pts=1),
    }
    points = {
        1: _point(1, (1.0, 2.0, 3.0), 0.5, [(1, 0), (2, 0)]),
        2: _point(2, (float("nan"), 2.0, 3.0), 0.5, [(1, 0)]),  # non-finite
        3: _point(3, (5.0, 5.0, 5.0), 50.0, [(2, 0)]),  # high error, singleton
        4: _point(4, (0.0, 0.0, 0.0), 0.5, [(1, 0)]),  # singleton track
    }
    model = Model(cameras={1: cam}, images=images, points=points)
    report = build_qa_report(model, "invalid_points_model")

    flag_ids = {o["point3D_id"] for o in report.outliers}
    assert 2 in flag_ids
    assert any("non_finite_position" in o["flags"] for o in report.outliers if o["point3D_id"] == 2)
    assert 3 in flag_ids
    assert any("high_reprojection_error" in o["flags"] for o in report.outliers if o["point3D_id"] == 3)
    assert 4 in flag_ids
    assert any("singleton_track" in o["flags"] for o in report.outliers if o["point3D_id"] == 4)


def test_model_with_image_referencing_missing_camera():
    cam = _camera()
    images = {
        1: _image(1, "frame_0000.png", camera_id=99, n_pts=0),
    }
    model = Model(cameras={1: cam}, images=images, points={})
    report = build_qa_report(model, "missing_camera_model")

    assert report.registered_images == 1
    assert report.camera_model == "SIMPLE_RADIAL"  # reports first known camera


def test_export_point_cloud_ply_structure(tmp_path):
    model = _complete_model()
    out = tmp_path / "points.ply"
    export_point_cloud_ply(model, out)

    text = out.read_text(encoding="utf-8")
    assert text.startswith("ply\n")
    assert "format ascii 1.0\n" in text
    assert "element vertex 3\n" in text
    assert "property float x\n" in text
    assert "property uchar red\n" in text
    assert "property float reprojection_error_px\n" in text
    assert "property uint track_length\n" in text
    assert "end_header\n" in text
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith(("ply", "format", "element", "property", "end_header"))]
    assert len(lines) == 3


def test_export_point_cloud_ply_excludes_invalid_by_default(tmp_path):
    cam = _camera()
    images = {1: _image(1, "frame_0000.png", n_pts=0)}
    points = {
        1: _point(1, (1.0, 2.0, 3.0), 0.5, [(1, 0)]),
        2: _point(2, (float("nan"), 2.0, 3.0), 0.5, [(1, 0)]),
    }
    model = Model(cameras={1: cam}, images=images, points=points)
    out = tmp_path / "points.ply"
    export_point_cloud_ply(model, out)

    text = out.read_text(encoding="utf-8")
    assert "element vertex 1\n" in text


def test_export_trajectory_csv_structure(tmp_path):
    model = _complete_model()
    out = tmp_path / "camera_trajectory.csv"
    export_trajectory_csv(model, out)

    text = out.read_text(encoding="utf-8")
    header = text.splitlines()[0]
    assert "image_id" in header
    assert "file" in header
    assert "qw" in header
    assert "tx" in header
    assert "camera_center_x" in header
    assert "num_points2D" in header
    rows = [ln for ln in text.splitlines()[1:] if ln.strip()]
    assert len(rows) == 3


def test_build_qa_report_with_frame_lookup(tmp_path):
    model = _complete_model()
    frame_lookup = {
        "frame_0000.png": {"source_frame_index": 0, "source_time_s": 0.0},
        "frame_0008.png": {"source_frame_index": 8, "source_time_s": 0.2667},
        "frame_0015.png": {"source_frame_index": 15, "source_time_s": 0.5},
    }
    report = build_qa_report(model, "lookup_model", frame_lookup=frame_lookup)
    assert report.trajectory[0]["source_frame_index"] == 0
    assert report.trajectory[0]["source_time_s"] == 0.0
    assert report.trajectory[2]["file"] == "frame_0015.png"


def test_load_colmap_model_parses_synthetic_binary(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    # cameras.bin: one SIMPLE_RADIAL camera, 4 params.
    cameras = struct_pack_camera()
    (model_dir / "cameras.bin").write_bytes(cameras)
    # images.bin: one image, identity pose, zero points2D.
    images = struct_pack_image(image_id=1, name="frame_0000.png", num_points2d=0)
    (model_dir / "images.bin").write_bytes(images)
    # points3D.bin: one point at origin, zero error, zero track.
    points = struct_pack_points(num_points=1)
    (model_dir / "points3D.bin").write_bytes(points)

    model = load_colmap_model(model_dir)
    assert len(model.cameras) == 1
    assert model.cameras[1].model == "SIMPLE_RADIAL"
    assert len(model.images) == 1
    assert model.images[1].name == "frame_0000.png"
    assert len(model.points) == 1
    assert model.points[1].xyz == (0.0, 0.0, 0.0)


def struct_pack_camera():
    import struct

    buf = struct.pack("<Q", 1)  # num_cameras u64
    buf += struct.pack("<IIIIQ", 1, 2, 960, 540, 4)  # camera
    buf += struct.pack("<4d", 534.46, 480.0, 270.0, -0.15)
    return buf


def struct_pack_image(image_id, name, num_points2d):
    import struct

    name_bytes = name.encode("utf-8") + b"\x00"
    buf = struct.pack("<Q", 1)  # num_images u64
    buf += struct.pack("<I4d3dI", image_id, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1)
    buf += name_bytes
    buf += struct.pack("<Q", num_points2d)
    return buf


def struct_pack_points(num_points):
    import struct

    buf = struct.pack("<Q", num_points)
    for _ in range(num_points):
        buf += struct.pack("<Q3ddQ", 1, 0.0, 0.0, 0.0, 0.0, 0)
        buf += struct.pack("<3B", 128, 128, 128)
    return buf
