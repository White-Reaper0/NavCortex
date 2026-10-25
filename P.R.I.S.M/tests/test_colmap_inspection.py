"""Model-free sparse-model QA tests using in-memory records."""
import math

import pytest

from prism.colmap_inspection import (ColmapCamera, ColmapImage, ColmapModelError, ColmapPoint,
                                     SparseModel, build_qa_report, trajectory_rows, write_ply)


def model(*, points=None, images=None):
    return SparseModel((ColmapCamera(1, 1, 100, 80, (50., 50., 40., 0.)),),
                       tuple(images if images is not None else (
                           ColmapImage(1, (1., 0., 0., 0.), (0., 0., 0.), 1, "frame_0000.png", (1,)),
                           ColmapImage(2, (1., 0., 0., 0.), (-1., 0., 0.), 1, "frame_0008.png", (1,)),
                       )), tuple(points if points is not None else (
                           ColmapPoint(1, (1., 2., 3.), (1, 2, 3), 0.5, ((1, 0), (2, 0))),)))


MANIFEST = {"frames": [{"file": "frame_0000.png", "source_frame_index": 0, "source_time_s": 0.0},
                       {"file": "frame_0008.png", "source_frame_index": 8, "source_time_s": 8 / 30}]}


def test_report_tracks_identity_reprojection_and_trajectory_continuity():
    report = build_qa_report(model(), MANIFEST)
    assert report["registered_images"] == 2 and report["points3D"] == 1
    assert report["image_rows"][0]["source_frame_index"] == 0
    assert report["reprojection_error_px"]["mean"] == pytest.approx(0.5)
    assert report["trajectory"]["camera_pose_count"] == 2
    assert report["trajectory"]["image_graph_components"] == 1
    assert report["physical_scale"] == "unverified_arbitrary_colmap_units"


def test_incomplete_models_and_missing_camera_poses_are_reported():
    broken = model(images=(ColmapImage(1, (0., 0., 0., 0.), (0., 0., 0.), 7, "missing.png", ()),))
    report = build_qa_report(broken, {"frames": []})
    assert any("missing camera 7" in issue for issue in report["issues"])
    assert any("invalid quaternion" in issue for issue in report["issues"])


def test_invalid_points_are_flagged_and_cannot_be_exported(tmp_path):
    bad = ColmapPoint(8, (math.nan, 0., 0.), (0, 0, 0), 1., ())
    report = build_qa_report(model(points=(bad,)), MANIFEST)
    assert report["invalid_point_ids"] == [8]
    with pytest.raises(ColmapModelError, match="non-finite"):
        write_ply((bad,), tmp_path / "bad.ply")


def test_trajectory_rows_are_source_ordered_and_ply_is_inspectable(tmp_path):
    rows = trajectory_rows(model(), MANIFEST)
    assert [row["source_frame_index"] for row in rows] == [0, 8]
    path = tmp_path / "points.ply"
    write_ply(model().points, path)
    assert "element vertex 1" in path.read_text() and "1 2 3 1 2 3" in path.read_text()
