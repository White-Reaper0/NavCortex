"""Model-free tests for explicit COLMAP scale-reference validation."""
import json

import pytest

from prism.colmap_inspection import ColmapCamera, ColmapImage, ColmapPoint, SparseModel
from prism.scale_alignment import (LandmarkReference, ScaleAlignmentError, ScaleReference, ScaleStatus,
                                   aligned_report, load_scale_reference, scale_points,
                                   verify_reference_against_model, write_scaled_exports)


def model():
    return SparseModel((ColmapCamera(1, 2, 100, 80, (50., 50., 40., 0.)),),
                       (ColmapImage(1, (1., 0., 0., 0.), (0., 0., 0.), 1, "frame_0000.png", (1, 2)),),
                       (ColmapPoint(1, (0., 0., 0.), (0, 0, 0), 0.1, ((1, 0),)),
                        ColmapPoint(2, (2., 0., 0.), (0, 0, 0), 0.2, ((1, 1),))))


def reference(**changes):
    values = dict(first=LandmarkReference("edge-a", 1, (0., 0., 0.)),
                  second=LandmarkReference("edge-b", 2, (2., 0., 0.)), measured_distance=4., units="m",
                  measurement_provenance={"method": "tape", "record": "field-1"}, reviewed=True,
                  uncertainty=0.1)
    values.update(changes)
    return ScaleReference(**values)


def test_reviewed_reference_calculates_scale_and_preserves_provenance():
    item = reference()
    assert item.reconstruction_distance == pytest.approx(2.)
    assert item.scale_metres_per_colmap_unit == pytest.approx(2.)
    report = aligned_report(item, calibration={"status": "MISSING"})
    assert report["status"] == ScaleStatus.METRIC_SCALE_ALIGNED.value
    assert report["reference"]["measurement_provenance"]["record"] == "field-1"
    assert report["relative_reference_uncertainty"] == pytest.approx(0.025)


def test_missing_or_unreviewed_reference_remains_explicitly_unverified():
    assert aligned_report(None, calibration={})["status"] == ScaleStatus.METRIC_SCALE_UNVERIFIED.value
    report = aligned_report(reference(reviewed=False), calibration={})
    assert report["status"] == ScaleStatus.METRIC_SCALE_UNVERIFIED.value
    assert "not marked reviewed" in report["reason"]


@pytest.mark.parametrize("changes", [{"measured_distance": 0.}, {"units": "feet"}])
def test_invalid_measurements_are_rejected_at_construction(changes):
    with pytest.raises(ScaleAlignmentError):
        reference(**changes)


def test_degenerate_baseline_is_rejected_before_reporting_or_export():
    item = reference(second=LandmarkReference("edge-b", 2, (0., 0., 0.)))
    with pytest.raises(ScaleAlignmentError, match="baseline"):
        _ = item.scale_metres_per_colmap_unit
    with pytest.raises(ScaleAlignmentError, match="baseline"):
        aligned_report(item, calibration={})


def test_model_binding_rejects_missing_or_mismatched_reconstruction_points():
    with pytest.raises(ScaleAlignmentError, match="missing"):
        verify_reference_against_model(reference(first=LandmarkReference("edge-a", 9, (0., 0., 0.))), model())
    with pytest.raises(ScaleAlignmentError, match="do not match"):
        verify_reference_against_model(reference(second=LandmarkReference("edge-b", 2, (3., 0., 0.))), model())


def test_scaled_exports_are_separate_and_metre_labeled(tmp_path):
    exports = write_scaled_exports(model(), {"frames": [{"file": "frame_0000.png", "source_frame_index": 0, "source_time_s": 0.0}]},
                                   tmp_path / "scaled", reference())
    assert exports["scaled_point_cloud"].endswith("_m.ply")
    assert "4 0 0" in (tmp_path / "scaled" / "sparse_points_scaled_m.ply").read_text()
    assert "center_x_m" in (tmp_path / "scaled" / "camera_trajectory_scaled_m.csv").read_text()


def test_reference_file_schema_validation(tmp_path):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({"schema": "wrong"}))
    with pytest.raises(ScaleAlignmentError, match="schema"):
        load_scale_reference(path)
