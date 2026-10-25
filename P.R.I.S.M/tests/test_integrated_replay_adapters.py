"""Model-free tests for the narrow integrated replay adapters."""
from pathlib import Path
import json

from prism.belief_ledger import BeliefLedger, EvidenceRelation
from prism.doubt_engine import DoubtEngine
from prism.integrations.geometry import open3d_sparse_inspection, sparse_geometry_report
from prism.integrations.localization import SELECTED_BACKEND, localization_backend_report
from prism.integrations.mapping_diagnostic import mapping_diagnostic_status
from prism.integrations.terrain import terrain_status
from prism.integrations.tracking import ImageSpaceIoUTracker, iou
from prism.observations import CoordinateFrame, EvidencePayload, ObjectDetection, Observation, Validity
from prism.colmap_inspection import read_sparse_model


ROOT = Path(__file__).resolve().parents[1]


def detection(box=(10.0, 10.0, 30.0, 30.0)):
    return ObjectDetection(1, "coco-2017@80", .8, box)


def test_iou_tracking_preserves_frame_timestamp_and_prediction_boundary():
    tracker = ImageSpaceIoUTracker(max_missed_frames=1)
    first = tracker.update((detection(),), frame_index=4, capture_at_ns=400)
    second = tracker.update((detection((11., 10., 31., 30.)),), frame_index=5, capture_at_ns=500)
    missing = tracker.update((), frame_index=6, capture_at_ns=600)
    assert first[0].track_id == second[0].track_id
    assert second[0].observed and not second[0].predicted
    assert missing[0].predicted and not missing[0].observed
    assert "not a new detector observation" in missing[0].prediction_note


def test_iou_is_bounded_and_class_mismatch_creates_new_track():
    assert iou((0., 0., 1., 1.), (2., 2., 3., 3.)) == 0.0
    tracker = ImageSpaceIoUTracker()
    first = tracker.update((detection(),), frame_index=0, capture_at_ns=0)
    other = ObjectDetection(2, "coco-2017@80", .8, (10., 10., 30., 30.))
    second = tracker.update((other,), frame_index=1, capture_at_ns=1)
    observed = next(item for item in second if item.observed)
    assert first[0].track_id != observed.track_id


def test_geometry_associates_only_exact_colmap_source_frame_identities():
    report = sparse_geometry_report(
        ROOT / "output/09_colmap_sparse/clip_01_repro/sparse/0",
        ROOT / "output/09_colmap_sparse/clip_01_repro/frames_manifest.json", (0, 1, 8))
    assert report["status"] == "RELATIVE_ONLY"
    assert [row["source_frame_index"] for row in report["exact_pose_matches"]] == [0, 8]
    assert report["unmatched_source_frame_indices"] == [1]
    assert report["physical_scale"] == "METRIC_SCALE_UNVERIFIED"


def test_terrain_is_explicitly_unavailable_without_valid_metric_spatial_input():
    report = terrain_status(geometry_report={"physical_scale": "METRIC_SCALE_UNVERIFIED"})
    assert report["status"] == "BLOCKED_MISSING_INPUT"
    assert report["grid"] is None


def test_open3d_loads_native_colmap_points_without_rescaling(tmp_path: Path):
    model = read_sparse_model(ROOT / "output/09_colmap_sparse/clip_01_repro/sparse/0")
    ply = tmp_path / "native.ply"
    summary = open3d_sparse_inspection(model, export_ply_path=ply)
    if summary["status"] == "UNAVAILABLE":
        return  # Open3D optional on some hosts
    assert summary["status"] == "LOADED"
    assert summary["point_count"] == len(model.points)
    assert "metres" not in summary["coordinate_frame"].lower()
    assert ply.is_file()


def test_localization_selects_opencv_colmap_and_blocks_slam_candidates():
    report = localization_backend_report(clip_has_rgb_only=True)
    assert report["selected_backend"] == SELECTED_BACKEND
    assert report["candidates"]["opencv_colmap_diagnostic"]["status"] == "ACTIVE"
    assert report["candidates"]["stella_vslam"]["status"] == "BLOCKED_MISSING_INPUT"
    assert report["candidates"]["rtabmap"]["status"] == "BLOCKED_MISSING_INPUT"


def test_mapping_diagnostic_does_not_claim_active_map_update():
    status = mapping_diagnostic_status(
        geometry_report={"physical_scale": "METRIC_SCALE_UNVERIFIED"},
        visual_motion_summary={"tracking_only": True, "trajectory_produced": False},
    )
    assert status["active_map_updated"] is False
    assert status["status"] == "DIAGNOSTIC_ONLY"


def test_doubt_engine_preserves_unavailable_terrain_in_advisory():
    ledger = BeliefLedger()
    terrain = Observation(
        observation_id="test:terrain-unavailable",
        source="test",
        captured_at_ns=10,
        coordinate_frame=CoordinateFrame.LOCAL_MAP,
        validity=Validity.UNAVAILABLE,
        provenance={"reason": "test"},
        evidence=EvidencePayload(),
    )
    ledger.record("terrain-spatial-evidence", terrain, EvidenceRelation.SUPPORTS, expires_at_ns=11)
    ledger.advance_to(10)
    assessment = DoubtEngine().assess(ledger, at_ns=10, hypothesis_ids=("terrain-spatial-evidence",))
    assert assessment.recommendations
    assert any(item.kind.value == "cautious_stop" for item in assessment.recommendations)


def test_integrated_runner_refuses_nonempty_output(tmp_path: Path):
    from tools.run_prism_integrated_replay import run

    output = tmp_path / "out"
    output.mkdir()
    (output / "marker.txt").write_text("keep", encoding="utf-8")
    clip = ROOT / "clips/clip_01.mp4"
    try:
        run(clip, output, max_frames=1, weights=ROOT / "model_assets/libreyolo/LibreYOLO9t.pt",
            model_dir=ROOT / "output/09_colmap_sparse/clip_01_repro/sparse/0",
            manifest=ROOT / "output/09_colmap_sparse/clip_01_repro/frames_manifest.json")
    except ValueError as error:
        assert "refusing to overwrite" in str(error)
    else:
        raise AssertionError("expected non-empty output refusal")


def test_sparse_geometry_report_is_deterministic_for_fixed_inputs():
    kwargs = dict(
        model_dir=ROOT / "output/09_colmap_sparse/clip_01_repro/sparse/0",
        manifest_path=ROOT / "output/09_colmap_sparse/clip_01_repro/frames_manifest.json",
        requested_source_frames=(0, 8),
    )
    first = json.dumps(sparse_geometry_report(**kwargs), sort_keys=True)
    second = json.dumps(sparse_geometry_report(**kwargs), sort_keys=True)
    assert first == second
