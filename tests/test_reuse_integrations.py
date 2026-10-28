"""Model-free tests for the reuse-first integration layer."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from prism.belief_ledger import EvidenceRelation
from prism.integrations.localization import localization_backend_report
from prism.integrations.provider_contract import FrameObservationProvider
from prism.integrations.terrain import terrain_status
from prism.integrations.terrain_grid import (
    LAYER_NAMES,
    slope_degrees,
    terrain_grid_from_bev,
    terrain_roughness,
)
from prism.integrations.yolop_provider import (
    YOLOPDetectionProvider,
    YOLOPProviderConfig,
    YOLOPProviderError,
    segmentation_summary,
)
from prism.video_observation_replay import DecodedFrame


SHA = "a" * 64


def _config(**overrides) -> YOLOPProviderConfig:
    values = dict(source="test", hypothesis_id="object-detection-evidence",
                  relation=EvidenceRelation.SUPPORTS, expires_after_ns=5,
                  weights_sha256=SHA, confidence_threshold=0.25)
    values.update(overrides)
    return YOLOPProviderConfig(**values)


def _frame() -> DecodedFrame:
    # 100x200 -> letterbox 320 gives scale 1.6 and 80 px vertical padding.
    return DecodedFrame(3, 300, 200, 100, np.zeros((100, 200, 3), np.uint8))


def _yolop_outputs():
    detections = np.array([
        [160.0, 80.0, 64.0, 32.0, 0.90, 0.80],   # conf 0.72, kept
        [40.0, 40.0, 20.0, 20.0, 0.10, 0.50],    # conf 0.05, dropped
    ], np.float32)
    drivable = np.array([[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]], np.float32)[None]
    lane = np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]], np.float32)[None]
    return [detections, drivable, lane]


def _bev(height: np.ndarray, observed: np.ndarray | None = None,
         trav: np.ndarray | None = None) -> SimpleNamespace:
    height = np.asarray(height, np.float32)
    seen = np.isfinite(height) if observed is None else np.asarray(observed, bool)
    classes = (np.where(seen, 0, 3).astype(np.uint8) if trav is None
               else np.asarray(trav, np.uint8))
    probability = np.zeros((4, *height.shape), np.float32)
    for index in range(4):
        probability[index] = classes == index
    return SimpleNamespace(height=height, trav=classes, trav_prob=probability,
                           conf=seen.astype(np.float32), age=np.zeros_like(height),
                           hits=seen.astype(np.float32), observed=seen)


def _fake_provider(width=200, height=100):
    return YOLOPDetectionProvider(lambda blob: _yolop_outputs(), _config())


def test_yolop_provider_satisfies_narrow_provider_contract():
    provider = _fake_provider()
    assert isinstance(provider, FrameObservationProvider)
    assert provider.config.taxonomy == "bdd100k-vehicle@1"


def test_yolop_decodes_single_vehicle_and_preserves_score_semantics():
    provider = _fake_provider()
    supplied, outputs = provider.run_detailed(_frame())
    evidence = supplied.observation.evidence.object_detections
    assert evidence is not None
    assert len(evidence.detections) == 1
    detection = evidence.detections[0]
    assert detection.class_id == "vehicle"
    assert detection.taxonomy == "bdd100k-vehicle@1"
    assert detection.score == pytest.approx(0.72)
    x0, y0, x1, y1 = detection.bbox_xyxy
    assert (x0, x1) == pytest.approx((80.0, 120.0))
    assert 0.0 <= y0 < y1 <= 100.0
    detector = supplied.observation.provenance["detector"]
    assert detector["segmentation_outputs_used"] is False
    assert "not traversability" in detector["segmentation_note"]
    assert "unverified" in detector["license_note"]
    assert len(outputs) == 3


def test_yolop_segmentation_summary_is_descriptive_not_traversability():
    summary = segmentation_summary(_yolop_outputs())
    assert summary["drivable_class_1_fraction"] == pytest.approx(0.5)
    assert summary["lane_class_1_fraction"] == pytest.approx(0.5)
    assert "not traversability" in summary["semantics"]
    assert "accuracy" in summary["semantics"]


def test_yolop_sha_mismatch_is_refused_before_loading(tmp_path: Path):
    checkpoint = tmp_path / "fake.onnx"
    checkpoint.write_bytes(b"not a real model")
    with pytest.raises(YOLOPProviderError, match="SHA-256 mismatch"):
        YOLOPDetectionProvider.from_onnx(
            checkpoint, expected_sha256="0" * 64, source="test",
            hypothesis_id="object-detection-evidence",
            relation=EvidenceRelation.SUPPORTS, expires_after_ns=1)


def test_yolop_malformed_detection_output_is_rejected():
    provider = YOLOPDetectionProvider(
        lambda blob: [np.zeros((2, 4), np.float32), np.zeros((1, 2, 2, 2)), np.zeros((1, 2, 2, 2))],
        _config())
    with pytest.raises(YOLOPProviderError):
        provider(_frame())


def test_terrain_grid_preserves_unknown_and_marks_missing_layers():
    ramp = np.tile((np.arange(12) * 0.04).astype(np.float32), (12, 1))
    ramp[5:7, 5:7] = np.nan
    grid = terrain_grid_from_bev(_bev(ramp), resolution_m=0.06)
    classes = grid.layer("trav_class")
    assert (classes[5:7, 5:7] == 3).all()
    assert np.isnan(grid.layer("elevation_m")[5:7, 5:7]).all()
    assert np.isnan(grid.layer("slope_deg")[5:7, 5:7]).all()
    assert np.isnan(grid.layer("roughness_m")[5:7, 5:7]).all()
    # Directly adjacent cells cannot have a slope: their 3x3 neighbourhood crosses unmeasured space.
    assert np.isnan(grid.layer("slope_deg")[4, 5])
    assert np.isfinite(grid.layer("slope_deg")[9, 9])
    assert grid.layer("slope_deg")[9, 9] == pytest.approx(
        np.degrees(np.arctan(0.04 / 0.06)), abs=0.5)
    summary = grid.summary()
    assert set(LAYER_NAMES) <= set(summary["layers"])
    assert summary["unknown_is_traversable"] is False
    assert summary["fractions"]["unknown"] == pytest.approx(4 / 144)
    assert set(summary["unsupported_layers"]) == {"variance", "normals"}
    assert summary["status"] == "EXPERIMENTAL"


def test_flat_region_has_zero_slope_but_borders_stay_unmeasured():
    flat = np.zeros((10, 10), np.float32)
    slope = slope_degrees(flat, 0.06)
    assert slope[5, 5] == pytest.approx(0.0)
    assert np.isnan(slope[0, 0])
    roughness = terrain_roughness(flat, 0.06)
    assert roughness[5, 5] == pytest.approx(0.0)


def test_terrain_status_requires_metric_scale_before_layers():
    bev = _bev(np.zeros((8, 8), np.float32))
    blocked = terrain_status(geometry_report={"physical_scale": "METRIC_SCALE_UNVERIFIED"})
    assert blocked["status"] == "BLOCKED_MISSING_INPUT"
    assert blocked["grid"] is None
    aligned = terrain_status(geometry_report={"physical_scale": "METRIC_SCALE_ALIGNED"})
    assert aligned["status"] == "UNVALIDATED"
    assert aligned["grid"] is None
    layered = terrain_status(geometry_report={"physical_scale": "METRIC_SCALE_ALIGNED"}, bev=bev)
    assert layered["status"] == "UNVALIDATED"
    assert layered["grid"].layer("trav_class").shape == (8, 8)


def test_localization_reports_basalt_blocked_without_fabricated_imu():
    report = localization_backend_report(clip_has_rgb_only=True)
    basalt = report["candidates"]["basalt"]
    assert basalt["status"] == "BLOCKED_MISSING_INPUT"
    assert "BSD-3-Clause" in basalt["license"]
    assert report["sensor_inputs"]["imu"] is False
    assert report["sensor_inputs"]["camera_imu_calibration"] is False
    assert any("IMU" in blocker for blocker in basalt["blockers"])
