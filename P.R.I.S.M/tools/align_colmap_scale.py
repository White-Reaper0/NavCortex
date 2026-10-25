"""Create an explicit COLMAP metric-scale status report and optional scaled copies.

This never runs COLMAP or alters the native sparse model.  A reference must be
human-reviewed and match two existing point IDs before exports are written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prism.calibration import load_calibration_artifact
from prism.colmap_inspection import read_sparse_model
from prism.scale_alignment import (ScaleAlignmentError, aligned_report, load_scale_reference,
                                   verify_reference_against_model, write_scaled_exports)


def calibration_summary(path: str | None) -> dict:
    if path is None:
        return {"status": "MISSING", "note": "no checkerboard calibration artifact was supplied"}
    artifact = load_calibration_artifact(path)
    return {"status": "CHECKERBOARD_VALIDATED" if artifact.valid else "CHECKERBOARD_INVALID",
            "artifact": str(path), "calibration_id": artifact.calibration_id,
            "image_size": list(artifact.image_size), "rms_px": artifact.rms_px,
            "note": "checkerboard intrinsics do not independently establish monocular metric scale"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="output/09_colmap_sparse/clip_01_repro/sparse/0")
    parser.add_argument("--manifest", default="output/09_colmap_sparse/clip_01_repro/frames_manifest.json")
    parser.add_argument("--reference", help="reviewed prism.colmap_scale_reference/v1 JSON")
    parser.add_argument("--calibration", help="optional validated prism checkerboard calibration JSON")
    parser.add_argument("--output", default="output/09_colmap_sparse/clip_01_repro/scale_alignment")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output}")
    calibration = calibration_summary(args.calibration)
    reference = load_scale_reference(args.reference) if args.reference else None
    model = None
    manifest = None
    if reference is not None and reference.reviewed:
        # Validate the binding before creating any result directory.  This
        # avoids leaving partial "scaled" outputs behind after a bad reference.
        model = read_sparse_model(args.model)
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        verify_reference_against_model(reference, model)
    report = aligned_report(reference, calibration=calibration)
    if reference is not None and reference.reviewed:
        assert model is not None and manifest is not None
        report["exports"] = write_scaled_exports(model, manifest, output, reference)
    else:
        output.mkdir(parents=True, exist_ok=False)
        report["exports"] = None
    (output / "scale_alignment_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"], "exports": report["exports"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScaleAlignmentError as error:
        raise SystemExit(f"scale alignment failed: {error}")
