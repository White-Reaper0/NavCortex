"""Export and QA an existing COLMAP sparse model without rerunning COLMAP."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prism.colmap_inspection import build_qa_report, read_sparse_model, trajectory_rows, write_ply


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="output/09_colmap_sparse/clip_01_repro/sparse/0")
    parser.add_argument("--manifest", default="output/09_colmap_sparse/clip_01_repro/frames_manifest.json")
    parser.add_argument("--output", default="output/09_colmap_sparse/clip_01_repro/inspection")
    args = parser.parse_args()
    model_path, manifest_path, output = Path(args.model), Path(args.manifest), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty inspection directory: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = read_sparse_model(model_path)
    report = build_qa_report(model, manifest)
    output.mkdir(parents=True, exist_ok=False)
    write_ply(model.points, output / "sparse_points.ply")
    rows = trajectory_rows(model, manifest)
    with (output / "camera_trajectory.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("image_id", "file", "source_frame_index", "source_time_s", "q_w", "q_x", "q_y", "q_z",
                         "t_x_world_to_camera", "t_y_world_to_camera", "t_z_world_to_camera",
                         "center_x_arbitrary", "center_y_arbitrary", "center_z_arbitrary"))
        for row in rows:
            writer.writerow((row["image_id"], row["file"], row["source_frame_index"], row["source_time_s"],
                             *row["qvec_wxyz"], *row["tvec_world_to_camera"], *row["camera_center_arbitrary_units"]))
    report["exports"] = {"ply": "sparse_points.ply", "trajectory_csv": "camera_trajectory.csv",
                         "visualization": "Open sparse_points.ply in MeshLab or COLMAP GUI; both are local desktop viewers."
                         }
    (output / "qa_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "registered_images": report["registered_images"],
                      "points3D": report["points3D"], "issues": report["issues"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
