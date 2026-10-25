"""Extract evenly spaced, human-review-only object-detection evaluation frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prism.detection_evaluation import (AnnotationFrame, COCO_2017_80_TAXONOMY,
                                        ReviewStatus, write_annotation_template)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="clips/clip_01.mp4")
    parser.add_argument("--output", default="evaluation/clip_01_libreyolo9t")
    parser.add_argument("--frames", type=int, default=27)
    args = parser.parse_args()
    if not 24 <= args.frames <= 30:
        raise SystemExit("--frames must be between 24 and 30")
    source = Path(args.video)
    output = Path(args.output)
    if not source.is_file():
        raise SystemExit(f"video not found: {source}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "frames"
    image_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total <= 0:
        cap.release()
        raise SystemExit("video has no positive FPS or frame count")
    selected = tuple(sorted({round(index * (total - 1) / (args.frames - 1)) for index in range(args.frames)}))
    frames: list[AnnotationFrame] = []
    wanted = iter(selected)
    next_index = next(wanted, None)
    frame_index = 0
    try:
        while next_index is not None:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise RuntimeError(f"decode failed at frame {frame_index}")
            if frame_index == next_index:
                height, width = bgr.shape[:2]
                filename = f"frame_{frame_index:06d}.png"
                if not cv2.imwrite(str(image_dir / filename), bgr):
                    raise RuntimeError(f"could not write {filename}")
                frames.append(AnnotationFrame(
                    frame_id=f"{source.stem}:{frame_index:06d}", source_video=str(source),
                    frame_index=frame_index, capture_at_ns=round(frame_index * 1_000_000_000 / fps),
                    width=width, height=height, review_status=ReviewStatus.UNREVIEWED,
                    image_path=str(Path("frames") / filename)))
                next_index = next(wanted, None)
            frame_index += 1
    finally:
        cap.release()
    manifest = {"schema_version": "prism-detection-frame-manifest/v1", "source_video": str(source),
                "fps": fps, "declared_frame_count": total, "taxonomy": COCO_2017_80_TAXONOMY,
                "coordinate_convention": "pixel_xyxy_exclusive",
                "frames": [{"frame_id": item.frame_id, "frame_index": item.frame_index,
                            "capture_at_ns": item.capture_at_ns, "width": item.width,
                            "height": item.height, "image_path": item.image_path} for item in frames]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_annotation_template(output / "annotations.json", source_video=str(source),
                              taxonomy=COCO_2017_80_TAXONOMY, frames=frames)
    print(json.dumps({"output": str(output), "frames_extracted": len(frames), "fps": fps,
                      "selected_frame_indices": selected}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
