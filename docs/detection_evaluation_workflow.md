# LibreYOLO9t ground-truth evaluation workflow

The prepared set at `evaluation/clip_01_libreyolo9t/` contains 27 images
uniformly spaced through `clips/clip_01.mp4`. Its `annotations.json` is an
**unreviewed template**, not a label set and not model output.

## Human annotation

For each frame, retain its identity fields and use pixel `xyxy` boxes with a
top-left origin, inclusive minima, and exclusive maxima:

```json
{
  "review_status": "reviewed",
  "objects": [{"class_id": 0, "bbox_xyxy": [12, 20, 45, 90]}]
}
```

Use only IDs from the embedded `coco-2017@80` class catalog. To record a
human-reviewed frame with no supported objects, use `"review_status":
"reviewed_empty"` and an empty `objects` list. Leave uncertain frames as
`"unreviewed"`, or use `"skip"`; neither is evaluated.

## Evaluation

Run the reviewed-only threshold sweep (0.10, 0.25, 0.50, IoU 0.50):

```powershell
.venv\Scripts\python.exe tools\evaluate_libreyolo.py
```

The resulting `evaluation_report.json` distinguishes evaluated and skipped
frames, and leaves recall undefined when there are no human ground-truth
positives. It does not calculate AP/mAP from this small, one-threshold matching
workflow. Per-class metrics are included only when a class has at least five
ground-truth instances.

## Resource benchmark

```powershell
.venv\Scripts\python.exe tools\benchmark_libreyolo.py
```

This is sequential CPU-only inference with batch size 1. It reports warm-up
separately from steady-state p50/p95 and the current process's Windows peak
working set. It is neither a real-time claim nor a GPU estimate.
