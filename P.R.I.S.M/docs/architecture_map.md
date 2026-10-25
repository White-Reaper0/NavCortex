# Architecture map

This map describes the boundaries that exist in code, not an aspirational
design. The goal of the reuse milestone is to keep optional third-party work
outside these contracts.

## 1. Boundary diagram

```
 recorded video / benchmark files
            │
            ▼
 ┌──────────────────────────────┐
 │ Frame & sensor layer         │  prism/video_observation_replay.py
 │ DecodedFrame(frame_index,    │  prism/benchmarks/tum_rgbd_adapter.py
 │  capture_at_ns, w, h, bgr)   │  (pixels + capture order + file provenance)
 └──────────────┬───────────────┘
                │  no depth/pose/terrain is inferred by decoding
                ▼
 ┌──────────────────────────────┐
 │ Perception providers         │  prism/libreyolo_provider.py
 │ FrameObservationProvider     │  prism/integrations/yolop_provider.py
 │  (config, __call__)          │  prism/integrations/tracking.py
 └──────────────┬───────────────┘
                │  Observation + hypothesis_id + relation + expires_at_ns
                ▼
 ┌──────────────────────────────┐
 │ Motion / localization        │  prism/visual_odometry_replay.py (relative)
 │                              │  prism/integrations/localization.py (status)
 │                              │  prism/integrations/geometry.py (COLMAP QA)
 └──────────────┬───────────────┘
                │  relative motion / sparse diagnostics; no metric claim
                ▼
 ┌──────────────────────────────┐
 │ Spatial / terrain            │  prism/perception/mapping.py → BEVMap
 │                              │  prism/integrations/terrain_grid.py (layers)
 │                              │  prism/integrations/terrain.py (gate)
 └──────────────┬───────────────┘
                │  BEVMap / TerrainGrid; unknown stays UNKNOWN
                ▼
 ┌──────────────────────────────┐
 │ PRISM observation contract   │  prism/observations.py
 │ Observation, EvidencePayload │  prism/runtime_adapter.py → FramePacket + audit
 └──────────────┬───────────────┘
                │  typed, validated evidence (or explicit rejection)
                ▼
 ┌──────────────────────────────┐
 │ Reasoning engine (Aaquib's)  │  prism/belief_ledger.py
 │                              │  prism/doubt_engine.py
 │                              │  prism/observation_replay.py
 └──────────────┬───────────────┘
                │  advisory decisions only
                ▼
        report / advisory JSON
        (no vehicle command in replay)

 Live path (unchanged): prism/runtime.py PrismNavigator → nav/planner,
 nav/supervisor, nav/goal_planner → velocity command.
```

## 2. Contracts at each seam

| Seam | Type | Invariants |
| --- | --- | --- |
| Decode | `DecodedFrame` | Pixels, dimensions, frame index, capture timestamp. No perception fields. |
| Provider | `FrameObservationProvider` (`provider_contract.py`) | `config` + `__call__(DecodedFrame) -> VideoReplayInput`. No control output, no fabrication. |
| Observation | `Observation` + `EvidencePayload` | Typed fields with units; 2-D detections carry taxonomy, score semantics, image bounds. |
| Runtime adaptation | `RuntimeAdapter.to_packet` | Rejects observations that cannot safely become a packet (e.g. boxes without geometry); audit record retained. |
| Map | `BEVMap` | `height=NaN` and `trav=UNKNOWN` for unobserved/stale/low-confidence cells. |
| Terrain layers | `TerrainGrid` | Named layers indexed `[forward, lateral]`; NaN where unmeasured; no fabricated variance/normals. |
| Reasoning | `BeliefLedger`, `DoubtEngine` | Expiry and unavailable evidence are first-class; advisory does not command the vehicle. |

## 3. Where reuse enters

* **OpenCV** already provides decode, ORB, and the DNN ONNX runtime. It is the
  substrate for both detectors; no new runtime was added.
* **YOLOP** enters only through `YOLOPDetectionProvider`, which emits
  `ObjectDetectionEvidence`. Its segmentation heads are exposed by
  `segmentation_summary()` as descriptive statistics for the evaluation tool,
  never as observation evidence.
* **grid_map** enters as concepts: multilayered named grids and explicit
  unknown handling. Its C++/ROS implementation is not linked. The PRISM adapter
  reads a `BEVMap` and produces layers.
* **COLMAP** enters as a read-only sparse model via `colmap_inspection`; the
  metric-scale step is a separate, explicit artifact (`scale_alignment`).
* **Norfair** enters as an optional tracker; the deterministic IoU tracker is
  the fallback.

## 4. What third-party internals must not leak

* No YOLOP tensor, ONNX node name, or raw output reaches `Observation`.
  Only `ObjectDetection` values with taxonomy and image-space boxes.
* No COLMAP camera model, point ID, or native coordinate leaks into a metric
  field. `sparse_geometry_report` keeps native units and says so.
* No grid_map type crosses the boundary; `TerrainGrid` is a PRISM dataclass.
* No ROS message, IMU sample, or depth image is synthesised to satisfy a
  downstream signature.

## 5. Evidence categories kept separate

The validation and evaluation reports keep these eight categories distinct:

1. software-path checks,
2. trajectory metrics (ATE/RPE with stated frame counts and alignment),
3. geometry/scale status,
4. mapping-interface checks,
5. terrain/traversability status,
6. detector evidence and latency,
7. tracking/prediction records,
8. belief/doubt advisory.

A pass in one category is never promoted to a claim in another. In particular,
a passing mapping interface is not traversability accuracy, and a detection
count is not detection accuracy.

## 6. Extension points

| Need | Add here | Do not |
| --- | --- | --- |
| New detector | A `FrameObservationProvider` implementation + provenance | Add fields to `Observation` that imply depth/safety |
| New motion backend | A localization candidate entry with blockers + a replay adapter emitting relative poses | Call vision-only output "visual-inertial" |
| New terrain source | Layer producer feeding `TerrainGrid`, with explicit metric status | Fill unknown cells with safe/zero |
| New evaluation | A bounded tool writing a versioned JSON report | Overwrite non-empty outputs or compare across taxonomies |

## 7. Frozen modules

`prism/config.py`, `prism/types.py`, `prism/io_utils.py`, `prism/viz_common.py`,
`prism/perception/geometry.py` were not modified. Their modification times were
verified unchanged against the pre-milestone state.
