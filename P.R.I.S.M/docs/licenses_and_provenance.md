# Licenses and provenance

Evidence date: 2026-10-24. Code licenses were re-verified by fetching the
upstream `LICENSE` files over HTTPS and quoting their first lines. Weight and
dataset terms are recorded separately because they are usually *not* covered by
the code license. Anything not verified is marked **UNVERIFIED** rather than
assumed.

## 1. Code licenses

| Component | Local artifact/version | License | Evidence (upstream file, verified 2026-10-24) | Redistribution |
| --- | --- | --- | --- | --- |
| OpenCV | `opencv-python 4.11.0.86` (`.venv`) | Apache-2.0 | PyPI wheel metadata; upstream OpenCV license | Allowed with notices. |
| NumPy | `1.26.4` | BSD-3-Clause | Installed wheel metadata | Allowed. |
| SciPy | `1.15.2` | BSD-3-Clause | Installed wheel metadata | Allowed. |
| PyTorch | `2.4.1+cpu` | BSD-3-Clause | Installed wheel metadata | Allowed. |
| psutil | `7.2.2` | BSD-3-Clause | Installed wheel metadata | Allowed. |
| Open3D | `0.20.0` | MIT | Installed wheel metadata | Allowed. |
| Norfair | `2.3.0` | BSD-3-Clause | Upstream `tryolabs/norfair` license | Allowed. |
| LibreYOLO | `libreyolo 1.6.0` | MIT (implementation); model card terms recorded locally | `docs/libreyolo9t_provenance.md` | Checkpoint hash `b4d7e93f9e0393830fb42e6135c0e3464b2673b05e5ecf4b7f2374ec18e39eb2`. |
| YOLOP | local ONNX only | MIT repository | `https://raw.githubusercontent.com/hustvl/YOLOP/main/LICENSE` → "MIT License / Copyright (c) 2021 Hust Visual Learning Team" | Local evaluation only; see chain-of-title caveat below. |
| Basalt | not downloaded | BSD-3-Clause | `https://raw.githubusercontent.com/VladyslavUsenko/basalt/master/LICENSE` → "BSD 3-Clause License / Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel." | Not redistributed. |
| RTAB-Map | not downloaded | BSD-3-Clause | `https://raw.githubusercontent.com/introlab/rtabmap/master/LICENSE` → BSD text, "Copyright (c) 2010-2025, Mathieu Labbe - IntRoLab" | Not redistributed. |
| stella_vslam | not downloaded | BSD-2-Clause | Recorded in the pre-existing integration manifest | Not redistributed. |
| grid_map | concept only | BSD-3-Clause | `https://raw.githubusercontent.com/ANYbotics/grid_map/master/LICENSE` → "Copyright 2019, ANYbotics AG. / Redistribution and use in source and binary forms..." | No source copied; PRISM implementation is original. |
| elevation_mapping | concept only | BSD-3-Clause | Pre-existing integration manifest (not re-fetched this session) | No source copied. |
| traversability_estimation | concept only | BSD-3-Clause | Pre-existing integration manifest (not re-fetched this session) | No source copied. |
| COLMAP | local `tools/external/colmap/4.2.1-nocuda` | BSD-3-Clause upstream | Pre-existing integration manifest | **Local-only.** Not vendored, copied, or redistributed; bundled third-party components carry their own terms. |
| TUM RGB-D benchmark data | `output/datasets/tum_rgbd/` (xyz, rpy, 360) | **Recorded, not independently re-verified** | Adapter provenance records "Dataset data: Creative Commons 4.0 Attribution (CC BY 4.0); accompanying source code: BSD-2-Clause"; not re-fetched from the TUM site this milestone | Local benchmark use only; attribution required; do not redistribute. |

## 2. Weight and dataset terms (separate from code)

| Asset | Terms / provenance | Status |
| --- | --- | --- |
| `tools/external/yolop/yolop-320-320.onnx` | Downloaded from the upstream YOLOP weights location; SHA-256 `86d6e8b6dfdef195c061e9bcad82d9487bb5ee1ac4a1cf9a3dc4736657a07369`; 31,817,335 bytes. No separate weights license file was found. | **UNVERIFIED for weights**; local evaluation only, gitignored, not redistributed. |
| YOLOP training data (BDD100K) | Dataset license: BSD-3-Clause (`https://raw.githubusercontent.com/bdd100k/bdd100k/master/LICENSE` → "BSD 3-Clause License / Copyright (c) 2018, Fisher Yu"). Dataset itself is not shipped. | Verified for dataset code/terms; no images redistributed. |
| YOLOP YOLOv5-derived helpers | Upstream README lists `YOLOv5s` among reference works (`https://github.com/ultralytics/yolov5`), and YOLOP helper modules follow YOLOv5 structure (Ultralytics code is AGPL-3.0). The repository's MIT declaration may not cover those derived files. | **Chain-of-title UNVERIFIED.** Consequence: no YOLOP source is vendored; only the local ONNX is executed, and the model stays optional. |
| LibreYOLO9t checkpoint | `model_assets/libreyolo/LibreYOLO9t.pt`, SHA-256 `b4d7e93f9e0393830fb42e6135c0e3464b2673b05e5ecf4b7f2374ec18e39eb2`; local provenance document claims MIT implementation/model card. | Verified only as far as the local provenance document; no AGPL weights used. |
| COCO taxonomy string | `coco-2017@80` is a class-name list, not the dataset. | No dataset shipped. |
| BDD100K taxonomy string | `bdd100k-vehicle@1` is a single vehicle class derived from YOLOP's output layout. | No dataset shipped. |
| TUM ground truth / depth | Benchmark reference inputs, used locally for validation. Adapter records CC BY 4.0 for data. | Not redistributed; attribution required. |

## 3. Rules applied

1. No Ultralytics package and no AGPL model weights are installed or used.
2. No YOLOP source code is copied into PRISM. Only the SHA-pinned ONNX is run
   through OpenCV DNN, and the file is gitignored.
3. COLMAP is invoked/read locally only. No binary is vendored and no model or
   binary is redistributed.
4. Third-party model output never carries a dataset license claim: taxonomy
   identifiers name the label space, not a shipped dataset.
5. New PRISM code in this milestone (`provider_contract.py`,
   `yolop_provider.py`, `terrain_grid.py`, the evaluation tool, tests,
   documentation) is original work.

## 4. Outstanding licensing actions

* Obtain an explicit weights license for the YOLOP ONNX before any distribution.
* Do not distribute the TUM archives (xyz, rpy, 360) or extracted frames.
* If YOLOP is ever promoted beyond local evaluation, replace or isolate the
  YOLOv5-derived helper path, or obtain clarification from upstream.
