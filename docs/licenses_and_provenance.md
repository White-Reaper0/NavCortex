# Licenses and provenance

Evidence date: 2026-10-24 for the code licenses; the pretrained-weight rows in
section 2 were added on 2026-10-25 from the files the installer put on disk. Code
licenses were re-verified by fetching the upstream `LICENSE` files over HTTPS and
quoting their first lines. Weight and dataset terms are recorded separately
because they are usually *not* covered by the code license. Anything not verified
is marked **UNVERIFIED** rather than assumed.

## 1. Code licenses

| Component | Local artifact/version | License | Evidence (upstream file, verified 2026-10-24) | Redistribution |
| --- | --- | --- | --- | --- |
| OpenCV | `opencv-python 5.0.0.93` (`.venv`) | Apache-2.0 | PyPI wheel metadata; upstream OpenCV license | Allowed with notices. |
| NumPy | `1.26.4` | BSD-3-Clause | Installed wheel metadata | Allowed. |
| SciPy | `1.15.2` | BSD-3-Clause | Installed wheel metadata | Allowed. |
| PyTorch | `2.4.1+cpu` | BSD-3-Clause | Installed wheel metadata | Allowed. |
| TorchVision | `0.19.1+cpu` | BSD-3-Clause | Installed wheel metadata | Allowed. |
| Transformers | `4.57.6` (`requirements.txt`: `transformers>=4.40,<5`; 5.x disables its PyTorch backend below torch 2.6) | Apache-2.0 | Installed wheel metadata: `License: Apache 2.0 License`, classifier `License :: OSI Approved :: Apache Software License` | Allowed. Used only to load local weights with `local_files_only=True`. |
| huggingface_hub | `0.36.2` | Apache-2.0 | Installed wheel metadata: `License: Apache`, OSI classifier | Allowed. Used only by `tools/download_prism_pretrained.py`, which is the only code path in the repository that reaches the network. |
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
| `model_assets/pretrained/depth_anything_v2_small_hf/` (`depth-anything/Depth-Anything-V2-Small-hf`, revision `5426e4f0f36572d16453bbda7a8389317b1bef99`, `model.safetensors` 99,173,660 bytes, SHA-256 `3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70`) | Model card metadata in the downloaded `README.md` declares `license: apache-2.0` (permissive, attribution and NOTICE retention). | Local-only. Gitignored (`model_assets/pretrained/**/*.safetensors`), never redistributed, never re-downloaded by the viewer. |
| `model_assets/pretrained/segformer_b0_ade_512_512/` (`nvidia/segformer-b0-finetuned-ade-512-512`, revision `489d5cd81a0b59fab9b7ea758d3548ebe99677da`, `model.safetensors` 15,036,944 bytes, SHA-256 `6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9`) | The downloaded card declares `license: other` and its License section says only: "The license for this model can be found [here](https://github.com/NVlabs/SegFormer/blob/master/LICENSE)" - i.e. the **NVIDIA Source Code License for SegFormer**, which is non-commercial. The full upstream text was **not** pulled into the repository, so the terms are recorded at link level. | **Terms recorded, full text UNVERIFIED locally.** Local research/evaluation use only; gitignored; not redistributed; no product or service dependency. Re-verify the upstream LICENSE before any distribution or commercial use, and note that the ADE20K label space it was fine-tuned on carries its own terms. |
| `~/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth` (`torchvision:mobilenet_v3_small`, ImageNet1K_V1, 10,306,551 bytes, SHA-256 `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`) | Distributed with PyTorch/torchvision from `download.pytorch.org/models/`; ImageNet classification backbone weights only. | In the shared TorchHub cache, outside the repository, so nothing is committed. Provides **no** trained P.R.I.S.M. terrain prediction; it is only a possible initialisation for the custom 17-channel terrain head, which stays untrained. |
| `model_assets/pretrained/manifest.json` | Generated by `tools/download_prism_pretrained.py`: model ids, source URLs, pinned revisions, sizes, SHA-256, license metadata, per-asset load verification. | Provenance text, not a weight: deliberately outside the gitignore rule so the checkouts stay auditable. |

## 3. Rules applied

1. No Ultralytics package and no AGPL model weights are installed or used.
2. No YOLOP source code is copied into PRISM. Only the SHA-pinned ONNX is run
   through OpenCV DNN, and the file is gitignored.
3. COLMAP is invoked/read locally only. No binary is vendored and no model or
   binary is redistributed.
4. Third-party model output never carries a dataset license claim: taxonomy
   identifiers name the label space, not a shipped dataset.
5. New PRISM code in this milestone (`provider_contract.py`,
   `yolop_provider.py`, `terrain_grid.py`, the evaluation tool,
   `run_prism_ipwebcam.py`, `run_prism_live_viewer.py`, tests, documentation) is
   original work. The live workbench adds no new third-party dependency: it uses
   the already-installed OpenCV window for display and the optional Open3D
   viewer, reads only the local LibreYOLO checkpoint, and shells out to the
   already-present local COLMAP binary without copying or redistributing it.
6. The pretrained depth/segmentation weights are **local artifacts only**:
   `tools/download_prism_pretrained.py` fetches them into
   `model_assets/pretrained/`, the gitignore rules keep every
   `.safetensors`/`.bin`/`.pth`/`.onnx` out of Git, and no entry point except the
   installer can download anything. The viewer and the training-label builder
   load through `prism/models/pretrained_paths.py` with
   `local_files_only=True`, and the headless verification run worked with every
   socket in the process blocked.
7. Pretrained output is never presented as P.R.I.S.M.-trained capability: the
   depth view is relative depth on an assumed scale, the segmentation view is
   ADE20K semantics mapped to prism-7, and the viewer never runs a learned
   traversability model. This milestone created no checkpoint; the wording on
   screen and in the report is derived from real file state, so it reads
   `NOT TRAINED (trav_unc_shared.pt does not exist)` while the file is absent
   and `NOT USED BY THIS VIEWER` once the separate manual training task has
   produced it.

## 4. Outstanding licensing actions

* Obtain an explicit weights license for the YOLOP ONNX before any distribution.
* Do not distribute the TUM archives (xyz, rpy, 360) or extracted frames.
* If YOLOP is ever promoted beyond local evaluation, replace or isolate the
  YOLOv5-derived helper path, or obtain clarification from upstream.
* Before any distribution, product use or commercial use of the segmentation
  view, read the NVIDIA Source Code License linked from
  `model_assets/pretrained/segformer_b0_ade_512_512/README.md` (its card only
  links it, and that license is non-commercial); if it does not fit, swap in a
  permissively licensed ADE20K segmentation model or train the project's own
  PIDNet head. The Depth Anything weights are Apache-2.0 and only require
  attribution/NOTICE retention.
* The downloaded weight directories must stay out of Git: re-check
  `git status`/`git check-ignore` before any commit that touches
  `model_assets/`.
