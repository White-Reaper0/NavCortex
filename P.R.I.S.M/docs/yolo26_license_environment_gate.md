# YOLO26 license and environment gate

Status: **BLOCKED — no download or installation authorized by this gate.**

## Intended first model

- Implementation: official Ultralytics YOLO26 Python implementation.
- Model identifier: `yolo26n.pt` (YOLO26 nano object-detection variant).
- Intended source: the official Ultralytics YOLO26 release path resolved by the
  `ultralytics` package for that identifier; model documentation:
  <https://docs.ultralytics.com/models/yolo26>.
- Planned project-local asset path after approval and download:
  `model_assets/ultralytics/yolo26n.pt`.
- The exact resolved release URL and SHA-256 are intentionally unknown until a
  permitted download occurs; they must be recorded and pinned at that time.

## License gate

The P.R.I.S.M. repository's `LICENSE` is MIT and explicitly covers only this
repository's source. It does not license third-party implementation code or
weights.

Ultralytics states that YOLO code and models are available under AGPL-3.0 or an
Enterprise license:

- <https://docs.ultralytics.com/models/yolo26>
- <https://www.ultralytics.com/license>
- <https://github.com/ultralytics/ultralytics>

Ultralytics' license guidance says that use of its code, architectures,
training pipelines, or trained/fine-tuned models requires either AGPL-3.0
compliance for the complete relevant work or an Enterprise license. Therefore
the official `yolo26n.pt` weight must be treated as AGPL-3.0 by default, not as
covered by the P.R.I.S.M. MIT license.

The intended P.R.I.S.M. distribution and its willingness to comply with AGPL-3.0
have not been established. Enterprise licensing would be a paid option, which
is outside the current no-paid-services constraint. The gate remains blocked
until the project owner explicitly confirms an acceptable AGPL-3.0 distribution
strategy or supplies a separately approved licensing basis.

## Verified local environment at gate time

- Project-local Python 3.11 virtual environment exists at `.venv`.
- Installed: `opencv-python 4.11.0.86`, `numpy 2.2.6`, and `scipy 1.15.2`.
- Not installed: `ultralytics`, `torch`, `torchvision`, `transformers`,
  `onnxruntime`, and `openvino`.
- No project-local model weights or inference assets were found. In particular,
  no `checkpoints/`, `weights/`, or `models/` directory exists.
- No OpenVINS references were found in the project outside the virtual
  environment.

## Consequence

No package installation, model download, inference-provider implementation,
or performance benchmark may proceed under the official Ultralytics route until
this gate passes. The typed detection-evidence contract remains usable for
fixture-only tests and for a future separately licensed provider.
