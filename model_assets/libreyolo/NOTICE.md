# LibreYOLO temporary detector assets

This directory holds non-source, locally downloaded detector assets.  They
must not be packaged or redistributed with P.R.I.S.M. without a separate
distribution review.

## Selected asset

- Model identifier: `LibreYOLO/LibreYOLO9t`, file `LibreYOLO9t.pt`
- Model card: <https://huggingface.co/LibreYOLO/LibreYOLO9t>
- Model-card license: MIT
- Source implementation: <https://github.com/LibreYOLO/libreyolo>
- Source implementation license: MIT
- Installed implementation version: `libreyolo==1.6.0`

The model card describes this checkpoint as a converted YOLOv9-T checkpoint
from the MultimediaTechLab/YOLO v1.0-alpha release.  Its stated license and
provenance, rather than any general-purpose detector claim, are the basis for
this temporary local use.  The downloaded file's SHA-256 is recorded after
download in `docs/libreyolo9t_provenance.md`.

P.R.I.S.M. uses this asset only to create typed 2-D object-detection evidence.
It does not infer depth, terrain, ego-motion, traversability, safety, or
vehicle actions from the detector.
