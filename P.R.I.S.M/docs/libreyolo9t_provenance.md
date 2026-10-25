# LibreYOLO9t local provenance record

This record covers a temporary local detector asset, not P.R.I.S.M. source.

| Field | Verified value |
| --- | --- |
| Implementation | LibreYOLO 1.6.0; [MIT source](https://github.com/LibreYOLO/libreyolo) |
| Model | [LibreYOLO/LibreYOLO9t](https://huggingface.co/LibreYOLO/LibreYOLO9t), model-card license MIT |
| Local file | `model_assets/libreyolo/LibreYOLO9t.pt` |
| Byte length | 8,645,036 |
| SHA-256 | `b4d7e93f9e0393830fb42e6135c0e3464b2673b05e5ecf4b7f2374ec18e39eb2` |
| Backend | CPU-only PyTorch 2.4.1+cpu |

The model card identifies the checkpoint as a YOLOv9-T conversion from the
MultimediaTechLab/YOLO v1.0-alpha release and says the conversion remaps only
state-dict keys. Reconfirm license and provenance before redistribution.

An earlier same-size local file with SHA-256
`2ca45b06295750170d3bbab0bc0622a2dca0b9653d56bbca54bd5eaa8dc6c81e`
failed LibreYOLO/PyTorch loading and did not match the official endpoint's
linked ETag. It is retained locally with an `.invalid-...` filename for audit,
not used by the provider.
