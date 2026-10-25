"""Monocular depth: Depth Anything V2-Small + metric alignment against the ground plane.

The network (Yang et al., NeurIPS 2024) predicts affine-invariant *relative inverse
depth* q. It is not a measurement in metres. `perception.geometry.fit_metric_ground`
solves 1/D = a*q + b jointly with the ground plane, using the assumed camera height as
the single metric anchor. Everything metric downstream inherits that assumption.
"""
from __future__ import annotations
import time
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from ..config import CFG, PROC_W, PROC_H, DEPTH_INPUT
from ..types import FramePacket, DepthResult
from ..io_utils import ego_mask, device as pick_device
from ..perception.geometry import fit_metric_ground, depth_from_q, GroundFit
from ..perception.fit_validity import FitValidityTracker

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class DepthStage:
    """Fills `packet.depth` with a metric-aligned `DepthResult`."""

    def __init__(self, device: Optional[str] = None, fp16: bool = True,
                 input_size: int = DEPTH_INPUT, temporal_alpha: float = 0.35):
        from transformers import AutoModelForDepthEstimation
        self.device = device or pick_device()
        self.fp16 = bool(fp16) and self.device == "cuda"
        self.input_size = int(input_size)
        self.temporal_alpha = float(temporal_alpha)
        self.model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
        self.model.eval().to(self.device)
        if self.fp16:
            self.model.half()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self._prev_fit: Optional[GroundFit] = None
        self._prev_q: Optional[np.ndarray] = None
        self.last_ms = 0.0
        self.fit_validity = FitValidityTracker()
        self.last_fit: Optional[GroundFit] = None
        self.last_validity = None

    def reset(self) -> None:
        self._prev_fit = None
        self._prev_q = None
        self.fit_validity.reset()

    # ------------------------------------------------------------------ core
    @torch.no_grad()
    def infer_q(self, bgr: np.ndarray) -> np.ndarray:
        """Raw relative inverse depth at PROC resolution, normalised to [0,1]."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        s = self.input_size
        img = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        t = torch.from_numpy(img.transpose(2, 0, 1))[None].to(self.device)
        if self.fp16:
            t = t.half()
        out = self.model(pixel_values=t).predicted_depth      # (1,h,w), larger = nearer
        out = F.interpolate(out[:, None].float(), size=(PROC_H, PROC_W),
                            mode="bicubic", align_corners=False)[0, 0]
        q = out.detach().cpu().numpy().astype(np.float32)
        lo, hi = np.percentile(q, 0.5), np.percentile(q, 99.5)
        return np.clip((q - lo) / max(hi - lo, 1e-6), 0.0, 1.2)

    def __call__(self, packet: FramePacket, seg_label: Optional[np.ndarray] = None) -> FramePacket:
        t0 = time.perf_counter()
        bgr = packet.rgb
        q = self.infer_q(bgr)

        # light temporal smoothing: the source is 30 fps handheld-style footage and
        # per-frame relative depth flickers, which would shake the metric fit
        if self._prev_q is not None and self.temporal_alpha > 0:
            q = (1 - self.temporal_alpha) * q + self.temporal_alpha * self._prev_q
        self._prev_q = q

        em = ego_mask(*q.shape)
        base_valid = em & np.isfinite(q)
        if seg_label is None and packet.seg is not None:
            seg_label = packet.seg.label
        sl = None
        if seg_label is not None and seg_label.shape == q.shape:
            sl = seg_label
            base_valid &= (seg_label != 0)          # sky carries no usable geometry

        fit = fit_metric_ground(q, base_valid, sl, prior=self._prev_fit)
        fit_ok = bool(fit.ok)
        if not fit.ok:
            fit = self._prev_fit if (self._prev_fit is not None and self._prev_fit.ok) else GroundFit(
                a=1.0, b=0.05, normal=np.array([0.0, 1.0, 0.0], np.float32),
                height=CFG.cam.height_above_ground_m, ok=False)
        else:
            self._prev_fit = fit

        depth, valid = depth_from_q(q, fit)
        valid &= em
        if sl is not None:
            valid &= (sl != 0)
        # where the ground fit is unreliable the region is invalid -> UNKNOWN downstream
        fv = self.fit_validity(q, base_valid, sl, fit, fit_ok)
        valid &= fv.valid
        self.last_fit, self.last_validity = fit, fv

        packet.depth = DepthResult(rel_inv=q, depth_m=depth, scale=fit.a, shift=fit.b,
                                   align_residual=fit.residual, align_inliers=fit.inliers,
                                   valid=valid)
        self.last_ms = (time.perf_counter() - t0) * 1e3
        packet.timings_ms["depth"] = self.last_ms
        return packet


def _self_test() -> None:
    from ..io_utils import read_frames
    from ..config import CLIP_IDS
    st = DepthStage()
    print(f"Depth Anything V2-Small: {st.n_params/1e6:.2f} M params, device={st.device}, fp16={st.fp16}")
    for cid in CLIP_IDS[:2]:
        st.reset()
        ms, stats = [], []
        for i, f in read_frames(cid, max_frames=12):
            p = st(FramePacket(clip_id=cid, idx=i, t=i / 30.0, rgb=f))
            d, v = p.depth.depth_m, p.depth.valid
            ms.append(p.timings_ms["depth"])
            stats.append((float(np.nanpercentile(d[v], 5)) if v.any() else np.nan,
                          float(np.nanmedian(d[v])) if v.any() else np.nan,
                          float(np.nanpercentile(d[v], 95)) if v.any() else np.nan,
                          p.depth.scale, p.depth.shift, p.depth.align_residual,
                          p.depth.align_inliers, float(v.mean())))
        a = np.array(stats, np.float64)
        print(f"{cid}: {np.mean(ms):5.1f} ms/frame | depth p5/med/p95 = "
              f"{np.nanmean(a[:,0]):.2f}/{np.nanmean(a[:,1]):.2f}/{np.nanmean(a[:,2]):.2f} m | "
              f"a={np.nanmean(a[:,3]):.3f} b={np.nanmean(a[:,4]):.3f} "
              f"resid={np.nanmean(a[:,5]):.4f} 1/m inl={a[:,6].mean():.0f} valid={a[:,7].mean()*100:.0f}%")


if __name__ == "__main__":
    _self_test()
