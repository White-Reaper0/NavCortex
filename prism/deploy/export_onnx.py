"""ONNX export for every prism network, with numerical verification on real frames.

Why this file exists
--------------------
The prism pitch is CPU-friendly deployment.  That claim is only worth anything if
there is an actual portable graph to run and if that graph provably computes the same
thing the PyTorch model computes.  So every export here is followed by a *verification*
pass: the same real clip frame is pushed through PyTorch and through ONNX Runtime and
the max absolute / relative deviation is recorded.  An export whose outputs do not
match is reported as **FAILED**, not quietly shipped.

Design notes
------------
* Each model is an independent :class:`ExportSpec`.  Checkpoints are produced by other
  agents in parallel, so a spec whose checkpoint is missing is *skipped with a clear
  message* and the script keeps going.  Re-running the script picks up whatever has
  appeared since; already-exported graphs are re-verified, and re-exported only when
  the checkpoint is newer than the .onnx (or ``--force``).
* Input shapes are fixed (batch 1, fixed spatial size).  Fixed shapes are what make
  ONNX Runtime's CPU kernels and the static INT8 quantizer behave predictably, and the
  prism pipeline only ever runs one frame at one resolution anyway.
* Verification inputs are **real clip frames**, never random tensors: quantisation and
  fused-kernel differences show up on real statistics and hide on Gaussian noise.

Sizes (all traceable to ``prism/config.py`` / the model modules, not guessed):
    depth_anything_v2_small   (1, 3, 518, 518)     DEPTH_INPUT
    pidnet_s_terrain          (1, 3, 288, 512)     SEG_INPUT = (w=512, h=288)
    trav_unc_shared           (1, 17, 180, 320)    traversability.NET_H/NET_W
    vpr_gem_mnv3              per models/vpr.py    (read at export time)
    world_model               (1, 8, 128, 192)     CFG.bev.H x CFG.bev.W, bev_utils.BEV_CH

Usage
-----
    python -m prism.deploy.export_onnx                 # export everything available
    python -m prism.deploy.export_onnx --only depth    # one model
    python -m prism.deploy.export_onnx --force         # re-export even if up to date
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..config import (CFG, CKPT_DIR, CLIP_IDS, DEPTH_INPUT, N_TERRAIN, PROC_H, PROC_W,
                      SEG_INPUT)
from ..io_utils import has_stage, load_stage, read_frames

ONNX_DIR = CKPT_DIR / "onnx"
OPSET = 17
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# Numerical tolerance for "this export matches PyTorch".  FP32 ONNX Runtime differs
# from PyTorch only by kernel fusion / accumulation order, which lands around 1e-5
# relative on these graphs; 1e-3 relative is generous but still catches a real bug
# (wrong weights, wrong preprocessing, a traced branch taken the wrong way).
REL_TOL = 1e-3
ABS_TOL = 1e-3


# ============================================================ hardware provenance

def _cpu_name() -> str:
    """Marketing name of the CPU (platform.processor() only gives the family on Windows)."""
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except Exception:
        pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _physical_cores() -> Optional[int]:
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores"
             " -Sum).Sum"],
            capture_output=True, text=True, timeout=25)
        v = out.stdout.strip()
        if v.isdigit():
            return int(v)
    except Exception:
        pass
    return None


_HW_CACHE: Optional[dict] = None


def hardware_info() -> dict:
    """Everything a reader needs to interpret a latency number in this repo."""
    global _HW_CACHE
    if _HW_CACHE is not None:
        return _HW_CACHE
    import onnxruntime as ort
    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "cpu": _cpu_name(),
        "cpu_physical_cores": _physical_cores(),
        "cpu_logical_processors": os.cpu_count(),
        "ram_gb": None,
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "onnxruntime_providers": list(ort.get_available_providers()),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_vram_gb": (round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
                        if torch.cuda.is_available() else None),
        "note": ("GPU is shared with other agents in this build; GPU numbers may carry "
                 "contention. CPU numbers were taken with the GPU idle."),
    }
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = _MS()
        ms.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        info["ram_gb"] = round(ms.ullTotalPhys / 2**30, 1)
    except Exception:
        pass
    _HW_CACHE = info
    return info


# ============================================================ real-frame providers

_FRAME_CACHE: dict[tuple, list] = {}
_LEN_CACHE: dict[str, int] = {}


def _clip_length(clip_id: str) -> int:
    if clip_id not in _LEN_CACHE:
        from ..io_utils import clip_path
        cap = cv2.VideoCapture(str(clip_path(clip_id)))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        _LEN_CACHE[clip_id] = n if n > 0 else 300
    return _LEN_CACHE[clip_id]


def sampled_frames(n_per_clip: int = 8, clip_ids: Optional[list[str]] = None
                   ) -> list[tuple[str, int, np.ndarray]]:
    """Evenly spaced real frames from every clip, at PROC resolution (BGR uint8).

    Both the daylight clips (01/02) and the low-light stress clips (04/05) are sampled,
    which matters for INT8 calibration: activation ranges differ substantially between
    them.
    """
    clip_ids = list(clip_ids or CLIP_IDS)
    key = (int(n_per_clip), tuple(clip_ids))
    if key in _FRAME_CACHE:
        return _FRAME_CACHE[key]
    out: list[tuple[str, int, np.ndarray]] = []
    for cid in clip_ids:
        n_total = _clip_length(cid)
        if n_total <= 0:
            continue
        want = set(np.linspace(0, n_total - 1,
                               min(n_per_clip, n_total)).astype(int).tolist())
        # decode once, keep only the sampled frames: a whole clip at PROC resolution is
        # ~200 MB and this is called for every model and every variant.
        for i, f in read_frames(cid):
            if i in want:
                out.append((cid, int(i), f.copy()))
    if len(_FRAME_CACHE) > 6:
        _FRAME_CACHE.clear()
    _FRAME_CACHE[key] = out
    return out


def _imagenet_chw(bgr: np.ndarray, size_wh: tuple[int, int],
                  interp: int = cv2.INTER_CUBIC) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, size_wh, interpolation=interp).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None])


def depth_input(bgr: np.ndarray) -> np.ndarray:
    """Exactly what ``models/depth.DepthStage.infer_q`` feeds the network."""
    return _imagenet_chw(bgr, (DEPTH_INPUT, DEPTH_INPUT), cv2.INTER_CUBIC)


def seg_input(bgr: np.ndarray) -> np.ndarray:
    """PIDNet-S student input at SEG_INPUT = 512x288, ImageNet normalised."""
    return _imagenet_chw(bgr, (SEG_INPUT[0], SEG_INPUT[1]), cv2.INTER_LINEAR)


# ---- the 17-channel traversability / uncertainty stack ---------------------------

#: Provenance of the arrays fed to a model's input builder, recorded so a reader can
#: tell a genuine cached perception input from a documented stand-in.  Cleared per model.
_TRAV_STACK_SOURCE: dict[str, str] = {}


def reset_input_source() -> None:
    _TRAV_STACK_SOURCE.clear()


def input_source() -> dict:
    return dict(_TRAV_STACK_SOURCE) or {"source": "real clip frame only"}


def trav_input(bgr: np.ndarray, clip_id: str = "", idx: int = -1) -> np.ndarray:
    """Build the documented 17-channel stack from **cached real perception outputs**.

    Depth comes from ``work/cache/<clip>/depth.npz`` (produced by the depth stage);
    terrain semantics come from ``seg.npz`` when the segmentation agent has written it.
    If the seg cache is absent the terrain channels fall back to a geometric prior
    (sky above the horizon, trail below) and that fact is recorded in the report as
    ``seg_source``, because it changes the honest reading of any INT8 calibration.
    """
    from ..models import traversability as T
    from ..perception import geometry as G

    h, w = PROC_H, PROC_W
    depth = np.full((h, w), 3.0, np.float32)
    valid = np.ones((h, w), bool)
    fit = None
    if clip_id and has_stage(clip_id, "depth"):
        z = load_stage(clip_id, "depth")
        j = int(np.clip(idx, 0, z["depth"].shape[0] - 1))
        depth = np.asarray(z["depth"][j], np.float32)
        valid = np.asarray(z["valid"][j], bool)
        fit = G.GroundFit(a=float(z["scale"][j]), b=float(z["shift"][j]),
                          normal=np.asarray(z["normal"][j], np.float32),
                          height=CFG.cam.height_above_ground_m, ok=True)
        _TRAV_STACK_SOURCE["depth"] = "cache"
    else:
        _TRAV_STACK_SOURCE["depth"] = "flat-prior fallback (no depth cache)"
    if fit is None:
        fit = G.GroundFit(a=1.0, b=0.0, normal=np.array([0.0, 1.0, 0.0], np.float32),
                          height=CFG.cam.height_above_ground_m, ok=True)

    hh, ss, rr, _ = T.geometry_from_depth(np.nan_to_num(depth, nan=0.0), valid, fit)

    if clip_id and has_stage(clip_id, "seg"):
        zs = load_stage(clip_id, "seg")
        j = int(np.clip(idx, 0, zs["label"].shape[0] - 1))
        lab = np.asarray(zs["label"][j], np.uint8)
        pmax = np.asarray(zs["prob_max"][j], np.float32) if "prob_max" in zs else \
            np.full((h, w), 0.8, np.float32)
        ent = np.asarray(zs["entropy"][j], np.float32) if "entropy" in zs else \
            np.full((h, w), 0.2, np.float32)
        _TRAV_STACK_SOURCE["seg"] = "cache"
    else:
        lab = np.ones((h, w), np.uint8)              # trail
        lab[:int(0.38 * h)] = 0                      # sky above a nominal horizon
        pmax = np.full((h, w), 0.7, np.float32)
        ent = np.full((h, w), 0.35, np.float32)
        _TRAV_STACK_SOURCE["seg"] = "geometric prior fallback (no seg cache)"

    stack = T.build_feature_stack(bgr, depth, valid, hh, ss, rr, lab, pmax, ent)
    return np.ascontiguousarray(T.resize_stack(stack)[None])


def world_model_input(bgr: np.ndarray, clip_id: str = "", idx: int = -1) -> np.ndarray:
    """(1, 8, 128, 192) BEV world state.

    Uses the real ``bev`` cache when the mapping agent has written it; otherwise a
    synthetic state from ``nav.bev_utils.synthetic_state`` (which is the same generator
    the world-model training uses for augmentation) - recorded as such.
    """
    from ..nav import bev_utils as bu
    st = bu.load_bev_states(clip_id) if clip_id else None
    if st is not None and len(st):
        j = int(np.clip(idx, 0, len(st) - 1))
        _TRAV_STACK_SOURCE["bev"] = "cache"
        return np.ascontiguousarray(np.asarray(st[j], np.float32)[None])
    _TRAV_STACK_SOURCE["bev"] = "synthetic_state (no bev cache)"
    kinds = ["clear", "wall", "wall_left", "low_conf", "kerb", "blind"]
    return np.ascontiguousarray(
        bu.synthetic_state(kinds[abs(idx) % len(kinds)],
                           rng=np.random.default_rng(abs(idx) + 1))[None].astype(np.float32))


# ============================================================ export specs

@dataclass
class ExportSpec:
    name: str
    desc: str
    provenance: str
    ckpt: Optional[Path]
    input_name: str
    input_shape: tuple
    output_names: list[str]
    build: Callable[[], nn.Module]
    make_input: Callable[..., np.ndarray]
    dtype: str = "float32"
    notes: str = ""

    @property
    def onnx_path(self) -> Path:
        return ONNX_DIR / f"{self.name}.onnx"

    def available(self) -> bool:
        return self.ckpt is None or self.ckpt.exists()


# ---- wrappers giving each network a single flat forward() suitable for tracing ----

class _DepthWrap(nn.Module):
    """Depth Anything V2-Small -> raw relative inverse depth at network resolution."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, pixel_values):
        out = self.m(pixel_values=pixel_values)
        d = out.predicted_depth if hasattr(out, "predicted_depth") else out[0]
        return d if d.dim() == 4 else d.unsqueeze(1)


class _TravUncWrap(nn.Module):
    """One trunk forward -> traversability logits + risk + 2-channel uncertainty.

    This is the graph the deployed pipeline actually wants: the trunk is shared, so
    exporting the two heads separately would double the measured CPU cost and
    misrepresent the design.  Outputs are upsampled to PROC (360x640) exactly as
    ``TraversabilityStage`` / ``UncertaintyStage`` do.
    """

    def __init__(self, trunk, trav_head, unc_head, out_size=(PROC_H, PROC_W)):
        super().__init__()
        self.trunk, self.trav_head, self.unc_head = trunk, trav_head, unc_head
        self.out_size = out_size

    def forward(self, x):
        import torch.nn.functional as F
        feat = self.trunk(x)
        logits, risk = self.trav_head(feat)
        raw = self.unc_head(feat)
        k = dict(size=self.out_size, mode="bilinear", align_corners=False)
        return (F.interpolate(logits, **k), F.interpolate(risk, **k),
                F.interpolate(raw, **k))


class _WorldModelWrap(nn.Module):
    """BEV state -> all-action rollout (the per-frame world-model cost in the pipeline)."""

    def __init__(self, model, horizon: int):
        super().__init__()
        self.m = model
        self.horizon = int(horizon)

    def forward(self, state):
        s0 = self.m.encode(state)
        _, occ, trav, risk = self.m.rollout_all_actions(s0, self.horizon)
        return occ, trav, risk


# ---- builders --------------------------------------------------------------------

def _build_depth() -> nn.Module:
    from transformers import AutoModelForDepthEstimation
    from ..models.pretrained_paths import (DEPTH_ENV_VAR, DEPTH_MODEL_ID,
                                           local_model_dir)
    local = local_model_dir(DEPTH_MODEL_ID, env_var=DEPTH_ENV_VAR)
    source = str(local) if local is not None else DEPTH_MODEL_ID
    m = AutoModelForDepthEstimation.from_pretrained(source, local_files_only=True)
    return _DepthWrap(m.eval()).eval()


def _build_pidnet() -> nn.Module:
    from ..models.seg_pidnet import PIDNetS
    net = PIDNetS(num_classes=N_TERRAIN, augment=False)
    sd = torch.load(CKPT_DIR / "pidnet_s_prism7.pt", map_location="cpu", weights_only=False)
    for key in ("model", "state_dict", "student", "net"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    # augment=False drops seghead_p / seghead_d; those are the only tolerated leftovers.
    hard = [k for k in missing if not k.startswith(("seghead_p", "seghead_d"))]
    if hard:
        raise RuntimeError(f"pidnet checkpoint is missing real weights: {hard[:6]}")
    return net.eval()


def _build_trav_unc() -> nn.Module:
    from ..models.traversability import SharedPerceptionTrunk, TravHead
    from ..models.uncertainty import UncHead
    sd = torch.load(CKPT_DIR / "trav_unc_shared.pt", map_location="cpu", weights_only=False)
    trunk = SharedPerceptionTrunk(pretrained=False)
    trunk.load_state_dict(sd["trunk"])
    th = TravHead(trunk.out_ch)
    th.load_state_dict(sd["trav_head"])
    uh = UncHead(trunk.out_ch)
    uh.load_state_dict(sd["unc_head"])
    w = _TravUncWrap(trunk, th, uh)
    w.eval()
    # MC-dropout is a *runtime* option of the stage; the exported graph is the
    # deterministic single-pass model, so dropout must be off in the trace.
    for mod in w.modules():
        if isinstance(mod, (nn.Dropout, nn.Dropout2d)):
            mod.eval()
    return w


def _build_vpr() -> nn.Module:
    from ..models import vpr as V  # noqa: F401  (module written by the VPR agent)
    for fn in ("build_export_model", "build_model", "GeMNet", "VPRNet", "VPRModel"):
        obj = getattr(V, fn, None)
        if obj is None:
            continue
        try:                       # the checkpoint supplies the weights; skip the download
            net = obj(pretrained=False)
        except TypeError:
            net = obj() if callable(obj) else obj
        break
    else:
        raise RuntimeError("prism/models/vpr.py exposes no known model constructor "
                           "(looked for build_export_model/build_model/VPRNet/GeMNet/VPRModel)")
    sd = torch.load(CKPT_DIR / "vpr_gem_mnv3.pt", map_location="cpu", weights_only=False)
    for key in ("model", "state_dict", "net"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
    return net.eval()


def _build_world_model() -> nn.Module:
    from ..models.world_model import PrismWorldModel
    m = PrismWorldModel()
    sd = torch.load(CKPT_DIR / "world_model.pt", map_location="cpu", weights_only=False)
    m.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
    return _WorldModelWrap(m.eval(), CFG.wm.pred_horizon).eval()


def _vpr_input_shape() -> tuple:
    """The VPR agent owns its input size; read it rather than assume."""
    try:
        from ..models import vpr as V
        for attr in ("VPR_INPUT", "INPUT_SIZE", "NET_SIZE"):
            v = getattr(V, attr, None)
            if v is not None:
                w, h = (int(v[0]), int(v[1])) if len(v) == 2 else (int(v), int(v))
                return (1, 3, h, w)
        h = int(getattr(V, "IN_H", getattr(V, "NET_H", 224)))
        w = int(getattr(V, "IN_W", getattr(V, "NET_W", 224)))
        return (1, 3, h, w)
    except Exception:
        return (1, 3, 224, 224)


def _vpr_input(bgr: np.ndarray, clip_id: str = "", idx: int = -1) -> np.ndarray:
    """Exactly ``models/vpr.preprocess`` when it is importable, else the same recipe."""
    try:
        from ..models import vpr as V
        return np.ascontiguousarray(V.preprocess(bgr).numpy().astype(np.float32))
    except Exception:
        shp = _vpr_input_shape()
        return _imagenet_chw(bgr, (shp[3], shp[2]), cv2.INTER_AREA)


def build_specs() -> list[ExportSpec]:
    """The five prism networks, in pipeline order."""
    return [
        ExportSpec(
            name="depth_anything_v2_small",
            desc="Depth Anything V2-Small (ViT-S/14 DINOv2 + DPT head), relative inverse depth",
            provenance="pretrained, downloaded weights (depth-anything/Depth-Anything-V2-Small-hf); not trained here",
            ckpt=None,
            input_name="pixel_values",
            input_shape=(1, 3, DEPTH_INPUT, DEPTH_INPUT),
            output_names=["relative_inverse_depth"],
            build=_build_depth,
            make_input=lambda bgr, clip_id="", idx=-1: depth_input(bgr),
            notes="Input side DEPTH_INPUT=518 from config.py; output is upsampled to PROC by the stage.",
        ),
        ExportSpec(
            name="pidnet_s_terrain",
            desc="PIDNet-S terrain student, prism-7 taxonomy, logits at stride 8",
            provenance="trained in this repo by distillation from a SegFormer-B0/ADE20K teacher on this footage",
            ckpt=CKPT_DIR / "pidnet_s_prism7.pt",
            input_name="image",
            input_shape=(1, 3, SEG_INPUT[1], SEG_INPUT[0]),
            output_names=["terrain_logits"],
            build=_build_pidnet,
            make_input=lambda bgr, clip_id="", idx=-1: seg_input(bgr),
            notes="augment=False (inference config): the P/D auxiliary heads are not exported.",
        ),
        ExportSpec(
            name="trav_unc_shared",
            desc="Shared MobileNetV3-Small trunk + traversability head + uncertainty head",
            provenance="trained in this repo on geometric+semantic pseudo-labels; trunk initialised from torchvision ImageNet weights",
            ckpt=CKPT_DIR / "trav_unc_shared.pt",
            input_name="feature_stack",
            input_shape=(1, 17, 180, 320),
            output_names=["trav_logits", "risk", "unc_raw"],
            build=_build_trav_unc,
            make_input=trav_input,
            notes="17-channel stack at traversability.NET_H x NET_W = 180x320 (PROC/2); "
                  "outputs upsampled to 360x640 inside the graph, as the stage does.",
        ),
        ExportSpec(
            name="vpr_gem_mnv3",
            desc="Visual place recognition: MobileNetV3-Small trunk + GeM pooling -> global descriptor",
            provenance="trained/assembled in this repo on this footage; trunk from torchvision ImageNet weights",
            ckpt=CKPT_DIR / "vpr_gem_mnv3.pt",
            input_name="image",
            input_shape=_vpr_input_shape(),
            output_names=["descriptor"],
            build=_build_vpr,
            make_input=_vpr_input,
            notes="Input size read from prism/models/vpr.py at export time.",
        ),
        ExportSpec(
            name="world_model",
            desc="Tiny BEV world model: encoder + GRU latent dynamics, all 6 actions x 6 steps",
            provenance="trained in this repo on BEV states from this footage plus synthetic augmentation",
            ckpt=CKPT_DIR / "world_model.pt",
            input_name="bev_state",
            input_shape=(1, 8, CFG.bev.H, CFG.bev.W),
            output_names=["occ_forecast", "trav_forecast", "collision_risk"],
            build=_build_world_model,
            make_input=world_model_input,
            notes="One graph = one frame's full planning oracle (6 actions x 6-step rollout).",
        ),
    ]


# ============================================================ export + verify

def _err(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(max absolute error, max relative error over entries with |ref| > 1e-3)."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.shape != b.shape:
        return float("inf"), float("inf")
    d = np.abs(a - b)
    m = np.abs(a) > 1e-3
    rel = float(np.max(d[m] / np.abs(a[m]))) if m.any() else 0.0
    return float(d.max()), rel


@contextlib.contextmanager
def _silence_native_output():
    """Silence stdout/stderr at the file-descriptor level (torch's graph dump is C-level)."""
    saved = (os.dup(1), os.dup(2))
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(devnull, 1); os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        os.close(saved[0]); os.close(saved[1]); os.close(devnull)


class _StaticAdaptiveAvgPool2d(nn.Module):
    """Exact, exportable stand-in for ``nn.AdaptiveAvgPool2d`` at one fixed input size.

    Neither ONNX exporter supports adaptive pooling when the output size is not a
    divisor of the input size - the world model's encoder pools 8x12 -> 3x4, and
    8 / 3 is not an integer, so the ragged overlapping bins have no GlobalAveragePool
    or AveragePool equivalent.  But adaptive average pooling *is separable*: output
    cell (i, j) averages the rectangle [h0_i, h1_i) x [w0_j, w1_j), so

        out = P_h @ x @ P_w^T ,   P_h[i, k] = 1/(h1_i - h0_i) if h0_i <= k < h1_i

    with the same bin edges PyTorch uses (start = floor(i*n/m), end = ceil((i+1)*n/m)).
    Two small constant matmuls, bit-comparable to the original to ~1e-7, and every op
    is core ONNX.  Only used for the exported copy; verification still runs against the
    unmodified PyTorch module.
    """

    def __init__(self, in_hw: tuple[int, int], out_hw: tuple[int, int]):
        super().__init__()
        self.in_hw, self.out_hw = tuple(in_hw), tuple(out_hw)
        self.register_buffer("ph", self._matrix(in_hw[0], out_hw[0]))
        self.register_buffer("pw", self._matrix(in_hw[1], out_hw[1]))

    @staticmethod
    def _matrix(n: int, m: int) -> torch.Tensor:
        p = torch.zeros(m, n)
        for i in range(m):
            a = (i * n) // m
            b = -((-(i + 1) * n) // m)          # ceil((i+1)*n/m)
            p[i, a:b] = 1.0 / float(b - a)
        return p

    def forward(self, x):
        # (B,C,H,W) -> (B,C,h,W) -> (B,C,h,w)
        x = torch.matmul(self.ph, x)
        return torch.matmul(x, self.pw.t())


def _rewrite_adaptive_pools(model: nn.Module, ex: torch.Tensor) -> tuple[nn.Module, int]:
    """Deep-copy `model` and replace every AdaptiveAvgPool2d with an exact static one."""
    import copy
    sizes: dict[int, tuple[int, int]] = {}
    handles = []
    for m in model.modules():
        if isinstance(m, nn.AdaptiveAvgPool2d):
            handles.append(m.register_forward_hook(
                lambda mod, inp, out, _s=sizes: _s.__setitem__(id(mod), tuple(inp[0].shape[-2:]))))
    with torch.no_grad():
        model(ex)
    for h in handles:
        h.remove()
    if not sizes:
        return model, 0

    clone = copy.deepcopy(model)
    # deepcopy changes ids, so walk both trees in lockstep
    src = [m for m in model.modules() if isinstance(m, nn.AdaptiveAvgPool2d)]
    order = {id(m): i for i, m in enumerate(src)}
    by_index = {order[k]: v for k, v in sizes.items() if k in order}

    n = [0]

    def _walk(smod: nn.Module, cmod: nn.Module, counter=[0]):
        for name, child in smod.named_children():
            cchild = getattr(cmod, name)
            if isinstance(child, nn.AdaptiveAvgPool2d):
                in_hw = by_index.get(counter[0])
                counter[0] += 1
                if in_hw is None:
                    continue
                o = child.output_size
                out_hw = (o, o) if isinstance(o, int) else (int(o[0]), int(o[1]))
                setattr(cmod, name, _StaticAdaptiveAvgPool2d(in_hw, out_hw))
                n[0] += 1
            else:
                _walk(child, cchild, counter)

    _walk(model, clone)
    return clone.eval(), n[0]


def _export_graph(model: nn.Module, ex: torch.Tensor, out: Path,
                  spec: "ExportSpec") -> tuple[Optional[str], str]:
    """Export with the TorchScript tracer, falling back to the dynamo exporter.

    The legacy tracer covers everything here except ``adaptive_avg_pool2d`` with a
    non-integer stride ratio (the world model pools 8x12 -> 3x4), which only the
    dynamo/ExportedProgram path handles.  Returns (exporter_used, error_text).
    """
    errors = []

    def _rewritten():
        m, n = _rewrite_adaptive_pools(model, ex)
        return m if n else None

    attempts: list[tuple[str, dict, Callable[[], Optional[nn.Module]]]] = [
        ("torchscript", dict(dynamo=False), lambda: model),
        ("torchscript+static_adaptive_pool", dict(dynamo=False), _rewritten),
        ("dynamo", dict(dynamo=True), lambda: model),
    ]
    for label, kw, get in attempts:
        try:
            mdl = get()
            if mdl is None:
                continue
            # torch dumps the whole traced graph when a symbolic fails; keep the
            # console readable and keep the exception message in the record instead.
            with torch.no_grad(), _silence_native_output():
                torch.onnx.export(
                    mdl, (ex,), str(out),
                    input_names=[spec.input_name], output_names=spec.output_names,
                    opset_version=OPSET, do_constant_folding=True, **kw)
            if out.exists():
                return label, ""
            errors.append(f"{label}: produced no file")
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
    return None, " | ".join(errors)


def export_one(spec: ExportSpec, force: bool = False, n_verify: int = 8,
               verbose: bool = True) -> dict:
    """Export + verify a single model.  Never raises; returns a status dict."""
    rec: dict[str, Any] = {
        "name": spec.name, "desc": spec.desc, "provenance": spec.provenance,
        "input_name": spec.input_name, "input_shape": list(spec.input_shape),
        "output_names": list(spec.output_names), "notes": spec.notes,
        "checkpoint": str(spec.ckpt) if spec.ckpt else None,
        "status": "skipped", "reason": "", "onnx": None, "onnx_mb": None,
        "verified": False, "verify": {},
    }
    if not spec.available():
        rec["reason"] = (f"checkpoint not found: {spec.ckpt} - produced by another agent, "
                         f"re-run this script once it exists")
        if verbose:
            print(f"  [SKIP] {spec.name}: {rec['reason']}")
        return rec

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    out = spec.onnx_path
    stale = force or (not out.exists()) or (
        spec.ckpt is not None and out.stat().st_mtime < spec.ckpt.stat().st_mtime)

    try:
        t0 = time.perf_counter()
        model = spec.build()
        rec["params"] = int(sum(p.numel() for p in model.parameters()))
        rec["params_mb_fp32"] = round(rec["params"] * 4 / 1e6, 3)
    except Exception as e:
        rec["status"] = "failed"
        rec["reason"] = f"could not build/load model: {type(e).__name__}: {e}"
        if verbose:
            print(f"  [FAIL] {spec.name}: {rec['reason']}")
        return rec

    # ---- real frames for tracing + verification ---------------------------------
    # three frames per clip (start / middle / end) so verification spans the daylight,
    # dusk and low-light clips *and* the within-clip variation, not just frame 0.
    frames = sampled_frames(3)
    if not frames:
        rec["status"] = "failed"
        rec["reason"] = "no clip frames available to trace/verify against"
        return rec
    sel = frames[:: max(1, len(frames) // max(n_verify, 1))][:n_verify]
    reset_input_source()
    inputs = []
    for cid, fi, bgr in sel:
        arr = np.asarray(spec.make_input(bgr, clip_id=cid, idx=fi), np.float32)
        if tuple(arr.shape) != tuple(spec.input_shape):
            rec["status"] = "failed"
            rec["reason"] = (f"input builder produced {tuple(arr.shape)}, "
                             f"spec says {tuple(spec.input_shape)}")
            return rec
        inputs.append(arr)
    rec["verify_frames"] = [f"{c}:{i}" for c, i, _ in sel]
    rec["input_source"] = input_source()

    # ---- export ------------------------------------------------------------------
    if stale:
        ex = torch.from_numpy(inputs[0])
        method, err = _export_graph(model, ex, out, spec)
        if method is None:
            rec["status"] = "failed"
            rec["reason"] = f"torch.onnx.export failed: {err}"
            if verbose:
                print(f"  [FAIL] {spec.name}: {rec['reason']}")
            return rec
        rec["exporter"] = method
        rec["export_seconds"] = round(time.perf_counter() - t0, 2)
    else:
        rec["export_seconds"] = 0.0
        rec["exporter"] = "cached (not re-exported; checkpoint older than the graph)"
        rec["reason"] = "already up to date (re-verified only)"

    rec["onnx"] = str(out)
    rec["onnx_mb"] = round(out.stat().st_size / 1e6, 3)

    # ---- verify against PyTorch on real frames -----------------------------------
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        per_out = {n: {"max_abs": 0.0, "max_rel": 0.0} for n in spec.output_names}
        with torch.no_grad():
            for arr in inputs:
                ref = model(torch.from_numpy(arr))
                ref = (ref,) if torch.is_tensor(ref) else tuple(ref)
                got = sess.run(None, {in_name: arr})
                for k, name in enumerate(spec.output_names[:len(got)]):
                    a, r = _err(ref[k].detach().cpu().numpy(), got[k])
                    per_out[name]["max_abs"] = max(per_out[name]["max_abs"], a)
                    per_out[name]["max_rel"] = max(per_out[name]["max_rel"], r)
        for name in per_out:
            per_out[name]["max_abs"] = float(f'{per_out[name]["max_abs"]:.3e}')
            per_out[name]["max_rel"] = float(f'{per_out[name]["max_rel"]:.3e}')
        rec["verify"] = per_out
        ok = all(v["max_abs"] <= ABS_TOL or v["max_rel"] <= REL_TOL for v in per_out.values())
        rec["verified"] = bool(ok)
        rec["status"] = "ok" if ok else "mismatch"
        if not ok:
            rec["reason"] = (f"ONNX Runtime output differs from PyTorch beyond tolerance "
                             f"(abs<={ABS_TOL} or rel<={REL_TOL}) - treat this export as FAILED")
        if verbose:
            tag = "OK  " if ok else "MISMATCH"
            worst = max(per_out.values(), key=lambda v: v["max_abs"])
            print(f"  [{tag}] {spec.name:<24s} {rec['onnx_mb']:7.2f} MB  "
                  f"max|abs|={worst['max_abs']:.2e}  max|rel|={worst['max_rel']:.2e}  "
                  f"params={rec.get('params', 0)/1e6:.2f}M")
    except Exception as e:
        rec["status"] = "failed"
        rec["reason"] = f"verification failed: {type(e).__name__}: {e}"
        if verbose:
            print(f"  [FAIL] {spec.name}: {rec['reason']}")
    return rec


def export_all(only: Optional[list[str]] = None, force: bool = False,
               n_verify: int = 8) -> dict:
    specs = build_specs()
    if only:
        specs = [s for s in specs if any(o in s.name for o in only)]
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"ONNX export -> {ONNX_DIR}")
    print(f"opset {OPSET}, torch {torch.__version__}, "
          f"tolerance abs<={ABS_TOL} / rel<={REL_TOL}\n")
    recs = [export_one(s, force=force, n_verify=n_verify) for s in specs]
    if only:
        # a filtered run must not erase the other models' results from the report
        prev = {r["name"]: r for r in load_export_report().get("models", [])}
        for r in recs:
            prev[r["name"]] = r
        order = [s.name for s in build_specs()]
        recs = [prev[n] for n in order if n in prev]
    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "opset": OPSET,
        "hardware": hardware_info(),
        "tolerance": {"abs": ABS_TOL, "rel": REL_TOL},
        "models": recs,
    }
    p = ONNX_DIR / "export_report.json"
    p.write_text(json.dumps(report, indent=2))
    ok = [r["name"] for r in recs if r["status"] == "ok"]
    bad = [r["name"] for r in recs if r["status"] == "mismatch"]
    skip = [r["name"] for r in recs if r["status"] == "skipped"]
    fail = [r["name"] for r in recs if r["status"] == "failed"]
    print(f"\nverified   : {', '.join(ok) if ok else '(none)'}")
    if bad:
        print(f"MISMATCHED : {', '.join(bad)}   <-- do not deploy these")
    if fail:
        print(f"FAILED     : {', '.join(fail)}")
    if skip:
        print(f"skipped    : {', '.join(skip)}  (checkpoint not written yet)")
    print(f"report     : {p}")
    return report


def load_export_report() -> dict:
    p = ONNX_DIR / "export_report.json"
    return json.loads(p.read_text()) if p.exists() else {"models": []}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None,
                    help="substring filter on model names, e.g. --only depth world")
    ap.add_argument("--force", action="store_true", help="re-export even if up to date")
    ap.add_argument("--verify-frames", type=int, default=8)
    a = ap.parse_args()
    hw = hardware_info()
    print(f"host: {hw['cpu']} ({hw['cpu_physical_cores']}P/{hw['cpu_logical_processors']}T), "
          f"GPU {hw['gpu']}, onnxruntime {hw['onnxruntime']}")
    export_all(a.only, a.force, a.verify_frames)
