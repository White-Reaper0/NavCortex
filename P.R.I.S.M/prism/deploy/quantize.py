"""FP16 / INT8 quantization of the exported prism graphs, with measured accuracy cost.

What this does
--------------
For every verified ONNX graph in ``checkpoints/onnx`` it produces up to three variants
and then **measures what they cost in accuracy on real clip frames**:

    fp16            weights and activations cast to float16 (deployment target: GPU /
                    Jetson; ONNX Runtime's CPU provider has to insert Cast nodes, so
                    this is not a CPU speed play and is not presented as one)
    int8_dynamic    weights quantized offline, activation ranges computed per inference
    int8_static     weights *and* activations quantized against a calibration set of
                    real frames (QDQ format, per-channel weights, uint8 activations)

Calibration
-----------
``ClipCalibrationReader`` streams genuine frames sampled uniformly across **all five
clips**.  That matters: clips 01/02 are daylight gravel trails, clip 03 is dusk, clips
04/05 are low light.  The activation ranges of the first convolution differ by roughly
a factor of two between those groups, so calibrating on daylight only would clip the
low-light clips - exactly the frames where the stack is already weakest.

Accuracy is *measured, never assumed*
-------------------------------------
The reference is always the FP32 ONNX graph (which ``export_onnx.py`` has already shown
matches PyTorch), evaluated on the same frames:

    depth       median / p95 absolute relative error of the **metric-aligned depth map**
                (q -> fit_metric_ground -> depth_from_q, i.e. the number the rest of
                prism consumes), plus the drift in the fitted inverse-depth scale a
    terrain     mIoU and pixel agreement of the argmax label vs the FP32 student
    trav/unc    mIoU of the 4-class traversability label, MAE of the risk and the two
                uncertainty channels
    vpr         mean cosine similarity of the descriptor and top-1 retrieval agreement
    world model MAE of predicted collision risk / traversability / occupancy

A quantized model that loses accuracy is reported as losing accuracy.  A quantized
model that does not get faster on this CPU is reported as not getting faster (see
``benchmark.py``) - INT8 speedup is hardware-dependent and is not assumed here.

Usage
-----
    python -m prism.deploy.quantize                 # all available models
    python -m prism.deploy.quantize --only depth
    python -m prism.deploy.quantize --calib 300 --eval 200
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader

from ..config import CFG, CLIP_IDS, N_TERRAIN, N_TRAV, OUT_DIR, PROC_H, PROC_W
from ..io_utils import ego_mask
from .export_onnx import (ONNX_DIR, ExportSpec, build_specs, hardware_info,
                          sampled_frames)

warnings.filterwarnings("ignore", category=DeprecationWarning)

VARIANTS = ("fp16", "int8_dynamic", "int8_static")
REPORT = OUT_DIR / "quantization.json"

#: Calibration method per model.  Chosen by *measurement*, not by folklore - the
#: comparison below was run on PIDNet-S with 150 calibration frames and 100 evaluation
#: frames, scoring mIoU agreement with the FP32 ONNX graph:
#:
#:      MinMax      per-channel  0.7728        (300 calib frames)
#:      Entropy     per-channel  0.7784   per-tensor  0.7806
#:      Percentile  per-channel  0.8030   per-tensor  0.7965
#:
#: Percentile wins, but only by ~3 mIoU points: the INT8 static loss on this network is
#: structural, not a calibration artefact, and no calibration method rescues it.
#: The depth ViT stays on MinMax because the histogram collectors have to buffer every
#: activation tensor of a 24.8 M-parameter transformer at 518x518 for every calibration
#: frame, which does not fit in a sane amount of RAM here.
CALIB_METHOD = {
    "depth_anything_v2_small": "MinMax",
    "pidnet_s_terrain": "Percentile",
    "trav_unc_shared": "Percentile",
    "vpr_gem_mnv3": "Percentile",
    "world_model": "Percentile",
}
CALIB_COMPARISON_PIDNET = {
    "MinMax_per_channel": 0.7728, "Entropy_per_channel": 0.7784,
    "Entropy_per_tensor": 0.7806, "Percentile_per_channel": 0.8030,
    "Percentile_per_tensor": 0.7965,
    "metric": "mIoU agreement with the FP32 ONNX student, PIDNet-S",
}

# Threads used for every accuracy pass, so any incidental timing here is comparable.
EVAL_THREADS = 4


# ============================================================ session helpers

def make_session(path: Path, threads: int = EVAL_THREADS,
                 provider: str = "CPUExecutionProvider") -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=[provider])


def run_all(sess: ort.InferenceSession, x: np.ndarray) -> list[np.ndarray]:
    name = sess.get_inputs()[0].name
    want = sess.get_inputs()[0].type
    if "float16" in want:
        x = x.astype(np.float16)
    return sess.run(None, {name: x})


def node_histogram(path: Path) -> dict:
    """Which quantized op types actually ended up in the graph (explains the speed)."""
    m = onnx.load(str(path), load_external_data=False)
    h: dict[str, int] = {}
    for n in m.graph.node:
        h[n.op_type] = h.get(n.op_type, 0) + 1
    keys = ("QLinearConv", "QLinearMatMul", "ConvInteger", "MatMulInteger",
            "DynamicQuantizeLinear", "QuantizeLinear", "DequantizeLinear",
            "Conv", "MatMul", "Gemm", "Cast")
    return {k: h[k] for k in keys if k in h}


# ============================================================ calibration

class ClipCalibrationReader(CalibrationDataReader):
    """Feeds genuine frames sampled uniformly across all five clips.

    Not random tensors, and not one clip: the daylight and the low-light clips have
    materially different activation statistics and the calibration has to see both.
    """

    def __init__(self, spec: ExportSpec, input_name: str, n_frames: int = 300):
        per_clip = max(1, int(round(n_frames / len(CLIP_IDS))))
        self.frames = sampled_frames(per_clip)
        self.spec = spec
        self.input_name = input_name
        self.n = len(self.frames)
        self._it = None
        self.clip_counts: dict[str, int] = {}
        for cid, _, _ in self.frames:
            self.clip_counts[cid] = self.clip_counts.get(cid, 0) + 1

    def _gen(self):
        for cid, fi, bgr in self.frames:
            arr = np.asarray(self.spec.make_input(bgr, clip_id=cid, idx=fi), np.float32)
            yield {self.input_name: arr}

    def get_next(self):
        if self._it is None:
            self._it = self._gen()
        return next(self._it, None)

    def rewind(self):
        self._it = None


# ============================================================ variant builders

def build_fp16(src: Path, dst: Path) -> dict:
    from onnxconverter_common import float16 as f16
    m = onnx.load(str(src))
    # keep_io_types=False -> the graph takes and returns float16 directly, which is what
    # a GPU/Jetson deployment wants; the runner casts on the way in.
    m16 = f16.convert_float_to_float16(m, keep_io_types=False, disable_shape_infer=True)
    onnx.save(m16, str(dst))
    return {"method": "onnxconverter_common.float16.convert_float_to_float16",
            "keep_io_types": False}


def _preprocessed(src: Path, tag: str) -> tuple[Path, Optional[Path]]:
    """Shape-inferred / folded copy of the graph.

    The ORT quantizer needs every intermediate tensor to have a known dtype; without
    this step the world-model graph fails with "Unable to find data type for
    weight_name=...".  Returns (path_to_use, temp_to_delete).
    """
    from onnxruntime.quantization.shape_inference import quant_pre_process
    tmp = src.with_suffix(f".{tag}.pre.onnx")
    try:
        quant_pre_process(str(src), str(tmp), skip_symbolic_shape=True)
        return tmp, tmp
    except Exception:
        tmp.unlink(missing_ok=True)
        return src, None


def _gemm_to_matmul_add(src: Path, dst: Path) -> int:
    """Rewrite ``Gemm`` with a constant B into ``MatMul`` + ``Add``.

    Worked around because ORT 1.23's dynamic quantizer does this conversion itself and
    gets ``transB=1`` wrong: it emits ``MatMulInteger(A[m,k], B[n,k])`` and the model
    then fails to load with "Incompatible dimensions for matrix multiplication".  Doing
    the conversion here - transposing the weight initializer up front - keeps the
    numerics identical (verified by the accuracy pass) and lets the quantizer see plain
    MatMul nodes it handles correctly.  Returns the number of rewritten nodes.
    """
    from onnx import helper, numpy_helper
    m = onnx.load(str(src))
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    n_done = 0
    new_nodes = []
    for node in g.node:
        attrs = {a.name: a for a in node.attribute}
        if (node.op_type != "Gemm" or len(node.input) < 2
                or node.input[1] not in inits
                or float(attrs.get("alpha", helper.make_attribute("alpha", 1.0)).f) != 1.0
                or float(attrs.get("beta", helper.make_attribute("beta", 1.0)).f) != 1.0
                or int(attrs.get("transA", helper.make_attribute("transA", 0)).i) != 0):
            new_nodes.append(node)
            continue
        b = numpy_helper.to_array(inits[node.input[1]])
        if int(attrs.get("transB", helper.make_attribute("transB", 0)).i):
            b = np.ascontiguousarray(b.T)
        bname = node.input[1] + "_mm"
        g.initializer.append(numpy_helper.from_array(b, bname))
        mm_out = node.output[0] + "_mm" if len(node.input) > 2 else node.output[0]
        new_nodes.append(helper.make_node("MatMul", [node.input[0], bname], [mm_out],
                                          name=(node.name or mm_out) + "_MatMul"))
        if len(node.input) > 2:
            new_nodes.append(helper.make_node("Add", [mm_out, node.input[2]],
                                              [node.output[0]],
                                              name=(node.name or mm_out) + "_Add"))
        n_done += 1
    if n_done:
        del g.node[:]
        g.node.extend(new_nodes)
        onnx.save(m, str(dst))
    return n_done


def build_int8_dynamic(src: Path, dst: Path) -> dict:
    """Dynamic INT8 restricted to MatMul/Gemm - which is the only thing ORT CPU can run.

    Two hard constraints discovered by actually running this, not assumed:
      * including Conv produces ``ConvInteger`` nodes for which the ORT 1.23 CPU
        provider has **no per-channel kernel** ("Could not find an implementation for
        ConvInteger(10)"), so a Conv-inclusive dynamic model loads and then throws.
      * the quantizer rewrites Gemm into MatMul+Add and then cannot infer the new
        tensor's dtype unless ``DefaultTensorType`` is supplied.
    Consequence, stated plainly in the report: dynamic INT8 does essentially nothing for
    the convolutional students (PIDNet-S, the MobileNet trunk); it only bites on the
    MatMul-dominated ViT depth model.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic
    base, tmp = _preprocessed(src, "dyn")
    flat = src.with_suffix(".dyn.flat.onnx")
    n_gemm = _gemm_to_matmul_add(base, flat)
    if n_gemm:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        base, tmp = flat, flat
    try:
        quantize_dynamic(str(base), str(dst), weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul"],
                         per_channel=False, reduce_range=False,
                         extra_options={"DefaultTensorType": int(onnx.TensorProto.FLOAT)})
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return {"method": "onnxruntime.quantization.quantize_dynamic",
            "weight_type": "QInt8", "per_channel": False,
            "op_types_to_quantize": ["MatMul"],
            "gemm_rewritten_to_matmul": n_gemm,
            "note": "Conv is deliberately excluded: ORT 1.23's CPU provider has no "
                    "usable per-channel ConvInteger kernel, so conv-dominated graphs "
                    "are barely touched by dynamic INT8."}


def build_int8_static(src: Path, dst: Path, reader: ClipCalibrationReader,
                      method: str = "MinMax") -> dict:
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    base, pre = _preprocessed(src, "stat")
    cm = getattr(CalibrationMethod, method)
    quantize_static(
        str(base), str(dst), reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=cm,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    if pre is not None:
        pre.unlink(missing_ok=True)
    for junk in dst.parent.glob("*-opt.onnx"):
        junk.unlink(missing_ok=True)
    return {"method": "onnxruntime.quantization.quantize_static",
            "quant_format": "QDQ", "activation_type": "QUInt8",
            "weight_type": "QInt8", "per_channel": True,
            "calibrate_method": method,
            "calibration_frames": reader.n,
            "calibration_frames_per_clip": reader.clip_counts}


# ============================================================ accuracy evaluators
#
# Every evaluator returns a dict of *measured* deltas against the FP32 ONNX graph
# evaluated on the same frames.  Keys prefixed "d_" are differences, not absolutes.


def _softmax_argmax(logits: np.ndarray) -> np.ndarray:
    return np.asarray(logits[0].argmax(0), np.uint8)


def _miou(a: np.ndarray, b: np.ndarray, n_cls: int) -> tuple[float, float]:
    """(mIoU, pixel agreement) treating `a` as reference and `b` as prediction."""
    k = (a.astype(np.int64) * n_cls + b.astype(np.int64)).ravel()
    cm = np.bincount(k, minlength=n_cls * n_cls).reshape(n_cls, n_cls).astype(np.float64)
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    present = union > 0
    miou = float(np.mean(inter[present] / union[present])) if present.any() else float("nan")
    return miou, float(inter.sum() / max(cm.sum(), 1))


def _depth_from_network(q_raw: np.ndarray) -> tuple[np.ndarray, float]:
    """Network output -> metric-aligned depth, exactly as ``models/depth.py`` does it."""
    import cv2
    from ..perception.geometry import GroundFit, depth_from_q, fit_metric_ground
    q = np.asarray(q_raw, np.float32)
    while q.ndim > 2:
        q = q[0]
    q = cv2.resize(q, (PROC_W, PROC_H), interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(q, 0.5), np.percentile(q, 99.5)
    q = np.clip((q - lo) / max(hi - lo, 1e-6), 0.0, 1.2)
    em = ego_mask(PROC_H, PROC_W)
    valid = em & np.isfinite(q)
    fit = fit_metric_ground(q, valid)
    if not fit.ok:
        fit = GroundFit(a=1.0, b=0.0, height=CFG.cam.height_above_ground_m, ok=False)
    d, v = depth_from_q(q, fit)
    d[~(v & em)] = np.nan
    return d, float(fit.a)


def eval_depth(ref: list, got: list) -> dict:
    """Median / p95 absolute relative error on the metric-aligned depth map."""
    rels, a_ref, a_got = [], [], []
    for r, g in zip(ref, got):
        dr, ar = _depth_from_network(r[0])
        dg, ag = _depth_from_network(g[0])
        m = np.isfinite(dr) & np.isfinite(dg) & (dr > 0.15)
        if m.sum() < 100:
            continue
        rels.append(np.abs(dg[m] - dr[m]) / dr[m])
        a_ref.append(ar)
        a_got.append(ag)
    if not rels:
        return {"error": "no valid pixels"}
    allr = np.concatenate(rels)
    return {
        "metric": "absolute relative error of metric-aligned depth vs FP32 ONNX",
        "median_abs_rel": round(float(np.median(allr)), 6),
        "p95_abs_rel": round(float(np.percentile(allr, 95)), 6),
        "mean_abs_rel": round(float(allr.mean()), 6),
        "n_frames": len(rels),
        "scale_a_fp32_mean": round(float(np.mean(a_ref)), 5),
        "scale_a_quant_mean": round(float(np.mean(a_got)), 5),
        "d_scale_a_pct": round(float(100 * (np.mean(a_got) - np.mean(a_ref)) /
                                     max(abs(np.mean(a_ref)), 1e-9)), 4),
    }


def eval_seg(ref: list, got: list) -> dict:
    n = N_TERRAIN
    cm = np.zeros((n, n), np.float64)
    for r, g in zip(ref, got):
        a, b = _softmax_argmax(r[0]), _softmax_argmax(g[0])
        k = (a.astype(np.int64) * n + b.astype(np.int64)).ravel()
        cm += np.bincount(k, minlength=n * n).reshape(n, n)
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    p = union > 0
    return {
        "metric": "agreement with the FP32 ONNX student (not ground truth)",
        "mIoU_vs_fp32": round(float(np.mean(inter[p] / union[p])), 5),
        "pixel_agreement": round(float(inter.sum() / max(cm.sum(), 1)), 5),
        "n_frames": len(ref),
        "classes_present": int(p.sum()),
    }


def eval_trav_unc(ref: list, got: list) -> dict:
    n = N_TRAV
    cm = np.zeros((n, n), np.float64)
    risk_err, unc_err = [], []
    for r, g in zip(ref, got):
        a, b = _softmax_argmax(r[0]), _softmax_argmax(g[0])
        k = (a.astype(np.int64) * n + b.astype(np.int64)).ravel()
        cm += np.bincount(k, minlength=n * n).reshape(n, n)
        rr = 1.0 / (1.0 + np.exp(-np.asarray(r[1], np.float32)))
        gg = 1.0 / (1.0 + np.exp(-np.asarray(g[1], np.float32)))
        risk_err.append(np.abs(gg - rr).mean())
        unc_err.append(np.abs(np.asarray(g[2], np.float32) -
                              np.asarray(r[2], np.float32)).mean())
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    p = union > 0
    return {
        "metric": "agreement with the FP32 ONNX head (labels are pseudo-labels, not GT)",
        "mIoU_vs_fp32": round(float(np.mean(inter[p] / union[p])), 5),
        "pixel_agreement": round(float(inter.sum() / max(cm.sum(), 1)), 5),
        "risk_mae": round(float(np.mean(risk_err)), 6),
        "uncertainty_raw_mae": round(float(np.mean(unc_err)), 6),
        "n_frames": len(ref),
    }


def eval_vpr(ref: list, got: list) -> dict:
    R = np.stack([np.asarray(r[0], np.float32).ravel() for r in ref])
    G = np.stack([np.asarray(g[0], np.float32).ravel() for g in got])
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-9
    G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-9
    cos = float(np.mean(np.sum(R * G, 1)))
    # does the quantized descriptor retrieve the same nearest neighbour?
    def top1(M):
        S = M @ M.T
        np.fill_diagonal(S, -2.0)
        return S.argmax(1)
    agree = float(np.mean(top1(R) == top1(G)))
    return {
        "metric": "descriptor fidelity vs FP32 ONNX",
        "mean_cosine_similarity": round(cos, 6),
        "top1_retrieval_agreement": round(agree, 4),
        "n_frames": len(ref),
    }


def eval_world_model(ref: list, got: list) -> dict:
    occ, trav, risk = [], [], []
    for r, g in zip(ref, got):
        occ.append(np.abs(np.asarray(g[0], np.float32) - np.asarray(r[0], np.float32)).mean())
        trav.append(np.abs(np.asarray(g[1], np.float32) - np.asarray(r[1], np.float32)).mean())
        risk.append(np.abs(np.asarray(g[2], np.float32) - np.asarray(r[2], np.float32)).mean())
    # does the risk ranking over the six actions survive quantization?
    same_best = np.mean([int(np.argmin(np.asarray(g[2]).mean(-1)) ==
                             np.argmin(np.asarray(r[2]).mean(-1)))
                         for r, g in zip(ref, got)])
    return {
        "metric": "rollout fidelity vs FP32 ONNX",
        "occupancy_mae": round(float(np.mean(occ)), 6),
        "trav_forecast_mae": round(float(np.mean(trav)), 6),
        "collision_risk_mae": round(float(np.mean(risk)), 6),
        "safest_action_agreement": round(float(same_best), 4),
        "n_frames": len(ref),
    }


EVALUATORS: dict[str, Callable[[list, list], dict]] = {
    "depth_anything_v2_small": eval_depth,
    "pidnet_s_terrain": eval_seg,
    "trav_unc_shared": eval_trav_unc,
    "vpr_gem_mnv3": eval_vpr,
    "world_model": eval_world_model,
}


# ============================================================ driver

def _eval_inputs(spec: ExportSpec, n_frames: int) -> list[np.ndarray]:
    per_clip = max(1, int(round(n_frames / len(CLIP_IDS))))
    return [np.asarray(spec.make_input(bgr, clip_id=cid, idx=fi), np.float32)
            for cid, fi, bgr in sampled_frames(per_clip)]


def quantize_model(spec: ExportSpec, calib_frames: int, eval_frames: int,
                   force: bool = False, verbose: bool = True) -> dict:
    src = spec.onnx_path
    rec = {"name": spec.name, "fp32_onnx": str(src),
           "fp32_mb": None, "variants": {}}
    if not src.exists():
        rec["status"] = "skipped"
        rec["reason"] = f"no FP32 export at {src} (run export_onnx.py; checkpoint may be missing)"
        if verbose:
            print(f"  [SKIP] {spec.name}: {rec['reason']}")
        return rec
    rec["fp32_mb"] = round(src.stat().st_size / 1e6, 3)
    rec["fp32_nodes"] = node_histogram(src)
    rec["status"] = "ok"

    # --- reference outputs on real frames -----------------------------------------
    from .export_onnx import input_source, reset_input_source
    reset_input_source()
    xs = _eval_inputs(spec, eval_frames)
    rec["input_source"] = input_source()
    ref_sess = make_session(src)
    in_name = ref_sess.get_inputs()[0].name
    t0 = time.perf_counter()
    ref = [run_all(ref_sess, x) for x in xs]
    rec["eval_frames"] = len(xs)
    rec["fp32_eval_seconds"] = round(time.perf_counter() - t0, 2)
    del ref_sess

    reader = None
    for variant in VARIANTS:
        dst = ONNX_DIR / f"{spec.name}.{variant}.onnx"
        v: dict = {"path": str(dst)}
        try:
            if force or not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                t1 = time.perf_counter()
                if variant == "fp16":
                    v.update(build_fp16(src, dst))
                elif variant == "int8_dynamic":
                    v.update(build_int8_dynamic(src, dst))
                else:
                    if reader is None:
                        reader = ClipCalibrationReader(spec, in_name, calib_frames)
                    reader.rewind()
                    v.update(build_int8_static(src, dst, reader,
                                               CALIB_METHOD.get(spec.name, "MinMax")))
                v["build_seconds"] = round(time.perf_counter() - t1, 1)
            else:
                v["build_seconds"] = 0.0
                v["note_cached"] = "already built and newer than the FP32 graph"
            v["size_mb"] = round(dst.stat().st_size / 1e6, 3)
            v["size_vs_fp32"] = round(v["size_mb"] / rec["fp32_mb"], 4)
            v["nodes"] = node_histogram(dst)

            sess = make_session(dst)
            t2 = time.perf_counter()
            got = [run_all(sess, x) for x in xs]
            v["eval_seconds"] = round(time.perf_counter() - t2, 2)
            del sess
            v["accuracy"] = EVALUATORS[spec.name](ref, got)
            v["status"] = "ok"
            if verbose:
                acc = v["accuracy"]
                head = next((f"{k}={acc[k]}" for k in
                             ("median_abs_rel", "mIoU_vs_fp32", "mean_cosine_similarity",
                              "collision_risk_mae") if k in acc), "")
                print(f"    {variant:<13s} {v['size_mb']:8.2f} MB "
                      f"({v['size_vs_fp32']*100:5.1f}% of fp32)  {head}")
        except Exception as e:
            v["status"] = "failed"
            v["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            if verbose:
                print(f"    {variant:<13s} FAILED: {v['error']}")
        rec["variants"][variant] = v
    return rec


def quantize_all(only: Optional[list[str]] = None, calib_frames: int = 300,
                 eval_frames: int = 200, force: bool = False) -> dict:
    specs = build_specs()
    if only:
        specs = [s for s in specs if any(o in s.name for o in only)]
    print(f"quantizing -> {ONNX_DIR}")
    print(f"calibration frames: {calib_frames} across {len(CLIP_IDS)} clips "
          f"(daylight + dusk + low light); accuracy frames: {eval_frames}\n")
    recs = []
    for s in specs:
        print(f"  {s.name}")
        recs.append(quantize_model(s, calib_frames, eval_frames, force))
    if only:
        # a filtered run must not erase the other models' results from the report
        prev = {r["name"]: r for r in load_report().get("models", [])}
        for r in recs:
            prev[r["name"]] = r
        recs = [prev[s.name] for s in build_specs() if s.name in prev]
    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": hardware_info(),
        "calibration_frames_requested": calib_frames,
        "eval_frames_requested": eval_frames,
        "eval_threads": EVAL_THREADS,
        "reference": "FP32 ONNX graph (already verified against PyTorch by export_onnx.py)",
        "calibration_method_per_model": CALIB_METHOD,
        "calibration_method_comparison": CALIB_COMPARISON_PIDNET,
        "models": recs,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {REPORT}")
    return report


def load_report() -> dict:
    return json.loads(REPORT.read_text()) if REPORT.exists() else {"models": []}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--calib", type=int, default=300, help="calibration frames (static INT8)")
    ap.add_argument("--eval", type=int, default=200, help="frames for the accuracy pass")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    hw = hardware_info()
    print(f"host: {hw['cpu']} ({hw['cpu_physical_cores']}P/{hw['cpu_logical_processors']}T), "
          f"onnxruntime {hw['onnxruntime']}\n")
    quantize_all(a.only, a.calib, a.eval, a.force)
