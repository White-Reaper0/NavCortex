"""The prism latency table: measured, per stage, per runtime, on named hardware.

This is the file that decides whether the "CPU-friendly deployment" claim in the pitch
is worth anything.  It measures - it does not estimate, except in one section that is
explicitly and repeatedly labelled as an estimate.

Methodology (deliberately boring, because that is what makes it trustworthy)
---------------------------------------------------------------------------
* Warm-up iterations before every measurement (default 10; the first CUDA launch and
  ONNX Runtime's first arena allocation are not representative of steady state).
* At least 30 timed iterations, and **median and p95 are reported, not just the mean** -
  on a laptop under thermal and scheduler noise the mean is dominated by outliers and
  flatters or damns a runtime for the wrong reason.
* CUDA work is synchronised inside the timing loop.
* ONNX Runtime thread count is pinned and reported with every number
  (``intra_op_num_threads``); an unpinned ORT will silently use every core and the
  number stops meaning anything about an embedded target.
* **GPU and CPU phases are sequenced, never interleaved.**  This machine's GPU is shared
  with other jobs; the CPU phase runs after the GPU phase has finished and the context
  has been freed, so the CPU numbers are not measuring GPU contention.
* Inputs are real clip frames wherever the model consumes an image.

What gets measured
------------------
Per network, whichever of these exist:
    PyTorch GPU FP16 / GPU FP32 / CPU FP32,
    ONNX Runtime CPU FP32 / INT8 static / INT8 dynamic / FP16.
Plus the non-network stages, which are where a surprising share of the frame budget
actually goes: the metric ground fit, the geometry maps, the 17-channel feature stack,
ORB visual odometry, BEV mapping, and the planner sweep.

The Jetson section is a **projection, not a measurement**.  There is no Jetson in this
environment.  It is derived from published throughput ratios, its reasoning is written
out in the JSON, and it is drawn in a visually separate, explicitly labelled block.

Outputs
-------
    output/benchmarks.json     every measurement, plus hardware and methodology
    output/benchmarks.png      the same table rendered with prism/viz_common.py

Usage
-----
    python -m prism.deploy.benchmark
    python -m prism.deploy.benchmark --iters 60 --threads 4
    python -m prism.deploy.benchmark --skip-gpu        # CPU-only pass
    python -m prism.deploy.benchmark --render-only     # redraw the PNG from the JSON
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch

from .. import viz_common as V
from ..config import CFG, CLIP_IDS, OUT_DIR, PROC_H, PROC_W
from .export_onnx import ONNX_DIR, build_specs, hardware_info, sampled_frames

BENCH_JSON = OUT_DIR / "benchmarks.json"
BENCH_PNG = OUT_DIR / "benchmarks.png"

DEFAULT_ITERS = 40
DEFAULT_WARMUP = 10
#: Pinned ORT / torch CPU thread count for the headline table.  Four threads is the
#: honest analogue of a small embedded CPU (a Jetson Orin Nano has six Cortex-A78AE
#: cores); the thread sweep at the end shows how the numbers move with more.
DEFAULT_THREADS = 4
VRAM_BUDGET_GB = 1.0


# ============================================================ timing core

def timed(fn: Callable[[], None], warmup: int, iters: int,
          sync: Optional[Callable[[], None]] = None) -> dict:
    """Run `fn` and return median / p95 / mean / min latency in milliseconds."""
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    ts = np.empty(iters, np.float64)
    for i in range(iters):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        ts[i] = (time.perf_counter() - t0) * 1e3
    return {
        "median_ms": round(float(np.median(ts)), 3),
        "p95_ms": round(float(np.percentile(ts, 95)), 3),
        "mean_ms": round(float(ts.mean()), 3),
        "min_ms": round(float(ts.min()), 3),
        "std_ms": round(float(ts.std()), 3),
        "fps_median": round(1000.0 / max(float(np.median(ts)), 1e-9), 2),
        "iters": int(iters),
        "warmup": int(warmup),
    }


def _cuda_sync():
    torch.cuda.synchronize()


# ============================================================ torch benchmarks

def bench_torch(build: Callable[[], torch.nn.Module], x: np.ndarray, device: str,
                dtype: str, iters: int, warmup: int, threads: int) -> dict:
    """One PyTorch measurement.  Returns a record with status/reason on failure."""
    rec = {"runtime": "pytorch", "device": device, "dtype": dtype,
           "threads": threads if device == "cpu" else None}
    try:
        model = build().eval()
        t = torch.from_numpy(np.ascontiguousarray(x))
        if device == "cuda":
            if not torch.cuda.is_available():
                rec.update(status="skipped", reason="no CUDA device")
                return rec
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = model.to("cuda")
            t = t.to("cuda")
            if dtype == "fp16":
                model, t = model.half(), t.half()
            with torch.no_grad():
                r = timed(lambda: model(t), warmup, iters, _cuda_sync)
            peak = torch.cuda.max_memory_allocated() / 2**30
            r["peak_vram_gb"] = round(peak, 3)
            r["vram_budget_gb"] = VRAM_BUDGET_GB
            if peak > VRAM_BUDGET_GB:
                r["vram_warning"] = (f"peak {peak:.2f} GB exceeded the {VRAM_BUDGET_GB} GB "
                                     f"budget agreed with the other agents on this GPU")
            model.cpu()
            del model, t
            torch.cuda.empty_cache()
        else:
            old = torch.get_num_threads()
            torch.set_num_threads(threads)
            try:
                with torch.no_grad():
                    r = timed(lambda: model(t), warmup, iters)
            finally:
                torch.set_num_threads(old)
            del model, t
        gc.collect()
        rec.update(r)
        rec["status"] = "ok"
    except Exception as e:
        rec.update(status="failed", reason=f"{type(e).__name__}: {str(e)[:200]}")
    return rec


# ============================================================ onnxruntime benchmarks

def bench_ort(path: Path, x: np.ndarray, iters: int, warmup: int, threads: int,
              label: str, provider: str = "CPUExecutionProvider") -> dict:
    import onnxruntime as ort
    rec = {"runtime": "onnxruntime", "device": "cpu" if provider.startswith("CPU") else "gpu",
           "dtype": label, "threads": threads, "provider": provider,
           "onnx": str(path), "onnx_mb": None}
    if not path.exists():
        rec.update(status="skipped", reason=f"variant not built: {path.name}")
        return rec
    rec["onnx_mb"] = round(path.stat().st_size / 1e6, 3)
    try:
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        sess = ort.InferenceSession(str(path), so, providers=[provider])
        name = sess.get_inputs()[0].name
        xin = x.astype(np.float16) if "float16" in sess.get_inputs()[0].type else x
        xin = np.ascontiguousarray(xin)
        io = {name: xin}
        rec.update(timed(lambda: sess.run(None, io), warmup, iters))
        rec["status"] = "ok"
        gc.collect()
    except Exception as e:
        rec.update(status="failed", reason=f"{type(e).__name__}: {str(e)[:200]}")
    return rec


# ============================================================ non-network stages

def bench_pipeline_stages(iters: int, warmup: int, threads: int) -> list[dict]:
    """The classical / numpy stages, so the frame budget is complete and honest.

    These dominate more of the budget than people expect, which is exactly why they are
    in the table: a pitch that only quotes network latency is not quoting the pipeline.
    """
    import cv2 as _cv
    from ..io_utils import ego_mask, has_stage, load_stage
    from ..models import traversability as T
    from ..nav import bev_utils as bu
    from ..perception import geometry as G

    _cv.setNumThreads(threads)
    out: list[dict] = []
    cid = CLIP_IDS[0]

    def add(name: str, desc: str, fn, extra: Optional[dict] = None):
        rec = {"stage": name, "desc": desc, "runtime": "numpy/opencv",
               "device": "cpu", "threads": threads}
        try:
            rec.update(timed(fn, warmup, iters))
            rec["status"] = "ok"
        except Exception as e:
            rec.update(status="failed", reason=f"{type(e).__name__}: {str(e)[:200]}")
        if extra:
            rec.update(extra)
        out.append(rec)
        return rec

    # ---- real depth cache -> metric alignment + geometry maps ---------------------
    if has_stage(cid, "depth"):
        z = load_stage(cid, "depth")
        q = np.asarray(z["q"][10], np.float32)
        depth = np.asarray(z["depth"][10], np.float32)
        valid = np.asarray(z["valid"][10], bool)
        fit = G.GroundFit(a=float(z["scale"][10]), b=float(z["shift"][10]),
                          normal=np.asarray(z["normal"][10], np.float32),
                          height=CFG.cam.height_above_ground_m, ok=True)
        em = ego_mask(PROC_H, PROC_W)
        base_valid = em & np.isfinite(q)
        add("metric_ground_fit",
            "geometry.fit_metric_ground - SVD + Cauchy IRLS over ~6k ground candidates",
            lambda: G.fit_metric_ground(q, base_valid),
            {"input": f"real q map from work/cache/{cid}/depth.npz, 640x360"})
        add("depth_from_q", "geometry.depth_from_q - apply 1/D = a*q + b",
            lambda: G.depth_from_q(q, fit))

        def _geom():
            pc = G.unproject(np.nan_to_num(depth, nan=0.0))
            pv = G.to_vehicle(pc, fit)
            G.height_slope_roughness(pv, valid)
        add("geometry_maps",
            "unproject + to_vehicle + height/slope/roughness at 640x360", _geom)

        # 17-channel traversability feature stack from the same real arrays
        lab = np.ones((PROC_H, PROC_W), np.uint8)
        lab[:int(0.38 * PROC_H)] = 0
        pmax = np.full((PROC_H, PROC_W), 0.7, np.float32)
        ent = np.full((PROC_H, PROC_W), 0.35, np.float32)
        hh, ss, rr, _ = T.geometry_from_depth(np.nan_to_num(depth, nan=0.0), valid, fit)
        add("trav_feature_stack",
            "traversability.build_feature_stack + resize_stack (17 ch, 640x360 -> 320x180)",
            lambda: T.resize_stack(T.build_feature_stack(
                np.zeros((PROC_H, PROC_W, 3), np.uint8), depth, valid, hh, ss, rr,
                lab, pmax, ent)))
    else:
        out.append({"stage": "metric_ground_fit", "status": "skipped",
                    "reason": f"no depth cache for {cid}"})

    # ---- ORB visual odometry on two consecutive real frames -----------------------
    try:
        from ..io_utils import read_frames
        from ..perception.odometry import OdometryStage
        from ..types import FramePacket
        frames = [f for _, f in read_frames(cid, max_frames=32)]
        st = OdometryStage(device="cpu")
        st.reset()
        counter = {"i": 0}

        def _vo():
            i = counter["i"]
            counter["i"] += 1
            if i and i % len(frames) == 0:
                st.reset()                      # wrap: restart the track, not a jump cut
            j = i % len(frames)
            st(FramePacket(clip_id=cid, idx=j, t=j / 30.0, rgb=frames[j]))
        add("orb_visual_odometry",
            "perception.odometry.OdometryStage - ORB detect/match/essential/pose (classical)",
            _vo, {"input": f"consecutive real frames from {cid}"})
    except Exception as e:
        out.append({"stage": "orb_visual_odometry", "status": "failed",
                    "reason": f"{type(e).__name__}: {str(e)[:200]}"})

    # ---- BEV mapping on the ray-traced synthetic scene ----------------------------
    try:
        from ..perception import mapping as M
        depth_s, valid_s, trav_s, seg_s, _ = M.synthetic_scene(0.0, 0.0)
        stage = M.MappingStage(device="cpu")
        stage.reset()
        pk = M._packet_from_scene(0, depth_s, valid_s, trav_s, seg_s, d_trans=0.02)
        add("bev_mapping",
            "perception.mapping.MappingStage - unproject, roll, splat, fuse into 128x192 BEV",
            lambda: stage(pk),
            {"input": "mapping.synthetic_scene (ray-traced, exactly known geometry) - "
                      "the real perception cache is not required to time this stage"})
    except Exception as e:
        out.append({"stage": "bev_mapping", "status": "failed",
                    "reason": f"{type(e).__name__}: {str(e)[:200]}"})

    # ---- planner + supervisor on a BEV world state --------------------------------
    state = bu.synthetic_state("wall", rng=np.random.default_rng(0))
    try:
        from ..nav.planner import Planner
        pl = Planner()
        add("planner_sweep",
            "nav.planner.Planner.plan - 6 candidate arcs, footprint sweep + cost",
            lambda: pl.plan(state),
            {"input": "bev_utils.synthetic_state('wall')"})
        trajs = pl.plan(state)
        try:
            from ..nav.supervisor import Supervisor
            sup = Supervisor(pl)
            add("safety_supervisor",
                "nav.supervisor.Supervisor.decide - rule cascade over the planner's arcs",
                lambda: sup.decide(state, trajs))
        except Exception as e:
            out.append({"stage": "safety_supervisor", "status": "skipped",
                        "reason": f"{type(e).__name__}: {str(e)[:160]}"})
    except Exception as e:
        # planner not written yet: time its geometric core instead, and say so
        out.append({"stage": "planner_sweep", "status": "skipped",
                    "reason": f"nav.planner unavailable ({type(e).__name__}); "
                              f"timing the footprint sweep instead"})
        step = bu.height_step_map(state)

        def _sweep():
            for a in range(6):
                v, w = bu.action_motion(a)
                xy, yaw = bu.arc_poses(v, w, bu.WM_DT, CFG.safety.n_rollout_steps)
                bu.sweep(state, xy, yaw, CFG.safety.corridor_margin_m, step)
        add("planner_footprint_sweep",
            "bev_utils.sweep over 6 action arcs (planner geometric core)", _sweep)
    return out


# ============================================================ the sweep

def model_inputs(spec, n: int = 1) -> np.ndarray:
    frames = sampled_frames(1)
    cid, fi, bgr = frames[len(frames) // 2]
    return np.asarray(spec.make_input(bgr, clip_id=cid, idx=fi), np.float32)


def run_gpu_phase(specs, inputs, iters, warmup) -> dict:
    """Every PyTorch GPU measurement, done first and then fully released."""
    res: dict[str, list[dict]] = {}
    if not torch.cuda.is_available():
        return res
    for spec in specs:
        if spec.name not in inputs:
            continue
        rows = []
        for dtype in ("fp16", "fp32"):
            print(f"    gpu  {spec.name:<24s} {dtype}", flush=True)
            rows.append(bench_torch(spec.build, inputs[spec.name], "cuda", dtype,
                                    iters, warmup, DEFAULT_THREADS))
        res[spec.name] = rows
    torch.cuda.empty_cache()
    gc.collect()
    return res


def run_cpu_phase(specs, inputs, iters, warmup, threads) -> dict:
    res: dict[str, list[dict]] = {}
    variants = [("fp32", ""), ("int8_static", ".int8_static"),
                ("int8_dynamic", ".int8_dynamic"), ("fp16", ".fp16")]
    for spec in specs:
        if spec.name not in inputs:
            continue
        x = inputs[spec.name]
        rows = [bench_torch(spec.build, x, "cpu", "fp32", iters, warmup, threads)]
        print(f"    cpu  {spec.name:<24s} pytorch fp32 "
              f"{rows[-1].get('median_ms', '-')} ms", flush=True)
        for label, suffix in variants:
            p = ONNX_DIR / f"{spec.name}{suffix}.onnx"
            r = bench_ort(p, x, iters, warmup, threads, label)
            print(f"    cpu  {spec.name:<24s} ort {label:<12s} "
                  f"{r.get('median_ms', r.get('reason', '-'))}", flush=True)
            rows.append(r)
        res[spec.name] = rows
    return res


def thread_sweep(spec, x, iters, warmup, counts=(1, 2, 4, 8)) -> list[dict]:
    """How ONNX Runtime CPU scales with threads - the number nobody pins and everyone quotes."""
    out = []
    for t in counts:
        r = bench_ort(ONNX_DIR / f"{spec.name}.onnx", x, iters, warmup, t, "fp32")
        r["stage"] = spec.name
        out.append(r)
    return out


# ============================================================ Jetson projection

def jetson_projection(measured: dict) -> dict:
    """A PROJECTION.  There is no Jetson in this environment and nothing here was run on one.

    The reasoning is written out so a reader can disagree with it precisely:

    * The measured CPU baseline is a 13th-gen Core i5-13500HX (Raptor Lake, 6 P-cores +
      8 E-cores).  A Jetson Orin Nano's CPU is 6x Arm Cortex-A78AE at 1.5 GHz.  Per
      published SPECint-class comparisons an A78AE core at 1.5 GHz lands roughly a
      factor of 3-4 below a Raptor Lake P-core at its turbo clock on scalar and
      lightly-vectorised integer work; both have 128-bit SIMD (NEON vs AVX2's 256-bit,
      which favours x86 further on the FP32 convolution kernels).  A 3.5x slowdown for
      the CPU rows is therefore the central estimate, with a 3-4x band.
    * The GPU rows are the more interesting projection and the less reliable one.  The
      RTX 4050 Laptop delivers on the order of 20 dense FP16 TFLOPS; an Orin Nano 8 GB
      is specified at 20 sparse INT8 TOPS, i.e. roughly 2.5 dense FP16 TFLOPS - about
      an 8x gap in raw arithmetic, but Jetson has TensorRT and DLA and this repo's
      PyTorch numbers have no TensorRT at all, so a like-for-like port would claw some
      of that back.  The band quoted is 5-8x slower than the measured laptop GPU FP16.
    * None of this accounts for memory bandwidth (Orin Nano 68 GB/s LPDDR5 vs the
      4050's 192 GB/s GDDR6), which for the depth ViT at 518x518 is likely the binding
      constraint, so the depth row in particular should be read as optimistic.

    Every value below is therefore labelled ``estimated``, carries its multiplier, and
    must never be quoted as a prism result.
    """
    cpu_mult = {"low": 3.0, "central": 3.5, "high": 4.0}
    gpu_mult = {"low": 5.0, "central": 6.5, "high": 8.0}
    rows = []
    for name, recs in measured.items():
        best_cpu = next((r for r in recs if r.get("runtime") == "onnxruntime"
                         and r.get("dtype") == "int8_static" and r.get("status") == "ok"), None)
        if best_cpu is None:
            best_cpu = next((r for r in recs if r.get("runtime") == "onnxruntime"
                             and r.get("dtype") == "fp32" and r.get("status") == "ok"), None)
        gpu = next((r for r in recs if r.get("device") == "cuda"
                    and r.get("dtype") == "fp16" and r.get("status") == "ok"), None)
        row = {"model": name, "basis": {}, "estimated": {}}
        if best_cpu:
            row["basis"]["cpu_source"] = f"ORT CPU {best_cpu['dtype']} @ {best_cpu['threads']} threads"
            row["basis"]["cpu_measured_ms"] = best_cpu["median_ms"]
            row["estimated"]["orin_nano_cpu_ms_central"] = round(
                best_cpu["median_ms"] * cpu_mult["central"], 1)
            row["estimated"]["orin_nano_cpu_ms_band"] = [
                round(best_cpu["median_ms"] * cpu_mult["low"], 1),
                round(best_cpu["median_ms"] * cpu_mult["high"], 1)]
        if gpu:
            row["basis"]["gpu_source"] = "PyTorch CUDA FP16 on RTX 4050 Laptop"
            row["basis"]["gpu_measured_ms"] = gpu["median_ms"]
            row["estimated"]["orin_nano_gpu_ms_central"] = round(
                gpu["median_ms"] * gpu_mult["central"], 1)
            row["estimated"]["orin_nano_gpu_ms_band"] = [
                round(gpu["median_ms"] * gpu_mult["low"], 1),
                round(gpu["median_ms"] * gpu_mult["high"], 1)]
        if row["estimated"]:
            rows.append(row)
    return {
        "STATUS": "PROJECTED - NOT MEASURED. No NVIDIA Jetson exists in this environment.",
        "target": "Jetson Orin Nano 8 GB (6x Cortex-A78AE @ 1.5 GHz, Ampere GPU, "
                  "68 GB/s LPDDR5)",
        "cpu_multiplier": cpu_mult,
        "gpu_multiplier": gpu_mult,
        "reasoning": jetson_projection.__doc__.strip(),
        "caveats": [
            "No TensorRT engine was built; a real Jetson port would use TensorRT and "
            "would very likely beat this projection on the GPU rows.",
            "Memory bandwidth is not modelled and is probably the binding constraint "
            "for the 518x518 depth ViT.",
            "Thermal behaviour, power mode (7 W vs 15 W) and DLA offload are not modelled.",
            "Do not quote any number in this section as a prism measurement.",
        ],
        "rows": rows,
    }


# ============================================================ driver

def run(iters: int = DEFAULT_ITERS, warmup: int = DEFAULT_WARMUP,
        threads: int = DEFAULT_THREADS, skip_gpu: bool = False,
        settle_s: float = 3.0) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hw = hardware_info()
    specs = build_specs()

    print(f"host : {hw['cpu']} ({hw['cpu_physical_cores']} physical / "
          f"{hw['cpu_logical_processors']} logical), {hw['ram_gb']} GB RAM")
    print(f"gpu  : {hw['gpu']} ({hw['gpu_vram_gb']} GB) - shared with other agents")
    print(f"iters: {iters} timed after {warmup} warm-up; ORT/torch CPU threads pinned "
          f"to {threads}\n")

    # inputs (real frames) for whatever actually exists
    inputs: dict[str, np.ndarray] = {}
    availability: dict[str, dict] = {}
    for spec in specs:
        onnx_ok = spec.onnx_path.exists()
        ck_ok = spec.available()
        availability[spec.name] = {
            "desc": spec.desc, "provenance": spec.provenance,
            "checkpoint": str(spec.ckpt) if spec.ckpt else "(pretrained download)",
            "checkpoint_present": bool(ck_ok),
            "onnx_present": bool(onnx_ok),
            "input_shape": list(spec.input_shape),
        }
        if not ck_ok:
            availability[spec.name]["missing_reason"] = (
                "checkpoint not written by the owning agent at benchmark time; "
                "re-run export_onnx.py then benchmark.py to fill this row")
            continue
        try:
            inputs[spec.name] = model_inputs(spec)
        except Exception as e:
            availability[spec.name]["missing_reason"] = f"input build failed: {e}"

    # ---------------- phase 1: GPU ------------------------------------------------
    gpu_rows: dict[str, list[dict]] = {}
    if not skip_gpu:
        print("  [phase 1/3] PyTorch GPU (fp16, fp32)")
        gpu_rows = run_gpu_phase(specs, inputs, iters, warmup)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        print(f"  ... GPU context released, settling {settle_s:.0f}s before the CPU phase "
              f"so CPU numbers are not measuring GPU contention\n")
        time.sleep(settle_s)
    else:
        print("  [phase 1/3] GPU phase skipped (--skip-gpu)\n")

    # ---------------- phase 2: CPU ------------------------------------------------
    print("  [phase 2/3] PyTorch CPU + ONNX Runtime CPU")
    cpu_rows = run_cpu_phase(specs, inputs, iters, warmup, threads)
    measured = {k: (gpu_rows.get(k, []) + cpu_rows.get(k, [])) for k in
                set(list(gpu_rows) + list(cpu_rows))}

    # ---------------- phase 3: non-network stages + thread sweep ------------------
    print("\n  [phase 3/3] non-network pipeline stages")
    stages = bench_pipeline_stages(iters, warmup, threads)
    for s in stages:
        print(f"    {s['stage']:<26s} {s.get('median_ms', s.get('reason', '-'))}")

    sweep_spec = next((s for s in specs if s.name in inputs and
                       (ONNX_DIR / f"{s.name}.onnx").exists()), None)
    sweep = thread_sweep(sweep_spec, inputs[sweep_spec.name], iters, warmup) \
        if sweep_spec else []

    # ---------------- assemble ----------------------------------------------------
    budget = frame_budget(measured, stages)
    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": hw,
        "methodology": {
            "warmup_iters": warmup,
            "timed_iters": iters,
            "statistic": "median and p95 reported; mean and std also recorded",
            "ort_intra_op_num_threads": threads,
            "ort_inter_op_num_threads": 1,
            "torch_cpu_threads": threads,
            "cuda_sync": "torch.cuda.synchronize() inside the timing loop",
            "phase_order": "all GPU work completes and the CUDA cache is emptied before "
                           "any CPU timing starts; GPU is shared with other agents on "
                           "this machine",
            "inputs": "real clip frames (or the documented synthetic stand-in where a "
                      "stage's upstream cache does not exist yet)",
            "vram_budget_gb": VRAM_BUDGET_GB,
        },
        "availability": availability,
        "models": measured,
        "pipeline_stages": stages,
        "ort_thread_scaling": {"model": sweep_spec.name if sweep_spec else None,
                               "rows": sweep},
        "frame_budget": budget,
        "jetson_projection": jetson_projection(measured),
        "honesty": [
            "Every millisecond in 'models', 'pipeline_stages' and 'frame_budget' was "
            "measured on the hardware named in 'hardware'.",
            "Nothing here was measured on a UGV, on a Jetson, or on an embedded CPU.",
            "The 'jetson_projection' block is an estimate and is labelled as one.",
            "Rows whose checkpoint did not exist at run time are listed in "
            "'availability' with the reason, not silently omitted.",
        ],
    }
    BENCH_JSON.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {BENCH_JSON}")
    render(report)
    return report


def frame_budget(measured: dict, stages: list[dict]) -> dict:
    """Sum the per-frame cost of one prism frame under three deployment stories."""
    def pick(name, pred):
        return next((r for r in measured.get(name, []) if pred(r) and r.get("status") == "ok"), None)

    nets = list(measured.keys())
    stories = {
        "gpu_fp16_pytorch": lambda n: pick(n, lambda r: r["device"] == "cuda" and r["dtype"] == "fp16"),
        "cpu_fp32_onnxruntime": lambda n: pick(n, lambda r: r["runtime"] == "onnxruntime" and r["dtype"] == "fp32"),
        "cpu_int8_onnxruntime": lambda n: (pick(n, lambda r: r["runtime"] == "onnxruntime" and r["dtype"] == "int8_static")
                                           or pick(n, lambda r: r["runtime"] == "onnxruntime" and r["dtype"] == "fp32")),
    }
    non_net = sum(s["median_ms"] for s in stages
                  if s.get("status") == "ok" and s.get("stage") != "safety_supervisor")
    out = {"non_network_stages_ms": round(non_net, 2),
           "non_network_included": [s["stage"] for s in stages if s.get("status") == "ok"],
           "note": "Sum of medians. The real pipeline overlaps some of this with IO; "
                   "this is an upper bound on the serial cost of one frame.",
           "stories": {}}
    for label, getter in stories.items():
        rows, total, missing = {}, 0.0, []
        for n in nets:
            r = getter(n)
            if r is None:
                missing.append(n)
                continue
            rows[n] = r["median_ms"]
            total += r["median_ms"]
        out["stories"][label] = {
            "networks_ms": {k: round(v, 2) for k, v in rows.items()},
            "networks_total_ms": round(total, 2),
            "plus_non_network_ms": round(total + non_net, 2),
            "fps": round(1000.0 / max(total + non_net, 1e-9), 2),
            "missing_networks": missing,
            "realtime_30fps": bool((total + non_net) <= 33.3),
        }
    return out


# ============================================================ rendering

_W, _H = 1920, 1080


def _fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def render(report: dict, path: Path = BENCH_PNG) -> Path:
    """Draw the benchmark slide with viz_common primitives only (dark instrument style).

    Layout, 1920x1080:
        row 1   network latency, every runtime x every model
        row 2   non-network stages | quantization cost | one-frame budget
        row 3   Jetson projection, visually separated and labelled ESTIMATE
    """
    hw = report["hardware"]
    meth = report["methodology"]
    M, W = 22, _W
    img = V.canvas(_W, _H)
    V.header(img, _W, "prism", "measured deployment benchmarks",
             right=report["generated"], h=54)
    V.text(img, f"{hw['cpu']}  -  {hw['cpu_physical_cores']} physical / "
                f"{hw['cpu_logical_processors']} logical cores  |  {hw['gpu']} "
                f"({hw['gpu_vram_gb']} GB)  |  torch {hw['torch']}  |  "
                f"onnxruntime {hw['onnxruntime']}  |  {hw['os'].split('(')[0].strip()}",
           (M, 76), 0.46, V.TEXT_DIM, 1)
    V.text(img, f"median of {meth['timed_iters']} timed iterations after "
                f"{meth['warmup_iters']} warm-up  |  ONNX Runtime intra-op threads pinned "
                f"to {meth['ort_intra_op_num_threads']}  |  GPU and CPU phases sequenced, "
                f"never concurrent  |  real clip frames as input",
           (M, 96), 0.42, V.ACCENT2, 1)

    # ================================================== row 1: network latency
    order = ["depth_anything_v2_small", "pidnet_s_terrain", "trav_unc_shared",
             "vpr_gem_mnv3", "world_model"]
    keys = [k for k in order if k in report["models"]] + \
           [k for k in report["models"] if k not in order]
    missing = [k for k, v in report["availability"].items()
               if not v.get("checkpoint_present", True)]
    n_rows = len(keys) + len(missing)

    tw = W - 2 * M
    th = 26 * max(n_rows, 1) + 62
    ty = 112
    ix, iy, iw, ih = V.panel(img, M, ty, tw, th,
                             "network latency   ms, median  (p95 beneath)",
                             "lower is better;  '-' = that variant does not exist for "
                             "this model")
    name_w = 330
    var_w = (iw - name_w - 20) // 7
    heads = ["PyTorch GPU fp16", "PyTorch GPU fp32", "PyTorch CPU fp32",
             "ORT CPU fp32", "ORT CPU int8 static", "ORT CPU int8 dynamic", "ORT CPU fp16"]
    variant_pick = [
        ("cuda", "fp16", "pytorch"), ("cuda", "fp32", "pytorch"),
        ("cpu", "fp32", "pytorch"), ("cpu", "fp32", "onnxruntime"),
        ("cpu", "int8_static", "onnxruntime"), ("cpu", "int8_dynamic", "onnxruntime"),
        ("cpu", "fp16", "onnxruntime"),
    ]
    V.text(img, "model", (ix + 10, iy + 20), 0.4, V.TEXT_DIM, 1)
    for i, h in enumerate(heads):
        V.text(img, h, (ix + 10 + name_w + i * var_w, iy + 20), 0.38, V.TEXT_DIM, 1)
    cv2.line(img, (ix + 4, iy + 29), (ix + iw - 4, iy + 29), V.EDGE, 1)

    y = iy + 50
    for k in keys:
        recs = report["models"][k]
        V.text(img, k, (ix + 10, y), 0.44, V.TEXT, 1)
        best = min([q["median_ms"] for q in recs if q.get("status") == "ok"] or [None])
        for i, (dev, dt, rt) in enumerate(variant_pick):
            x = ix + 10 + name_w + i * var_w
            r = next((q for q in recs if q.get("device") == dev and q.get("dtype") == dt
                      and q.get("runtime") == rt and q.get("status") == "ok"), None)
            if r is None:
                V.text(img, "-", (x, y), 0.42, (92, 88, 84), 1)
                continue
            col = V.OK if (best is not None and r["median_ms"] <= best * 1.02) else V.TEXT
            V.text(img, f"{r['median_ms']:.1f}", (x, y), 0.48, col, 1)
            V.text(img, f"p95 {r['p95_ms']:.1f}", (x + 56, y), 0.33, V.TEXT_DIM, 1)
        y += 26
    for k in missing:
        V.text(img, k, (ix + 10, y), 0.44, (120, 116, 110), 1)
        V.text(img, "checkpoint not written by its owning agent when this ran - "
                    "re-run export_onnx.py then benchmark.py to fill this row",
               (ix + 10 + name_w, y), 0.38, V.BAD, 1)
        y += 26

    # ================================================== row 2
    stages = [s for s in report["pipeline_stages"] if s.get("status") == "ok"]
    sy = ty + th + 12
    sh = 26 * max(len(stages), 1) + 62
    qz = _quant_rows(report)
    sh = max(sh, 26 * max(len(qz), 1) + 62, 200)

    # -------- 2a non-network stages
    w_a = 690
    ix, iy, iw, ih = V.panel(img, M, sy, w_a, sh, "non-network stages   CPU, ms",
                             "the rest of the frame budget")
    V.text(img, "stage", (ix + 10, iy + 20), 0.38, V.TEXT_DIM, 1)
    V.text(img, "median", (ix + 400, iy + 20), 0.38, V.TEXT_DIM, 1)
    V.text(img, "p95", (ix + 500, iy + 20), 0.38, V.TEXT_DIM, 1)
    V.text(img, "share", (ix + 580, iy + 20), 0.38, V.TEXT_DIM, 1)
    cv2.line(img, (ix + 4, iy + 29), (ix + iw - 4, iy + 29), V.EDGE, 1)
    tot = max(sum(s["median_ms"] for s in stages), 1e-6)
    y = iy + 50
    for s in stages:
        V.text(img, s["stage"], (ix + 10, y), 0.42, V.TEXT, 1)
        V.text(img, f"{s['median_ms']:.2f}", (ix + 400, y), 0.44, V.ACCENT, 1)
        V.text(img, f"{s['p95_ms']:.2f}", (ix + 500, y), 0.38, V.TEXT_DIM, 1)
        V.bar_meter(img, ix + 578, y - 9, 92, 10, s["median_ms"] / tot, color=V.ACCENT2)
        y += 26

    # -------- 2b quantization cost
    bx = M + w_a + 12
    w_b = 700
    ix, iy, iw, ih = V.panel(img, bx, sy, w_b, sh, "quantization   size and measured cost",
                             "accuracy vs the FP32 ONNX graph on real frames")
    cols_q = [("model  (fp32 size)", 190), ("fp16", 170), ("int8 dynamic", 170),
              ("int8 static", 170)]
    x = ix + 10
    for lab, w in cols_q:
        V.text(img, lab, (x, iy + 20), 0.38, V.TEXT_DIM, 1)
        x += w
    cv2.line(img, (ix + 4, iy + 29), (ix + iw - 4, iy + 29), V.EDGE, 1)
    y = iy + 50
    for r in qz[: (ih - 56) // 26]:
        V.text(img, r["model"], (ix + 10, y), 0.4, V.TEXT, 1)
        V.text(img, r["fp32_size"], (ix + 10, y + 11), 0.32, V.TEXT_DIM, 1)
        x = ix + 10 + cols_q[0][1]
        for vname in ("fp16", "int8_dynamic", "int8_static"):
            cell = r["cells"].get(vname)
            if cell is None:
                V.text(img, "-", (x, y), 0.4, (92, 88, 84), 1)
            else:
                V.text(img, cell["acc"], (x, y), 0.4, cell["col"], 1)
                V.text(img, cell["size"], (x, y + 11), 0.32, V.TEXT_DIM, 1)
            x += 170
        y += 26
    if not qz:
        V.text(img, "run  python -m prism.deploy.quantize  to fill this panel",
               (ix + 10, iy + 50), 0.4, V.TEXT_DIM, 1)

    # -------- 2c one-frame budget
    cx = bx + w_b + 12
    w_c = W - M - cx
    fb = report["frame_budget"]
    ix, iy, iw, ih = V.panel(img, cx, sy, w_c, sh, "one-frame budget",
                             "sum of medians, serial upper bound")
    labels = {"gpu_fp16_pytorch": "GPU fp16 (PyTorch)",
              "cpu_fp32_onnxruntime": "CPU fp32 (ONNX Runtime)",
              "cpu_int8_onnxruntime": "CPU int8 (ONNX Runtime)"}
    y = iy + 26
    worst = max([st["plus_non_network_ms"] for st in fb["stories"].values()] or [1.0])
    for key, lab in labels.items():
        st = fb["stories"].get(key)
        if st is None:
            continue
        V.text(img, lab, (ix + 12, y), 0.42, V.TEXT, 1)
        col = V.OK if st["realtime_30fps"] else V.WARN
        V.text(img, f"{st['plus_non_network_ms']:.0f} ms", (ix + 250, y), 0.5, col, 1)
        V.text(img, f"{st['fps']:.1f} fps", (ix + 340, y), 0.42, V.TEXT_DIM, 1)
        V.bar_meter(img, ix + 12, y + 8, iw - 26, 9,
                    st["plus_non_network_ms"], color=col, lo=0, hi=max(worst, 33.3))
        y += 22
        if st["missing_networks"]:
            V.text(img, "incomplete - missing " + ", ".join(st["missing_networks"]),
                   (ix + 12, y + 12), 0.33, V.BAD, 1)
            y += 14
        y += 18
    V.text(img, f"non-network stages {fb['non_network_stages_ms']:.0f} ms of every frame",
           (ix + 12, iy + ih - 30), 0.38, V.TEXT_DIM, 1)
    V.text(img, "30 fps budget = 33.3 ms  ->  not met on CPU here",
           (ix + 12, iy + ih - 12), 0.38, V.ACCENT2, 1)

    # ================================================== row 3: Jetson projection
    jp = report["jetson_projection"]
    jy = sy + sh + 12
    jh = 84 + 24 * min(len(jp["rows"]), 6) + 44
    w_j = 1260
    ix, iy, iw, ih = V.panel(img, M, jy, w_j, jh,
                             "Jetson Orin Nano   -   PROJECTED, NOT MEASURED",
                             "there is no Jetson in this environment; these are "
                             "estimates and must never be quoted as prism results",
                             accent=V.BAD)
    V.badge(img, ix + 10, iy + 10, "ESTIMATE", V.BAD, 0.44)
    V.text(img, jp["target"], (ix + 140, iy + 30), 0.42, V.TEXT_DIM, 1)
    V.text(img, f"CPU x{jp['cpu_multiplier']['central']} "
                f"(band {jp['cpu_multiplier']['low']}-{jp['cpu_multiplier']['high']}), "
                f"GPU x{jp['gpu_multiplier']['central']} "
                f"(band {jp['gpu_multiplier']['low']}-{jp['gpu_multiplier']['high']}) "
                f"- no TensorRT engine built, memory bandwidth not modelled",
           (ix + 10, iy + 56), 0.38, V.TEXT_DIM, 1)
    y = iy + 84
    V.text(img, "model", (ix + 10, y), 0.38, V.TEXT_DIM, 1)
    V.text(img, "measured here: CPU int8 / GPU fp16", (ix + 330, y), 0.38, V.TEXT_DIM, 1)
    V.text(img, "projected Orin Nano CPU", (ix + 700, y), 0.38, V.TEXT_DIM, 1)
    V.text(img, "projected Orin Nano GPU", (ix + 980, y), 0.38, V.TEXT_DIM, 1)
    cv2.line(img, (ix + 4, iy + 92), (ix + iw - 4, iy + 92), V.EDGE, 1)
    y += 22
    for row in jp["rows"][:6]:
        e, b = row["estimated"], row["basis"]
        V.text(img, row["model"], (ix + 10, y), 0.42, V.TEXT, 1)
        V.text(img, f"{b.get('cpu_measured_ms', '-')} / {b.get('gpu_measured_ms', '-')} ms",
               (ix + 330, y), 0.4, V.TEXT_DIM, 1)
        if "orin_nano_cpu_ms_central" in e:
            V.text(img, f"~{e['orin_nano_cpu_ms_central']} ms  "
                        f"[{e['orin_nano_cpu_ms_band'][0]}-{e['orin_nano_cpu_ms_band'][1]}]",
                   (ix + 700, y), 0.42, V.WARN, 1)
        if "orin_nano_gpu_ms_central" in e:
            V.text(img, f"~{e['orin_nano_gpu_ms_central']} ms  "
                        f"[{e['orin_nano_gpu_ms_band'][0]}-{e['orin_nano_gpu_ms_band'][1]}]",
                   (ix + 980, y), 0.42, V.WARN, 1)
        y += 24

    # -------- 3b ONNX Runtime CPU thread scaling
    tx = M + w_j + 12
    ix, iy, iw, ih = V.panel(img, tx, jy, W - M - tx, jh,
                             "ORT CPU thread scaling",
                             f"{report.get('ort_thread_scaling', {}).get('model') or '-'}, "
                             f"fp32")
    sw = report.get("ort_thread_scaling", {})
    ok = [r for r in sw.get("rows", []) if r.get("status") == "ok"]
    if ok:
        hi = max(r["median_ms"] for r in ok)
        y = iy + 26
        for r in ok:
            V.text(img, f"{r['threads']} thread" + ("s" if r["threads"] != 1 else ""),
                   (ix + 12, y), 0.4, V.TEXT, 1)
            V.text(img, f"{r['median_ms']:.1f} ms", (ix + 150, y), 0.44, V.ACCENT, 1)
            V.bar_meter(img, ix + 12, y + 8, iw - 26, 9, r["median_ms"],
                        color=V.ACCENT2, lo=0, hi=hi)
            y += 34
        V.text(img, "the headline table is pinned at "
                    f"{meth['ort_intra_op_num_threads']} threads",
               (ix + 12, iy + ih - 14), 0.36, V.TEXT_DIM, 1)
    else:
        V.text(img, "not measured in this run", (ix + 12, iy + 30), 0.4, V.TEXT_DIM, 1)

    # ================================================== row 4: honesty strip
    ny = jy + jh + 12
    nh = _H - ny - 32
    ix, iy, iw, ih = V.panel(img, M, ny, W - 2 * M, nh,
                             "read this with the table", "what these numbers do and do "
                             "not claim", accent=V.BAD)
    notes = [
        "Every millisecond outside the Jetson block was measured on the CPU and GPU named in the header, in this repository, today.",
        "Nothing here was measured on a UGV, on a Jetson, or on any embedded CPU. Offline video throughput is not physical autonomy.",
        "CPU-only real-time operation is an optimisation objective of this project, not a demonstrated result - the CPU budget above does not reach 33.3 ms.",
        "Quantization does not guarantee a speedup on every CPU. The INT8 columns are what this machine actually did, including where INT8 was no faster.",
        "Accuracy costs of every quantized variant are measured against the FP32 graph on real clip frames and recorded in output/quantization.json.",
    ]
    if missing:
        notes.append("Rows absent because the owning agent had not written the checkpoint when this ran: "
                     + ", ".join(missing) + ". Re-run export_onnx.py, quantize.py, then benchmark.py.")
    y = iy + 24
    for n in notes:
        V.text(img, "-", (ix + 12, y), 0.42, V.BAD, 1)
        V.text(img, n, (ix + 26, y), 0.42, V.TEXT_DIM, 1)
        y += 21
    V.text(img, "reproduce:  python -m prism.deploy.export_onnx  ->  "
                "python -m prism.deploy.quantize  ->  python -m prism.deploy.benchmark",
           (ix + 12, iy + ih - 12), 0.4, V.ACCENT2, 1)

    V.footer(img, _W, _H,
             left="prism - camera-primary navigation perception - deployment track",
             right="output/benchmarks.json holds every raw measurement, "
                   "output/quantization.json every accuracy delta")
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    print(f"wrote {path}")
    return path


def _quant_rows(report: dict) -> list[dict]:
    """Compact per-model quantization summary read from output/quantization.json."""
    p = OUT_DIR / "quantization.json"
    if not p.exists():
        return []
    try:
        q = json.loads(p.read_text())
    except Exception:
        return []
    short = {"depth_anything_v2_small": "depth", "pidnet_s_terrain": "terrain",
             "trav_unc_shared": "trav+unc", "vpr_gem_mnv3": "vpr", "world_model": "world"}
    # (accuracy key, format, "bad above" threshold or None for the mIoU-style keys)
    keymap = [("median_abs_rel", "AbsRel {:.2%}", 0.02),
              ("mIoU_vs_fp32", "mIoU {:.3f}", None),
              ("mean_cosine_similarity", "cos {:.4f}", None),
              ("collision_risk_mae", "riskMAE {:.4f}", 0.01)]
    rows = []
    for m in q.get("models", []):
        if m.get("status") != "ok":
            continue
        base = m.get("fp32_mb") or 1.0
        row = {"model": short.get(m["name"], m["name"]),
               "fp32_size": f"fp32 {base:.1f} MB", "cells": {}}
        for vname in ("fp16", "int8_dynamic", "int8_static"):
            v = m.get("variants", {}).get(vname, {})
            if v.get("status") != "ok":
                row["cells"][vname] = {"acc": "unsupported", "size": "-", "col": V.BAD}
                continue
            acc = v.get("accuracy", {})
            txt, col = "-", V.OK
            for k, fmt, warn in keymap:
                if k in acc:
                    txt = fmt.format(acc[k])
                    if k == "mIoU_vs_fp32":
                        col = V.OK if acc[k] > 0.95 else (V.WARN if acc[k] > 0.85 else V.BAD)
                    elif k == "mean_cosine_similarity":
                        col = V.OK if acc[k] > 0.995 else (V.WARN if acc[k] > 0.98 else V.BAD)
                    elif warn is not None:
                        col = V.OK if acc[k] < warn else (V.WARN if acc[k] < 4 * warn else V.BAD)
                    break
            row["cells"][vname] = {
                "acc": txt,
                "size": f"{v['size_mb']:.1f} MB  {v['size_vs_fp32']*100:.0f}%",
                "col": col}
        rows.append(row)
    return rows


# ============================================================ markdown export

_SHORT = {"depth_anything_v2_small": "Depth Anything V2-S (518x518)",
          "pidnet_s_terrain": "PIDNet-S terrain (512x288)",
          "trav_unc_shared": "Shared trunk: trav + unc (320x180)",
          "vpr_gem_mnv3": "VPR GeM MobileNetV3-S (320x180)",
          "world_model": "World model, 6 actions x 6 steps"}


def markdown_table(report: dict) -> str:
    """The benchmark table as Markdown, so the README can never drift from the JSON."""
    hw, meth = report["hardware"], report["methodology"]
    L: list[str] = []
    L.append(f"Measured on **{hw['cpu']}** ({hw['cpu_physical_cores']} physical / "
             f"{hw['cpu_logical_processors']} logical cores, {hw['ram_gb']} GB RAM) and "
             f"**{hw['gpu']}** ({hw['gpu_vram_gb']} GB), {hw['os'].split('(')[0].strip()}, "
             f"Python {hw['python']}, torch {hw['torch']}, onnxruntime {hw['onnxruntime']}.")
    L.append("")
    L.append(f"Median of **{meth['timed_iters']} timed iterations** after "
             f"{meth['warmup_iters']} warm-up iterations; p95 in brackets. ONNX Runtime "
             f"`intra_op_num_threads` and torch CPU threads are both pinned to "
             f"**{meth['ort_intra_op_num_threads']}**. CUDA work is synchronised inside "
             f"the timing loop. **The GPU phase completes and the CUDA cache is freed "
             f"before any CPU timing starts** - this GPU is shared with other jobs, and "
             f"interleaving would make the CPU column meaningless. Inputs are real clip "
             f"frames.")
    L.append("")
    L.append("### Per-network latency (ms, median [p95])")
    L.append("")
    L.append("| Model | PyTorch GPU fp16 | PyTorch GPU fp32 | PyTorch CPU fp32 | "
             "ORT CPU fp32 | ORT CPU int8 static | ORT CPU int8 dynamic | ORT CPU fp16 |")
    L.append("|---|---|---|---|---|---|---|---|")
    picks = [("cuda", "fp16", "pytorch"), ("cuda", "fp32", "pytorch"),
             ("cpu", "fp32", "pytorch"), ("cpu", "fp32", "onnxruntime"),
             ("cpu", "int8_static", "onnxruntime"),
             ("cpu", "int8_dynamic", "onnxruntime"), ("cpu", "fp16", "onnxruntime")]
    order = list(_SHORT)
    keys = [k for k in order if k in report["models"]] + \
           [k for k in report["models"] if k not in order]
    for k in keys:
        cells = []
        for dev, dt, rt in picks:
            r = next((q for q in report["models"][k]
                      if q.get("device") == dev and q.get("dtype") == dt
                      and q.get("runtime") == rt and q.get("status") == "ok"), None)
            cells.append("-" if r is None else f"{r['median_ms']:.1f} [{r['p95_ms']:.1f}]")
        L.append(f"| {_SHORT.get(k, k)} | " + " | ".join(cells) + " |")
    for k, v in report["availability"].items():
        if not v.get("checkpoint_present", True):
            L.append(f"| {_SHORT.get(k, k)} | *missing - "
                     f"{v.get('missing_reason', 'checkpoint absent')}* | | | | | | |")
    L.append("")

    L.append("### Non-network stages (CPU, ms)")
    L.append("")
    L.append("| Stage | What it is | median | p95 |")
    L.append("|---|---|---|---|")
    for st in report["pipeline_stages"]:
        if st.get("status") != "ok":
            continue
        L.append(f"| `{st['stage']}` | {st.get('desc', '')} | {st['median_ms']:.2f} | "
                 f"{st['p95_ms']:.2f} |")
    L.append("")

    fb = report["frame_budget"]
    L.append("### One-frame budget (sum of medians, serial upper bound)")
    L.append("")
    L.append("| Deployment | Networks | + non-network | fps | 30 fps met? |")
    L.append("|---|---|---|---|---|")
    labels = {"gpu_fp16_pytorch": "GPU fp16 (PyTorch)",
              "cpu_fp32_onnxruntime": "CPU fp32 (ONNX Runtime, 4 threads)",
              "cpu_int8_onnxruntime": "CPU int8 (ONNX Runtime, 4 threads)"}
    for key, lab in labels.items():
        st = fb["stories"].get(key)
        if st is None:
            continue
        miss = (" *(incomplete: missing " + ", ".join(st["missing_networks"]) + ")*"
                if st["missing_networks"] else "")
        L.append(f"| {lab}{miss} | {st['networks_total_ms']:.1f} ms | "
                 f"{st['plus_non_network_ms']:.1f} ms | {st['fps']:.1f} | "
                 f"{'yes' if st['realtime_30fps'] else '**no**'} |")
    L.append("")
    L.append(f"Non-network stages alone cost **{fb['non_network_stages_ms']:.1f} ms** of "
             f"every frame. A 30 fps budget is 33.3 ms.")
    L.append("")

    sw = report.get("ort_thread_scaling", {})
    ok = [r for r in sw.get("rows", []) if r.get("status") == "ok"]
    if ok:
        L.append(f"### ONNX Runtime CPU thread scaling - `{sw['model']}`, fp32")
        L.append("")
        L.append("| threads | median ms | speedup vs 1 thread |")
        L.append("|---|---|---|")
        base = ok[0]["median_ms"]
        for r in ok:
            L.append(f"| {r['threads']} | {r['median_ms']:.1f} | "
                     f"x{base / max(r['median_ms'], 1e-9):.2f} |")
        L.append("")

    jp = report["jetson_projection"]
    L.append("### Jetson Orin Nano - **PROJECTED, NOT MEASURED**")
    L.append("")
    L.append(f"> **There is no NVIDIA Jetson in this environment and nothing below was "
             f"run on one.** Target: {jp['target']}. CPU multiplier "
             f"x{jp['cpu_multiplier']['central']} (band {jp['cpu_multiplier']['low']}-"
             f"{jp['cpu_multiplier']['high']}), GPU multiplier "
             f"x{jp['gpu_multiplier']['central']} (band {jp['gpu_multiplier']['low']}-"
             f"{jp['gpu_multiplier']['high']}). No TensorRT engine was built and memory "
             f"bandwidth is not modelled, so these are estimates with wide error bars and "
             f"must never be quoted as prism results. The full reasoning is in "
             f"`output/benchmarks.json` under `jetson_projection.reasoning`.")
    L.append("")
    L.append("| Model | measured here (CPU int8 / GPU fp16) | projected Orin Nano CPU | "
             "projected Orin Nano GPU |")
    L.append("|---|---|---|---|")
    for row in jp["rows"]:
        e, b = row["estimated"], row["basis"]
        cpu = (f"~{e['orin_nano_cpu_ms_central']} ms "
               f"[{e['orin_nano_cpu_ms_band'][0]}-{e['orin_nano_cpu_ms_band'][1]}]"
               if "orin_nano_cpu_ms_central" in e else "-")
        gpu = (f"~{e['orin_nano_gpu_ms_central']} ms "
               f"[{e['orin_nano_gpu_ms_band'][0]}-{e['orin_nano_gpu_ms_band'][1]}]"
               if "orin_nano_gpu_ms_central" in e else "-")
        L.append(f"| {_SHORT.get(row['model'], row['model'])} | "
                 f"{b.get('cpu_measured_ms', '-')} / {b.get('gpu_measured_ms', '-')} ms | "
                 f"{cpu} | {gpu} |")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--render-only", action="store_true",
                    help="redraw output/benchmarks.png from the existing JSON")
    ap.add_argument("--markdown", action="store_true",
                    help="print the benchmark table as Markdown (for the README)")
    a = ap.parse_args()
    if a.markdown:
        print(markdown_table(json.loads(BENCH_JSON.read_text())))
    elif a.render_only:
        render(json.loads(BENCH_JSON.read_text()))
    else:
        run(a.iters, a.warmup, a.threads, a.skip_gpu)
