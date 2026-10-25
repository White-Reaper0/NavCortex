"""Bounded COLMAP sparse-reconstruction probe.

This tool is optional and is not part of the navigation runtime.  It extracts a
small, evenly spaced frame set from one clip, runs COLMAP feature extraction,
exhaustive matching and sparse mapping on the CPU, and preserves the native
COLMAP database/model plus a command and configuration log.

The COLMAP binary is external and is not vendored or redistributed here.  The
official COLMAP source is BSD-3-Clause, but the official binary bundles the
separately licensed LSD component, which is AGPL-3.0.  Treat the binary as an
optional local evaluation tool and do not redistribute it with this repository.

This probe does not run dense stereo, fusion, Poisson reconstruction or meshing.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import time

import cv2
import numpy as np


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _evenly_spaced(count: int, total: int) -> list[int]:
    if total <= 0:
        raise ValueError("video reports no frames")
    if count < 1:
        raise ValueError("frame count must be positive")
    if count >= total:
        return list(range(total))
    values = np.linspace(0, total - 1, count)
    return sorted({int(round(value)) for value in values})


def extract_frames(
    video: Path,
    output: Path,
    count: int,
    width: int,
    height: int,
    overwrite: bool,
) -> dict:
    frames_dir = output / "frames"
    manifest_path = output / "frames_manifest.json"
    if overwrite and frames_dir.exists():
        shutil.rmtree(frames_dir)
    if frames_dir.exists() and any(frames_dir.iterdir()):
        raise RuntimeError(
            f"frame directory already exists and is not empty: {frames_dir}; "
            "use --overwrite to recreate it")
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError("video reports no positive finite FPS")
    if source_width <= 0 or source_height <= 0:
        capture.release()
        raise RuntimeError("video reports invalid dimensions")
    if abs(width / height - source_width / source_height) > 0.01 * (
            source_width / source_height):
        capture.release()
        raise RuntimeError(
            "requested frame size does not preserve the source aspect ratio")

    wanted = set(_evenly_spaced(count, total))
    records: list[dict] = []
    frame_index = 0
    try:
        while frame_index < total:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index in wanted:
                resized = _resize_frame(frame, width, height)
                name = f"frame_{frame_index:04d}.png"
                destination = frames_dir / name
                if not cv2.imwrite(str(destination), resized):
                    raise RuntimeError(f"could not write frame: {destination}")
                records.append({
                    "file": name,
                    "source_frame_index": frame_index,
                    "source_time_s": frame_index / fps,
                    "sha256": _sha256_file(destination),
                    "width": int(resized.shape[1]),
                    "height": int(resized.shape[0]),
                })
            frame_index += 1
    finally:
        capture.release()

    missing = wanted - {record["source_frame_index"] for record in records}
    if missing:
        raise RuntimeError(f"could not decode requested frame indices: {sorted(missing)}")

    manifest = {
        "source_video": str(video),
        "source_sha256": _sha256_file(video),
        "source_width": source_width,
        "source_height": source_height,
        "source_frame_count": total,
        "source_fps": fps,
        "selection": "evenly spaced decode-order frames including first and last",
        "requested_frames": count,
        "extracted_frames": len(records),
        "output_width": width,
        "output_height": height,
        "resize_interpolation": "INTER_AREA",
        "frames": records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _run_step(
    name: str,
    command: list[str],
    cwd: Path,
    logs_dir: Path,
    timeout_s: int,
) -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", timeout=timeout_s,
        check=False)
    elapsed = time.perf_counter() - started
    log_path = logs_dir / f"{name}.log"
    log_path.write_text(
        "$ " + " ".join(command) + "\n\n" + completed.stdout,
        encoding="utf-8")
    return {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "elapsed_s": elapsed,
        "log": str(log_path),
        "stdout": completed.stdout,
    }


def _parse_model_analysis(text: str) -> dict:
    patterns = {
        "registered_images": r"Registered images:\s*(\d+)",
        "points": r"Points:\s*(\d+)",
        "mean_reprojection_error_px": r"Mean reprojection error:\s*([0-9.]+)px",
    }
    parsed = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match is None:
            parsed[key] = None
        elif key == "mean_reprojection_error_px":
            parsed[key] = float(match.group(1))
        else:
            parsed[key] = int(match.group(1))
    return parsed


def _resolve_image_ids(database: Path, name1: str, name2: str) -> tuple[int, int]:
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "select image_id, name from images where name in (?, ?)",
            (name1, name2)).fetchall()
    finally:
        connection.close()
    mapping = {name: image_id for image_id, name in rows}
    missing = [name for name in (name1, name2) if name not in mapping]
    if missing:
        raise ValueError(f"initial frame names not found in database: {missing}")
    if mapping[name1] == mapping[name2]:
        raise ValueError("initial frame names must be different")
    return mapping[name1], mapping[name2]


def _set_free_focal_length(database: Path) -> None:
    connection = sqlite3.connect(database)
    try:
        connection.execute("update cameras set prior_focal_length = 0")
        connection.commit()
    finally:
        connection.close()


def _set_camera_params(
    database: Path,
    params: str,
    free_focal_length: bool,
) -> None:
    parts = [float(value) for value in params.split(",")]
    if len(parts) != 4:
        raise ValueError("camera_params must be f,cx,cy,k")
    focal, cx, cy, k = parts
    prior = 0 if free_focal_length else 1
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "update cameras set model = ?, params = ?, prior_focal_length = ? "
            "where camera_id = 1",
            (2, struct.pack("<4d", focal, cx, cy, k), prior))
        connection.commit()
    finally:
        connection.close()


def run_colmap(
    colmap_exe: Path,
    output: Path,
    frames_dir: Path,
    threads: int,
    timeout_s: int,
    overwrite: bool,
    camera_model: str,
    camera_params: str | None,
    default_focal_length_factor: float | None,
    init_image_id1: int | None,
    init_image_id2: int | None,
    init_image_name1: str | None,
    init_image_name2: str | None,
    free_focal_length: bool,
    reuse_database: bool = False,
    source_database: str | None = None,
) -> dict:
    database = output / "database.db"
    sparse_dir = output / "sparse"
    logs_dir = output / "logs"
    if overwrite:
        if database.exists():
            database.unlink()
        if sparse_dir.exists():
            shutil.rmtree(sparse_dir)
    if database.exists() or (sparse_dir.exists() and any(sparse_dir.iterdir())):
        raise RuntimeError(
            f"COLMAP database or sparse output already exists under {output}; "
            "use --overwrite to recreate them")
    sparse_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    version = subprocess.run(
        [str(colmap_exe), "version"], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        errors="replace", check=True).stdout.strip()

    feature_command = [
        str(colmap_exe), "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(frames_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", camera_model,
        "--FeatureExtraction.use_gpu", "0",
        "--FeatureExtraction.num_threads", str(threads),
    ]
    if camera_params:
        feature_command.extend(["--ImageReader.camera_params", camera_params])
    if default_focal_length_factor is not None:
        feature_command.extend([
            "--ImageReader.default_focal_length_factor",
            str(default_focal_length_factor)])

    results: list[dict] = []
    feature_result = None
    if reuse_database:
        source = Path(source_database) if source_database is not None else None
        if source is None or not source.is_file():
            raise ValueError("--source-database is required with --reuse-database")
        shutil.copyfile(source, database)
        if camera_params:
            _set_camera_params(database, camera_params, free_focal_length)
        elif free_focal_length:
            _set_free_focal_length(database)
        results.append({
            "name": "source_database_reuse",
            "command": ["copy", str(source), str(database)],
            "returncode": 0,
            "elapsed_s": 0.0,
            "log": str(logs_dir / "source_database_reuse.log"),
        })
    else:
        feature_result = _run_step(
        "feature_extractor", feature_command, output, logs_dir, timeout_s)
        results.append(feature_result)

    if feature_result is None or feature_result["returncode"] == 0:
        if free_focal_length:
            _set_free_focal_length(database)
        if init_image_name1 is not None or init_image_name2 is not None:
            if init_image_name1 is None or init_image_name2 is None:
                raise ValueError("both initial frame names must be provided together")
            init_image_id1, init_image_id2 = _resolve_image_ids(
                database, init_image_name1, init_image_name2)
        if init_image_id1 is not None or init_image_id2 is not None:
            if init_image_id1 is None or init_image_id2 is None:
                raise ValueError("both initial image ids must be provided together")
            if init_image_id1 == init_image_id2:
                raise ValueError("initial image ids must be different")

        matcher_command = [
            str(colmap_exe), "exhaustive_matcher",
            "--database_path", str(database),
            "--FeatureMatching.use_gpu", "0",
            "--FeatureMatching.num_threads", str(threads),
        ]
        mapper_command = [
            str(colmap_exe), "mapper",
            "--database_path", str(database),
            "--image_path", str(frames_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.num_threads", str(threads),
            "--Mapper.multiple_models", "0",
            "--Mapper.extract_colors", "0",
        ]
        if init_image_id1 is not None:
            mapper_command.extend([
                "--Mapper.init_image_id1", str(init_image_id1),
                "--Mapper.init_image_id2", str(init_image_id2),
            ])
        if reuse_database:
            result = _run_step(
                "mapper", mapper_command, output, logs_dir, timeout_s)
            results.append(result)
        else:
            for name, command in (("exhaustive_matcher", matcher_command),
                                  ("mapper", mapper_command)):
                result = _run_step(name, command, output, logs_dir, timeout_s)
                results.append(result)
                if result["returncode"] != 0:
                    break

    analysis = None
    model_metrics = {}
    model_path = sparse_dir / "0"
    if model_path.is_dir():
        analysis_result = _run_step(
            "model_analyzer",
            [str(colmap_exe), "model_analyzer", "--path", str(model_path)],
            output, logs_dir, timeout_s)
        analysis_text = analysis_result.pop("stdout", "")
        results.append(analysis_result)
        analysis = analysis_result
        model_metrics = _parse_model_analysis(analysis_text)
        frame_count = len(list(frames_dir.glob("*.png")))
        model_metrics["frame_count"] = frame_count
        model_metrics["minimum_registered_images"] = max(
            3, math.ceil(0.5 * frame_count))
        model_metrics["usable"] = bool(
            model_metrics.get("registered_images") is not None
            and model_metrics.get("points") is not None
            and model_metrics["registered_images"]
            >= model_metrics["minimum_registered_images"]
            and model_metrics["points"] >= 100)

    for result in results:
        result.pop("stdout", None)

    return {
        "colmap_version": version,
        "colmap_exe": str(colmap_exe),
        "colmap_exe_sha256": _sha256_file(colmap_exe),
        "database": str(database),
        "sparse_dir": str(sparse_dir),
        "camera_model": camera_model,
        "camera_params": camera_params,
        "default_focal_length_factor": default_focal_length_factor,
        "init_image_id1": init_image_id1,
        "init_image_id2": init_image_id2,
        "init_image_name1": init_image_name1,
        "init_image_name2": init_image_name2,
        "free_focal_length": free_focal_length,
        "reuse_database": reuse_database,
        "source_database": str(source_database) if source_database else None,
        "model_path": str(model_path) if model_path.is_dir() else None,
        "model_produced": model_path.is_dir(),
        "model_metrics": model_metrics,
        "model_analyzer": analysis,
        "steps": results,
        "success": bool(model_path.is_dir())
        and all(step["returncode"] == 0 for step in results)
        and bool(model_metrics.get("usable")),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract a bounded frame set and run COLMAP sparse mapping.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--colmap-exe", required=True)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--camera-model", default="SIMPLE_RADIAL")
    parser.add_argument("--camera-params", default=None,
                        help="initial COLMAP camera parameters, e.g. f,cx,cy,k")
    parser.add_argument("--default-focal-length-factor", type=float, default=None)
    parser.add_argument("--init-image-id1", type=int, default=None)
    parser.add_argument("--init-image-id2", type=int, default=None)
    parser.add_argument("--init-image-name1", default=None)
    parser.add_argument("--init-image-name2", default=None)
    parser.add_argument("--free-focal-length", action="store_true",
                        help="let COLMAP refine the supplied initial focal length")
    parser.add_argument("--reuse-database", action="store_true",
                        help="reuse an existing COLMAP database instead of "
                        "re-running feature extraction and matching")
    parser.add_argument("--source-database", default=None,
                        help="COLMAP database.db to reuse with --reuse-database")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if not 30 <= args.frames <= 60:
        print("frames must be between 30 and 60", file=sys.stderr)
        return 2
    if args.width < 1 or args.height < 1:
        print("width and height must be positive", file=sys.stderr)
        return 2
    if args.threads < 1 or args.timeout_s < 1:
        print("threads and timeout-s must be positive", file=sys.stderr)
        return 2
    if (args.init_image_name1 is None) != (args.init_image_name2 is None):
        print("both initial frame names must be provided together", file=sys.stderr)
        return 2
    if (args.init_image_id1 is None) != (args.init_image_id2 is None):
        print("both initial image ids must be provided together", file=sys.stderr)
        return 2
    if args.reuse_database and not args.source_database:
        print("--source-database is required with --reuse-database", file=sys.stderr)
        return 2

    video = Path(args.video).resolve()
    output = Path(args.output).resolve()
    colmap_exe = Path(args.colmap_exe).resolve()
    if not video.is_file():
        print(f"video does not exist: {video}", file=sys.stderr)
        return 2
    if not colmap_exe.is_file():
        print(f"COLMAP executable does not exist: {colmap_exe}", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)

    manifest = extract_frames(
        video, output, args.frames, args.width, args.height, args.overwrite)
    colmap_result = run_colmap(
        colmap_exe, output, output / "frames", args.threads, args.timeout_s,
        args.overwrite, args.camera_model, args.camera_params,
        args.default_focal_length_factor, args.init_image_id1,
        args.init_image_id2, args.init_image_name1, args.init_image_name2,
        args.free_focal_length, args.reuse_database, args.source_database)

    config = {
        "tool": "tools/colmap_sparse_probe.py",
        "scope": "sparse reconstruction only; no dense, fusion or meshing",
        "source_video": str(video),
        "frames_manifest": str(output / "frames_manifest.json"),
        "extracted_frames": manifest["extracted_frames"],
        "output_width": manifest["output_width"],
        "output_height": manifest["output_height"],
        "threads": args.threads,
        "timeout_s": args.timeout_s,
        "camera_model": args.camera_model,
        "camera_params": args.camera_params,
        "camera_params_provenance": (
            "initialization prior for feasibility; not a measured calibration"
            if args.camera_params else None),
        "default_focal_length_factor": args.default_focal_length_factor,
        "init_image_id1": args.init_image_id1,
        "init_image_id2": args.init_image_id2,
        "init_image_name1": args.init_image_name1,
        "init_image_name2": args.init_image_name2,
        "free_focal_length": args.free_focal_length,
        "reuse_database": args.reuse_database,
        "source_database": args.source_database,
        "colmap": colmap_result,
        "license": {
            "colmap": "new BSD (3-clause text)",
            "bundled_lsd": "AGPL-3.0",
            "distribution": "external optional binary; not vendored or redistributed",
        },
        "notes": [
            "COLMAP refines the supplied initial camera model from these frames.",
            "No checkerboard calibration artifact is supplied to this probe.",
            "If camera_params is set, those initial values are a prior, not a "
            "measured calibration.",
            "If init_image_id1/2 are set, that pair was selected for feasibility "
            "on this clip, not from ground truth.",
            "Monocular sparse reconstruction has arbitrary scale.",
            "Fresh feature extraction and matching are not bit-identical between "
            "runs; use --reuse-database with a stored database for deterministic "
            "re-runs of the same probe.",
        ],
    }
    (output / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "output": str(output),
        "frames": manifest["extracted_frames"],
        "colmap_success": colmap_result["success"],
        "model_path": colmap_result["model_path"],
        "model_metrics": colmap_result["model_metrics"],
    }, indent=2))
    return 0 if colmap_result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
