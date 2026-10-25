"""Localization backend selection for the offline replay prototype.

stella_vslam, Basalt and RTAB-Map are evaluated as optional candidates.  Only
one operational backend is selected for the integrated runner; the existing
OpenCV tracking-only motion plus read-only COLMAP sparse diagnostics remain the
fallback when SLAM packages cannot run on the host.

Basalt is recorded as *blocked* rather than adopted: its VIO pipeline requires
synchronised inertial measurements and a camera-IMU calibration, neither of
which exists in the replay clips.  Running its monocular mode would not justify
the "visual-inertial" label, so that shortcut is explicitly refused.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Any


SELECTED_BACKEND = "opencv_colmap_diagnostic"


def _normalize_probe_text(value: str) -> str:
    text = value.strip()
    if "\x00" in text:
        try:
            text = text.encode("utf-16-le", errors="surrogatepass").decode("utf-16-le", errors="replace")
        except UnicodeError:
            text = text.replace("\x00", "")
    return text.strip()


def _probe_wsl() -> dict[str, Any]:
    executable = shutil.which("wsl")
    if executable is None:
        return {"status": "UNAVAILABLE", "reason": "wsl.exe not found on PATH"}
    try:
        completed = subprocess.run(
            [executable, "--status"],
            capture_output=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "UNAVAILABLE", "reason": f"WSL probe failed: {type(error).__name__}: {error}"}
    raw = completed.stdout or completed.stderr or b""
    try:
        detail = raw.decode("utf-16-le").strip()
    except UnicodeError:
        detail = _normalize_probe_text(raw.decode("utf-8", errors="replace"))
    if completed.returncode != 0:
        return {
            "status": "UNAVAILABLE",
            "reason": "wsl --status returned non-zero; WSL is not configured for stella_vslam builds",
            "detail": detail[:500] or None,
        }
    return {"status": "DETECTED", "detail": detail[:500] or None}


def localization_backend_report(*, clip_has_rgb_only: bool = True) -> dict[str, Any]:
    """Return auditable candidate status without starting SLAM binaries."""
    wsl = _probe_wsl()
    stella_blockers = [
        "Official stella_vslam Windows guidance expects WSL2 and a C++ build chain (g2o, SuiteSparse, FBoW).",
        "No isolated stella_vslam build or vocabulary bundle is vendored under tools/external/.",
    ]
    if wsl["status"] != "DETECTED":
        stella_blockers.append(f"WSL probe: {wsl.get('reason', wsl['status'])}")

    rtab_blockers = [
        "RTAB-Map CPU workflows for this prototype require RGB-D or supported sensor logs; clip input is monocular RGB only.",
        "ROS/GPU dependencies were not installed for this bounded offline runner.",
        "Arbitrary-scale COLMAP sparse points must not be converted into synthetic depth images.",
    ]
    if clip_has_rgb_only:
        rtab_blockers.append("No depth stream or metric pose stream is bundled with clips/clip_01.mp4.")

    basalt_blockers = [
        "Basalt is a visual-inertial pipeline: it needs synchronised IMU samples and a camera-IMU calibration.",
        "The replay clips contain monocular RGB frames only; there is no IMU stream and no calibration.",
        "Vision-only operation would not be visual-inertial SLAM and would not supply metric scale; using it under the VIO label is refused.",
        "No Windows binary is vendored under tools/external/; building Basalt needs a C++ toolchain, Eigen, and its own calibration format.",
    ]

    return {
        "selected_backend": SELECTED_BACKEND,
        "selection_policy": "Exactly one operational localization path in the integrated runner; SLAM candidates are probed independently and remain optional.",
        "sensor_inputs": {
            "monocular_rgb": True,
            "depth": not clip_has_rgb_only,
            "imu": False,
            "camera_imu_calibration": False,
            "note": "Sensor availability is a fact about the bundled clips, not an inference from installed packages.",
        },
        "candidates": {
            "stella_vslam": {
                "status": "BLOCKED_MISSING_INPUT",
                "license": "BSD-2-Clause (upstream stella-cv/stella_vslam)",
                "blockers": stella_blockers,
                "wsl_probe": wsl,
                "export_format": "not produced on this host",
            },
            "rtabmap": {
                "status": "BLOCKED_MISSING_INPUT",
                "license": "BSD-3-Clause (upstream introlab/rtabmap)",
                "blockers": rtab_blockers,
                "export_format": "not produced on this host",
            },
            "basalt": {
                "status": "BLOCKED_MISSING_INPUT",
                "license": "BSD-3-Clause (upstream VladyslavUsenko/basalt)",
                "capability": "visual-inertial odometry / SLAM",
                "blockers": basalt_blockers,
                "export_format": "not produced on this host",
            },
            "opencv_colmap_diagnostic": {
                "status": "ACTIVE",
                "components": [
                    "prism.visual_odometry_replay (OpenCV ORB; tracking-only without measured calibration)",
                    "prism.integrations.geometry sparse COLMAP pose QA (exact manifest frame identity only)",
                ],
                "coordinate_frame": "camera/image and COLMAP reconstruction world; metric scale unverified",
                "trajectory_metric_claim": False,
            },
        },
    }
