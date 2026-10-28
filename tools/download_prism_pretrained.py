"""Idempotent downloader for the pretrained assets P.R.I.S.M. is allowed to use.

Downloads, verifies and records:

* Depth Anything V2-Small (HF ``depth-anything/Depth-Anything-V2-Small-hf``)
  -> ``model_assets/pretrained/depth_anything_v2_small_hf/``
* SegFormer-B0 ADE20K (HF ``nvidia/segformer-b0-finetuned-ade-512-512``)
  -> ``model_assets/pretrained/segformer_b0_ade_512_512/``
* MobileNetV3-Small ImageNet backbone (official TorchVision weights)
  -> the local TorchVision hub cache (``~/.cache/torch/hub/checkpoints``)

It also verifies the existing LibreYOLO checkpoint and the local COLMAP
executable and *never* downloads YOLOP (unverified weight provenance).

Rules this tool enforces on itself:
* only the approved repositories/files above are ever fetched;
* one weight format per HF model (``model.safetensors`` when the repo has one,
  otherwise ``pytorch_model.bin``) - no redundant duplicates;
* already-valid files are reused (content SHA-256 check), not re-downloaded;
* model ID, source URL, resolved revision, license, sizes and hashes are
  recorded in ``model_assets/pretrained/manifest.json``;
* weights stay out of Git (``.gitignore``), the manifest does not;
* failures print actionable messages (network, disk space, corruption).

Usage:
    .venv/Scripts/python.exe tools/download_prism_pretrained.py
    .venv/Scripts/python.exe tools/download_prism_pretrained.py --verify-only
    .venv/Scripts/python.exe tools/download_prism_pretrained.py --only depth
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PRETRAINED_ROOT = PROJECT_ROOT / "model_assets" / "pretrained"
MANIFEST_PATH = PRETRAINED_ROOT / "manifest.json"
MANIFEST_SCHEMA = "prism-pretrained-manifest/v1"

HF_ENDPOINT = "https://huggingface.co"

DEPTH_KEY = "depth_anything_v2_small_hf"
DEPTH_REPO = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_DIR = PRETRAINED_ROOT / "depth_anything_v2_small_hf"

SEGMENTATION_KEY = "segformer_b0_ade_512_512"
SEGMENTATION_REPO = "nvidia/segformer-b0-finetuned-ade-512-512"
SEGMENTATION_DIR = PRETRAINED_ROOT / "segformer_b0_ade_512_512"

MOBILENET_KEY = "mobilenet_v3_small_imagenet1k_v1"
MOBILENET_URL = "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"
MOBILENET_FILE = "mobilenet_v3_small-047dcff4.pth"
# torchvision's own convention: the filename suffix is the first 8 hex chars
# of the checkpoint SHA-256, and load_state_dict_from_url(check_hash=True)
# validates exactly this prefix.
MOBILENET_HASH_PREFIX = "047dcff4"

LIBREYOLO_KEY = "libreyolo_detector"
LIBREYOLO_PATH = PROJECT_ROOT / "model_assets" / "libreyolo" / "LibreYOLO9t.pt"

COLMAP_EXE = (PROJECT_ROOT / "tools" / "external" / "colmap" / "4.2.1-nocuda" / "bin"
              / "colmap.exe")

YOLOP_KEY = "yolop"
YOLOP_NOTE = ("optional evaluation-only asset; not downloaded: weight licensing/provenance "
              "is documented as unverified in docs/integration_manifest.md")

# Non-weight files worth keeping next to the weights for provenance.
HF_CONFIG_FILES = ("config.json", "preprocessor_config.json", "processor_config.json",
                   "image_processor_config.json", "LICENSE", "LICENSE.md", "README.md")

# Terms verified from the upstream license documents on 2026-10-25, kept here so
# the manifest records substance rather than the bare HF "other" tag.
LICENSE_TERMS = {
    DEPTH_REPO: {
        "id": "apache-2.0",
        "title": "Apache License 2.0",
        "terms": "permissive, commercial use allowed; attribution and NOTICE retention required",
    },
    SEGMENTATION_REPO: {
        "id": "other",
        "title": "NVIDIA Source Code License for SegFormer",
        "terms": "non-commercial; notice retention required; the HF model card links the "
                 "license at github.com/NVlabs/SegFormer/blob/master/LICENSE",
    },
}

MIN_FREE_DISK_BYTES = 400 * 1024 * 1024


def _sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.is_file():
        try:
            data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if data.get("schema") == MANIFEST_SCHEMA:
                return data
        except (OSError, json.JSONDecodeError):
            pass
    return {"schema": MANIFEST_SCHEMA, "assets": {}}


def _save_manifest(manifest: dict[str, Any]) -> None:
    PRETRAINED_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")


def _check_disk_space(label: str) -> str | None:
    PRETRAINED_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        free = shutil.disk_usage(PRETRAINED_ROOT).free
    except OSError as error:
        return f"could not check free disk space for {label}: {error}"
    if free < MIN_FREE_DISK_BYTES:
        return (f"insufficient disk space for {label}: {free / 2**20:.0f} MiB free, "
                f"at least {MIN_FREE_DISK_BYTES / 2**20:.0f} MiB required")
    return None


class DownloadFailure(RuntimeError):
    """One asset failed; the message must be actionable."""


# --------------------------------------------------------------------------- HF assets

def _hf_weight_files(repo_files: list[str]) -> list[str]:
    """Exactly one weight file: safetensors preferred over .bin, never both."""
    weights = [name for name in repo_files
               if name in ("model.safetensors", "pytorch_model.bin")]
    if "model.safetensors" in weights:
        return ["model.safetensors"]
    return weights


def _hf_repo_state(repo_id: str) -> dict[str, Any]:
    """Network probe for revision + license. Raises DownloadFailure on error."""
    from huggingface_hub import model_info
    try:
        info = model_info(repo_id)
    except Exception as error:  # requests/urllib/HTTP errors all land here
        raise DownloadFailure(
            f"could not reach the Hugging Face API for {repo_id} "
            f"({type(error).__name__}: {error}). Check the network connection and retry; "
            "assets already verified locally are unaffected.") from error
    card = getattr(info, "card_data", None)
    license_id = getattr(card, "license", None) if card is not None else None
    terms = LICENSE_TERMS.get(repo_id, {})
    return {
        "revision": info.sha,
        "license": {
            "id": license_id,
            "title": terms.get("title"),
            "terms": terms.get("terms"),
            "source": f"model card metadata at {HF_ENDPOINT}/{repo_id}",
            "note": ("id/title/terms recorded from the repository's card and the upstream "
                     "license document (verified 2026-10-25); weights stay local and are "
                     "not redistributed."),
        },
    }


def _hf_files_valid(local_dir: Path, files: list[dict[str, Any]]) -> bool:
    for entry in files:
        path = local_dir / entry["file"]
        if not path.is_file() or path.stat().st_size != entry["size"]:
            return False
        if _sha256(path) != entry["sha256"]:
            return False
    return True


def download_hf_asset(key: str, repo_id: str, local_dir: Path,
                      manifest: dict[str, Any], *, download: bool = True) -> dict[str, Any]:
    """Fetch or reuse one HF model; returns the manifest entry."""
    previous = manifest["assets"].get(key, {})
    files = previous.get("files") or []
    if files and _hf_files_valid(local_dir, files):
        entry = dict(previous)
        entry["status"] = "reused"
        entry["last_checked_at"] = _now_iso()
        terms = LICENSE_TERMS.get(repo_id)
        if terms and isinstance(entry.get("license"), dict):
            entry["license"].setdefault("title", terms["title"])
            entry["license"].setdefault("terms", terms["terms"])
        return entry
    if not download:
        raise DownloadFailure(
            f"{repo_id}: local files under {local_dir} are missing or do not match the "
            "recorded hashes; run the downloader without --verify-only")

    space_error = _check_disk_space(repo_id)
    if space_error:
        raise DownloadFailure(space_error)

    from huggingface_hub import list_repo_files, snapshot_download
    try:
        repo_files = list_repo_files(repo_id)
    except Exception as error:
        raise DownloadFailure(
            f"could not list files of {repo_id} ({type(error).__name__}: {error}). "
            "Check the network connection and retry.") from error

    wanted = [name for name in HF_CONFIG_FILES if name in repo_files]
    wanted += _hf_weight_files(repo_files)
    if not any(name in wanted for name in ("model.safetensors", "pytorch_model.bin")):
        raise DownloadFailure(
            f"{repo_id}: no safetensors or pytorch .bin checkpoint found in the repository")

    state = _hf_repo_state(repo_id)
    try:
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir),
                          allow_patterns=wanted, revision=state["revision"])
    except Exception as error:
        raise DownloadFailure(
            f"download of {repo_id} failed ({type(error).__name__}: {error}). "
            "Re-run this script: interrupted transfers resume; files already verified "
            "are not re-downloaded.") from error

    records = []
    for name in wanted:
        path = local_dir / name
        if not path.is_file():
            raise DownloadFailure(f"{repo_id}: expected file {name} is missing after download")
        records.append({"file": name, "size": path.stat().st_size, "sha256": _sha256(path)})

    return {
        "model_id": repo_id,
        "source_url": f"{HF_ENDPOINT}/{repo_id}",
        "local_dir": str(local_dir.relative_to(PROJECT_ROOT)),
        "revision": state["revision"],
        "license": state["license"],
        "downloaded_at": _now_iso(),
        "files": records,
        "status": "downloaded",
        "weight_format": next((r["file"] for r in records
                               if r["file"].endswith((".safetensors", ".bin"))), None),
    }


def verify_hf_model_load(local_dir: Path, model_type: str) -> dict[str, Any]:
    """Instantiate the model from the local directory only - no network."""
    from transformers import AutoModelForDepthEstimation, SegformerForSemanticSegmentation
    started = time.perf_counter()
    if model_type == "depth":
        model = AutoModelForDepthEstimation.from_pretrained(str(local_dir),
                                                            local_files_only=True)
    else:
        model = SegformerForSemanticSegmentation.from_pretrained(str(local_dir),
                                                                 local_files_only=True)
    n_params = sum(p.numel() for p in model.parameters())
    return {"loaded": True, "n_params": n_params,
            "local_files_only": True, "device": "cpu",
            "load_s": round(time.perf_counter() - started, 2)}


# --------------------------------------------------------------------------- torchvision

def _torchvision_cache_dir() -> Path:
    import torch
    return Path(torch.hub.get_dir()) / "checkpoints"


def download_mobilenet(manifest: dict[str, Any], *, download: bool = True
                       ) -> dict[str, Any]:
    """Fetch the official TorchVision MobileNetV3-Small weights into the hub cache."""
    destination = _torchvision_cache_dir() / MOBILENET_FILE
    previous = manifest["assets"].get(MOBILENET_KEY, {})
    if destination.is_file():
        digest = _sha256(destination)
        if digest.startswith(MOBILENET_HASH_PREFIX):
            entry = dict(previous)
            entry.update({"file": str(destination), "size": destination.stat().st_size,
                          "sha256": digest, "status": "reused",
                          "last_checked_at": _now_iso()})
            return entry
        if not download:
            raise DownloadFailure(
                f"{destination} exists but its SHA-256 {digest[:8]} does not start with "
                f"the expected torchvision prefix {MOBILENET_HASH_PREFIX}; delete the "
                "corrupt file and re-run")
        destination.unlink()  # corrupt or foreign file: replace with the official one

    if not download:
        raise DownloadFailure(
            f"MobileNetV3-Small weights are not in the TorchVision cache "
            f"({_torchvision_cache_dir()}); run the downloader without --verify-only")

    space_error = _check_disk_space(MOBILENET_FILE)
    if space_error:
        raise DownloadFailure(space_error)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")
    try:
        with urllib.request.urlopen(MOBILENET_URL, timeout=60) as response, \
                temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle, 1024 * 1024)
        digest = _sha256(temporary)
        if not digest.startswith(MOBILENET_HASH_PREFIX):
            raise DownloadFailure(
                f"downloaded {MOBILENET_URL} has SHA-256 {digest[:8]}, expected prefix "
                f"{MOBILENET_HASH_PREFIX}: the file is corrupt or the URL changed. "
                "Nothing was kept.")
        temporary.replace(destination)
    except urllib.error.URLError as error:
        temporary.unlink(missing_ok=True)
        raise DownloadFailure(
            f"could not download {MOBILENET_URL} ({error}). Check the network "
            "connection and re-run; nothing was kept.") from error
    except DownloadFailure:
        temporary.unlink(missing_ok=True)
        raise
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise DownloadFailure(
            f"could not write {destination} ({error}). Check disk space and permissions; "
            "nothing was kept.") from error

    return {
        "model_id": "torchvision:mobilenet_v3_small",
        "weights_enum": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
        "source_url": MOBILENET_URL,
        "file": str(destination),
        "size": destination.stat().st_size,
        "sha256": _sha256(destination),
        "downloaded_at": _now_iso(),
        "status": "downloaded",
        "license": {
            "id": "torchvision pretrained weights (ImageNet-trained backbone)",
            "source": "https://download.pytorch.org/models/ - distributed with PyTorch/torchvision",
            "note": ("ImageNet classification backbone weights only; they provide no trained "
                     "P.R.I.S.M. terrain predictions. Not redistributed."),
        },
    }


def verify_mobilenet_load() -> dict[str, Any]:
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    started = time.perf_counter()
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    return {"loaded": True,
            "n_params": sum(p.numel() for p in model.parameters()),
            "cache_dir": str(_torchvision_cache_dir()),
            "load_s": round(time.perf_counter() - started, 2)}


# --------------------------------------------------------------------------- local assets

def verify_libreyolo() -> dict[str, Any]:
    """Presence + loadability of the existing local detector checkpoint. No download."""
    if not LIBREYOLO_PATH.is_file():
        raise DownloadFailure(
            f"expected the existing LibreYOLO checkpoint at {LIBREYOLO_PATH}; it is "
            "missing. Do not re-download blindly: restore it from its documented source "
            "and keep the recorded SHA-256 consistent.")
    from prism.libreyolo_provider import LibreYOLODetectionProvider
    from prism.belief_ledger import EvidenceRelation
    try:
        provider = LibreYOLODetectionProvider.from_checkpoint(
            LIBREYOLO_PATH, source="pretrained-asset-verification",
            hypothesis_id="verification:detection", relation=EvidenceRelation.SUPPORTS,
            expires_after_ns=1_000_000_000, confidence_threshold=0.25)
    except Exception as error:
        raise DownloadFailure(
            f"the existing checkpoint {LIBREYOLO_PATH} could not be loaded by "
            f"LibreYOLODetectionProvider ({type(error).__name__}: {error})") from error
    return {
        "file": str(LIBREYOLO_PATH.relative_to(PROJECT_ROOT)),
        "size": LIBREYOLO_PATH.stat().st_size,
        "sha256": _sha256(LIBREYOLO_PATH),
        "verification": {"loaded": True,
                         "loader": "prism.libreyolo_provider.LibreYOLODetectionProvider"},
        "status": "present",
        "note": "existing local asset, verified only - never re-downloaded by this tool",
    }


def check_colmap() -> dict[str, Any]:
    return {
        "status": "present" if COLMAP_EXE.is_file() else "not_found",
        "file": str(COLMAP_EXE.relative_to(PROJECT_ROOT)) if COLMAP_EXE.is_file()
        else str(COLMAP_EXE),
        "note": "local binary only; this project does not vendor or download COLMAP",
    }


def check_open3d() -> dict[str, Any]:
    try:
        import open3d  # noqa: F401
        return {"status": "available"}
    except ImportError as error:
        return {"status": "unavailable", "reason": str(error)}


# --------------------------------------------------------------------------- driver

def run(only: str | None = None, download: bool = True) -> int:
    manifest = _load_manifest()
    failures: list[str] = []
    plan = [
        (DEPTH_KEY, "depth",
         lambda: download_hf_asset(DEPTH_KEY, DEPTH_REPO, DEPTH_DIR, manifest,
                                   download=download)),
        (SEGMENTATION_KEY, "segmentation",
         lambda: download_hf_asset(SEGMENTATION_KEY, SEGMENTATION_REPO, SEGMENTATION_DIR,
                                   manifest, download=download)),
        (MOBILENET_KEY, "mobilenet",
         lambda: download_mobilenet(manifest, download=download)),
    ]
    for key, label, action in plan:
        if only is not None and only != label:
            continue
        try:
            manifest["assets"][key] = action()
            entry = manifest["assets"][key]
            print(f"[{label}] {entry['status']}: {entry.get('model_id', key)} "
                  f"-> {entry.get('local_dir') or entry.get('file')}")
        except DownloadFailure as error:
            failures.append(f"{label}: {error}")
            manifest["assets"][key] = {"status": "failed", "error": str(error)}
            print(f"[{label}] FAILED: {error}", file=sys.stderr)

    verifications: dict[str, Any] = {}
    for key, label, local_dir in ((DEPTH_KEY, "depth", DEPTH_DIR),
                                  (SEGMENTATION_KEY, "segmentation", SEGMENTATION_DIR)):
        if only is not None and only != label:
            continue
        if manifest["assets"].get(key, {}).get("status") in ("downloaded", "reused"):
            try:
                verifications[key] = verify_hf_model_load(local_dir, label)
                print(f"[{label}] loaded locally on CPU: "
                      f"{verifications[key]['n_params'] / 1e6:.2f} M parameters")
            except Exception as error:
                failures.append(f"{label}: model load failed: {type(error).__name__}: {error}")
                verifications[key] = {"loaded": False, "error": str(error)}
                print(f"[{label}] LOAD FAILED: {error}", file=sys.stderr)
    if only in (None, "mobilenet") and manifest["assets"].get(MOBILENET_KEY, {}).get(
            "status") in ("downloaded", "reused"):
        try:
            verifications[MOBILENET_KEY] = verify_mobilenet_load()
            print(f"[mobilenet] loaded from the TorchVision cache: "
                  f"{verifications[MOBILENET_KEY]['n_params'] / 1e6:.2f} M parameters")
        except Exception as error:
            failures.append(f"mobilenet: model load failed: {type(error).__name__}: {error}")
            verifications[MOBILENET_KEY] = {"loaded": False, "error": str(error)}
            print(f"[mobilenet] LOAD FAILED: {error}", file=sys.stderr)

    if only is None:
        manifest["assets"].pop("colmap.bat", None)  # stale key from a pre-rename run
        for key, action, label in ((LIBREYOLO_KEY, verify_libreyolo, "libreyolo"),
                                   (COLMAP_EXE.name.lower(), check_colmap, "colmap"),
                                   ("open3d", check_open3d, "open3d")):
            try:
                manifest["assets"][key] = action()
                print(f"[{label}] {manifest['assets'][key]['status']}")
            except DownloadFailure as error:
                failures.append(f"{label}: {error}")
                manifest["assets"][key] = {"status": "failed", "error": str(error)}
                print(f"[{label}] FAILED: {error}", file=sys.stderr)
        manifest["assets"][YOLOP_KEY] = {"status": "not_downloaded_optional", "note": YOLOP_NOTE}
        print("[yolop] not downloaded (optional, unverified provenance)")

    manifest["verifications"] = verifications
    manifest["updated_at"] = _now_iso()
    _save_manifest(manifest)
    print(f"manifest: {MANIFEST_PATH}")
    if failures:
        print("\nFAILURES:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_prism_pretrained",
        description="Download and verify the pretrained model assets P.R.I.S.M. uses. "
                    "Idempotent: valid files are reused, weights never enter Git history, "
                    "YOLOP is never fetched.")
    parser.add_argument("--only", choices=("depth", "segmentation", "mobilenet"), default=None,
                        help="restrict the run to one asset (default: all)")
    parser.add_argument("--verify-only", action="store_true",
                        help="no network: only check files against the manifest and load them")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(only=args.only, download=not args.verify_only)


if __name__ == "__main__":
    raise SystemExit(main())
