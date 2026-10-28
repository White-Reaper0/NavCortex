"""Resolve pretrained Hugging Face models to local directories.

Central rule: inference code prefers the weights under
``model_assets/pretrained/`` (installed by ``tools/download_prism_pretrained.py``
or an explicit path/environment override) and loads them ``local_files_only``,
so no P.R.I.S.M. entry point ever triggers a surprise network download. When no
local copy exists the repo id is returned and the caller still loads with
``local_files_only=True``: the shared HF cache may satisfy it, the network may
not.

Nothing in this module talks to the network.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
SEGMENTATION_MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"

DEPTH_ENV_VAR = "PRISM_DEPTH_MODEL_PATH"
SEGMENTATION_ENV_VAR = "PRISM_SEGMENTATION_MODEL_PATH"

_DEFAULT_DIRS = {
    DEPTH_MODEL_ID: PROJECT_ROOT / "model_assets" / "pretrained"
    / "depth_anything_v2_small_hf",
    SEGMENTATION_MODEL_ID: PROJECT_ROOT / "model_assets" / "pretrained"
    / "segformer_b0_ade_512_512",
}
_ENV_VARS = {
    DEPTH_MODEL_ID: DEPTH_ENV_VAR,
    SEGMENTATION_MODEL_ID: SEGMENTATION_ENV_VAR,
}

_WEIGHT_SUFFIXES = (".safetensors", ".bin")


def _is_complete_model_dir(path: Path) -> bool:
    """A directory we can load offline: config plus one weight file."""
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return any(p.is_file() for suffix in _WEIGHT_SUFFIXES for p in path.glob(f"*{suffix}"))


def local_model_dir(model_id: str, env_var: Optional[str] = None,
                    environ: Optional[dict] = None) -> Optional[Path]:
    """Local directory holding `model_id`, or None.

    Resolution order: explicit ``env_var`` (PRISM_DEPTH_MODEL_PATH /
    PRISM_SEGMENTATION_MODEL_PATH), then the conventional
    ``model_assets/pretrained/<name>`` directory. Only directories that look
    complete (config + weights) are returned, so a half-written download never
    silently changes the loading path.
    """
    env = os.environ if environ is None else environ
    candidates = []
    if env_var is not None and env.get(env_var):
        candidates.append(Path(env[env_var]))
    default = _DEFAULT_DIRS.get(model_id)
    if default is not None:
        candidates.append(default)
    for candidate in candidates:
        if _is_complete_model_dir(candidate):
            return candidate
    return None


def resolve_model_source(model_id: str, env_var: Optional[str] = None,
                         environ: Optional[dict] = None) -> Union[str, Path]:
    """Local directory when one is installed, otherwise the HF repo id.

    Callers should pass the result to ``from_pretrained(..., local_files_only=True)``
    either way: a repo id then resolves from the shared HF cache if present and
    never from the network.
    """
    local = local_model_dir(model_id, env_var=env_var, environ=environ)
    return local if local is not None else model_id


def model_source_report(model_id: str, env_var: Optional[str] = None,
                        environ: Optional[dict] = None) -> dict:
    """Provenance summary for logs and reports."""
    env = os.environ if environ is None else environ
    local = local_model_dir(model_id, env_var=env_var, environ=environ)
    env_value = env.get(env_var) if env_var is not None else None
    return {
        "model_id": model_id,
        "source": str(local) if local is not None else model_id,
        "origin": ("environment override" if (local is not None and env_value
                    and Path(env_value) == local)
                   else ("local pretrained directory" if local is not None
                         else "hf repo id (cache-only load, no download)")),
        "local_files_only": True,
        "installer": "tools/download_prism_pretrained.py",
    }
