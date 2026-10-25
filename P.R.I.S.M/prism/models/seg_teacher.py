"""SegFormer-B0 (ADE20K) teacher -> prism-7 soft terrain targets.

The teacher is `nvidia/segformer-b0-finetuned-ade-512-512` from the local HF cache.
It predicts 150 ADE20K classes; prism needs the 7-class off-road taxonomy in
`config.TERRAIN_CLASSES`. The bridge is an explicit **150 x 7 weight matrix**, not a
hard lookup, because several ADE classes genuinely straddle two prism groups and the
whole point of this module is to emit *soft* targets for distillation.

How the mapping is built
------------------------
Every ADE index is looked up by name from the checkpoint's own ``id2label`` (never by
guessed index), then assigned a row of the matrix whose entries sum to 1.0. Rows that
sum to 0.0 are "abstain": their probability mass is dropped, and a pixel whose total
retained mass falls below ``min_mass`` becomes IGNORE_INDEX (255) so it cannot pollute
distillation.

Deliberate soft splits (documented, not accidental):
  * ``tree`` / ``palm`` -> 0.60 rough_veg + 0.40 obstacle.  ADE has no trunk class; a
    tree is canopy (drive-through-able foliage at head height, irrelevant to a 22 cm
    UGV) *plus* a trunk that is a hard obstacle. Splitting the mass says exactly that,
    and the downstream traversability stage resolves the ambiguity with height above
    ground: rough_veg pixels that are tall and vertical are obstacles.
  * ``plant`` -> 0.85 rough_veg + 0.15 obstacle (mostly soft foliage, sometimes a pot).
  * ``car`` / ``truck`` / ``van`` / ``bus`` -> 0.60 obstacle + 0.40 dynamic. In clips
    04/05 these are parked, i.e. static obstacles, but they are vehicles and may move,
    so ``config``'s "dynamic = person, animal, vehicle" note keeps a real share.
  * ``bicycle`` / ``minibike`` -> 0.65 dynamic + 0.35 obstacle (usually ridden here).
  * ``fountain`` -> 0.60 obstacle + 0.40 water.
Both classes in every split have TERRAIN_DRIVE_PRIOR == 0, so navigation safety never
depends on which side of a split wins the argmax.

Abstentions (-> 255) are indoor furniture and small props that cannot occur in this
footage, plus genuinely ambiguous terrain words (``mountain``, ``hill``, ``bridge``,
``canopy``) where a wrong hard label would teach the student something false.
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..config import (CFG, IGNORE_INDEX, N_TERRAIN, PROC_H, PROC_W, TERRAIN_CLASSES)

TEACHER_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
TEACHER_SIZE = 512                      # the side the checkpoint was finetuned at

# Input geometry.  The checkpoint's nominal input is a 512x512 square, but squashing a
# 16:9 POV frame into a square wrecks it on this footage: measured on clip_01 frame 0 the
# square pass labels 38.6% of the frame "wall" (it is asphalt trail), while an
# aspect-preserving resize with the *short side at 512* - the protocol SegFormer actually
# uses to evaluate on ADE20K - gives earth 36.7% / tree 32.3% / sky 15.0%, which is right.
# So "short512" (910x512 for 16:9) is the default and "square512" is kept for comparison.
INPUT_MODES = ("short512", "square512")

SKY, TRAIL, GRASS, ROUGH_VEG, OBSTACLE, WATER, DYNAMIC = range(7)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# --------------------------------------------------------------------------------------
# ADE20K name -> {prism-7 class: weight}.  Names are the checkpoint's own id2label
# strings (lower case, note the trailing space on "bed "). Anything not listed abstains.
# --------------------------------------------------------------------------------------
ADE_NAME_TO_DRISHTI: dict[str, dict[int, float]] = {
    # ---- sky ------------------------------------------------------------------------
    "sky": {SKY: 1.0},

    # ---- trail: drivable prepared / bare ground --------------------------------------
    "road": {TRAIL: 1.0},
    "path": {TRAIL: 1.0},
    "sidewalk": {TRAIL: 1.0},
    "earth": {TRAIL: 1.0},
    "dirt track": {TRAIL: 1.0},
    "runway": {TRAIL: 1.0},
    "sand": {TRAIL: 1.0},
    "land": {TRAIL: 0.75, GRASS: 0.25},     # ADE "land" = bare shoreline/ground
    "floor": {TRAIL: 1.0},                  # fires on paved yard surfaces in clip_04

    # ---- grass: low vegetation, traversable but soft ---------------------------------
    "grass": {GRASS: 1.0},
    "field": {GRASS: 1.0},

    # ---- rough_veg: bushes, hedges, foliage ------------------------------------------
    "tree": {ROUGH_VEG: 0.60, OBSTACLE: 0.40},      # canopy + trunk, see module docstring
    "palm": {ROUGH_VEG: 0.60, OBSTACLE: 0.40},
    "plant": {ROUGH_VEG: 0.85, OBSTACLE: 0.15},
    "flower": {ROUGH_VEG: 1.0},

    # ---- obstacle: rigid, non-traversable --------------------------------------------
    "wall": {OBSTACLE: 1.0},
    "building": {OBSTACLE: 1.0},
    "house": {OBSTACLE: 1.0},
    "hovel": {OBSTACLE: 1.0},
    "skyscraper": {OBSTACLE: 1.0},
    "tower": {OBSTACLE: 1.0},
    "fence": {OBSTACLE: 1.0},
    "railing": {OBSTACLE: 1.0},
    "bannister": {OBSTACLE: 1.0},
    "door": {OBSTACLE: 1.0},
    "windowpane": {OBSTACLE: 1.0},
    "column": {OBSTACLE: 1.0},
    "pole": {OBSTACLE: 1.0},
    "streetlight": {OBSTACLE: 1.0},
    "traffic light": {OBSTACLE: 1.0},
    "signboard": {OBSTACLE: 1.0},
    "bulletin board": {OBSTACLE: 1.0},
    "poster": {OBSTACLE: 1.0},
    "rock": {OBSTACLE: 1.0},
    "stairs": {OBSTACLE: 1.0},              # a step is not traversable for a 4.5 cm clearance
    "stairway": {OBSTACLE: 1.0},
    "step": {OBSTACLE: 1.0},
    "bench": {OBSTACLE: 1.0},
    "sculpture": {OBSTACLE: 1.0},
    "booth": {OBSTACLE: 1.0},
    "grandstand": {OBSTACLE: 1.0},
    "tent": {OBSTACLE: 1.0},
    "awning": {OBSTACLE: 1.0},
    "barrel": {OBSTACLE: 1.0},
    "box": {OBSTACLE: 1.0},
    "ashcan": {OBSTACLE: 1.0},
    "fountain": {OBSTACLE: 0.60, WATER: 0.40},

    # ---- water -----------------------------------------------------------------------
    "water": {WATER: 1.0},
    "sea": {WATER: 1.0},
    "river": {WATER: 1.0},
    "lake": {WATER: 1.0},
    "swimming pool": {WATER: 1.0},
    "waterfall": {WATER: 1.0},

    # ---- dynamic ---------------------------------------------------------------------
    "person": {DYNAMIC: 1.0},
    "animal": {DYNAMIC: 1.0},
    "bicycle": {DYNAMIC: 0.65, OBSTACLE: 0.35},
    "minibike": {DYNAMIC: 0.65, OBSTACLE: 0.35},
    "car": {OBSTACLE: 0.60, DYNAMIC: 0.40},
    "truck": {OBSTACLE: 0.60, DYNAMIC: 0.40},
    "van": {OBSTACLE: 0.60, DYNAMIC: 0.40},
    "bus": {OBSTACLE: 0.60, DYNAMIC: 0.40},
    "boat": {OBSTACLE: 0.60, DYNAMIC: 0.40},
    "ship": {OBSTACLE: 0.60, DYNAMIC: 0.40},
    "airplane": {OBSTACLE: 0.60, DYNAMIC: 0.40},
    "tank": {OBSTACLE: 0.60, DYNAMIC: 0.40},
}

# Explicit abstentions we *want* on record (everything unlisted also abstains, but these
# are the ones a reviewer would ask about).
ADE_ABSTAIN_NOTES = {
    "mountain": "distant terrain, neither trail nor obstacle at UGV scale",
    "hill": "distant terrain, ambiguous support",
    "bridge": "drivable deck vs. structure - cannot be resolved from the label alone",
    "canopy": "overhead structure, irrelevant to a 22 cm-wide chassis",
    "trade name": "text pasted on other surfaces",
    "flag": "thin, arbitrary support",
    "ceiling": "indoor only",
    "escalator": "indoor only",
    "pier": "water/structure boundary",
}


def build_mapping_matrix(id2label: dict[int, str]) -> np.ndarray:
    """(150, 7) float32 row-stochastic-or-zero matrix, built by *name* lookup."""
    n = len(id2label)
    m = np.zeros((n, N_TERRAIN), np.float32)
    for idx in range(n):
        name = str(id2label[idx]).strip().lower()
        spec = ADE_NAME_TO_DRISHTI.get(name)
        if spec is None:
            spec = ADE_NAME_TO_DRISHTI.get(str(id2label[idx]).lower())
        if not spec:
            continue
        for c, w in spec.items():
            m[idx, c] = w
        s = m[idx].sum()
        if s > 0:
            m[idx] /= s
    return m


class SegTeacher:
    """SegFormer-B0/ADE20K wrapped as a prism-7 soft-target producer.

    ``__call__(bgr)`` -> (prob7 float32 (7,h,w), label uint8 (h,w) with 255=ignore).
    """

    def __init__(self, device: Optional[str] = None, fp16: bool = True,
                 min_mass: float = 0.35, out_size: tuple[int, int] = (PROC_W, PROC_H),
                 input_mode: str = "short512", flip_tta: bool = False):
        from transformers import SegformerForSemanticSegmentation

        assert input_mode in INPUT_MODES, input_mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = bool(fp16 and self.device == "cuda")
        self.min_mass = float(min_mass)
        self.out_size = out_size
        self.input_mode = input_mode
        self.flip_tta = bool(flip_tta)

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            TEACHER_ID, local_files_only=False)
        self.model.eval().to(self.device)
        if self.fp16:
            self.model.half()

        raw = self.model.config.id2label
        self.id2label = {int(k): v for k, v in raw.items()}
        self.n_ade = len(self.id2label)
        self.M = build_mapping_matrix(self.id2label)
        self._M_t = torch.from_numpy(self.M).to(self.device)
        if self.fp16:
            self._M_t = self._M_t.half()

        self.n_mapped = int((self.M.sum(1) > 0).sum())
        self.n_abstain = self.n_ade - self.n_mapped

    # ------------------------------------------------------------------ helpers
    def mapping_report(self) -> str:
        lines = [f"ADE20K -> prism-7 mapping: {self.n_mapped}/{self.n_ade} classes mapped, "
                 f"{self.n_abstain} abstain -> {IGNORE_INDEX}"]
        for c in range(N_TERRAIN):
            names = [self.id2label[i] for i in np.nonzero(self.M[:, c] > 0)[0]]
            lines.append(f"  {c} {TERRAIN_CLASSES[c]:<10s} <- " + ", ".join(sorted(names)))
        return "\n".join(lines)

    def _input_size(self, bgr: np.ndarray) -> tuple[int, int]:
        """(w, h) fed to the network, both multiples of 32."""
        if self.input_mode == "square512":
            return TEACHER_SIZE, TEACHER_SIZE
        h, w = bgr.shape[:2]
        s = TEACHER_SIZE / min(h, w)
        return (int(round(w * s / 32)) * 32, int(round(h * s / 32)) * 32)

    def _preprocess(self, bgr: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
        x = rgb.astype(np.float32) / 255.0
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0)
        return t.half() if self.fp16 else t

    @torch.no_grad()
    def _ade_probs(self, x: torch.Tensor) -> torch.Tensor:
        """(B,150,h/4,w/4) float32 softmax, with optional horizontal-flip TTA."""
        p = self.model(pixel_values=x).logits.float().softmax(1)
        if self.flip_tta:
            pf = self.model(pixel_values=torch.flip(x, dims=[3])).logits.float().softmax(1)
            p = 0.5 * (p + torch.flip(pf, dims=[3]))
        return p

    def _to_drishti(self, p150: torch.Tensor, oh: int, ow: int
                    ) -> tuple[np.ndarray, np.ndarray]:
        p7 = torch.einsum("bchw,cd->bdhw", p150, self._M_t.float())
        p7 = F.interpolate(p7, size=(oh, ow), mode="bilinear", align_corners=False)
        mass = p7.sum(1, keepdim=True)
        keep = mass >= self.min_mass
        p7 = p7 / mass.clamp_min(1e-6)
        lab = p7.argmax(1).to(torch.uint8)
        lab[~keep[:, 0]] = IGNORE_INDEX
        # abstained pixels get a uniform (maximum-entropy) soft target; they are masked
        # out of every loss anyway, this only keeps the array well-formed.
        p7 = torch.where(keep, p7, torch.full_like(p7, 1.0 / N_TERRAIN))
        return p7.cpu().numpy().astype(np.float32), lab.cpu().numpy()

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def probs(self, bgr: np.ndarray, out_size: Optional[tuple[int, int]] = None
              ) -> tuple[np.ndarray, np.ndarray]:
        """Soft prism-7 probabilities + hard label for one BGR frame.

        Returns (prob (7,h,w) float32 summing to 1 on kept pixels, label (h,w) uint8).
        """
        ow, oh = out_size or self.out_size
        x = self._preprocess(bgr, self._input_size(bgr)).to(self.device, non_blocking=True)
        p150 = self._ade_probs(x)
        p, l = self._to_drishti(p150, oh, ow)
        del x, p150
        return p[0], l[0]

    @torch.no_grad()
    def probs_batch(self, bgr_list: list[np.ndarray],
                    out_size: Optional[tuple[int, int]] = None
                    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched version. Keep the batch <= 4 to respect the shared-GPU budget."""
        ow, oh = out_size or self.out_size
        size = self._input_size(bgr_list[0])
        x = torch.cat([self._preprocess(b, size) for b in bgr_list], 0).to(self.device)
        p150 = self._ade_probs(x)
        p, l = self._to_drishti(p150, oh, ow)
        del x, p150
        return p, l

    @torch.no_grad()
    def ade_label(self, bgr: np.ndarray, out_size: Optional[tuple[int, int]] = None
                  ) -> np.ndarray:
        """Raw ADE20K argmax at `out_size` - diagnostics / mapping audits only."""
        ow, oh = out_size or self.out_size
        x = self._preprocess(bgr, self._input_size(bgr)).to(self.device)
        p = self._ade_probs(x)
        p = F.interpolate(p, size=(oh, ow), mode="bilinear", align_corners=False)
        return p[0].argmax(0).cpu().numpy().astype(np.uint8)

    def __call__(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.probs(bgr)

    def close(self) -> None:
        self.model.cpu()
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ------------------------------------------------------------------------------ self-test
if __name__ == "__main__":
    from .. import io_utils

    dev = io_utils.device()
    t0 = time.time()
    teacher = SegTeacher(device=dev)
    print(f"[teacher] loaded {TEACHER_ID} on {dev} fp16={teacher.fp16} "
          f"input_mode={teacher.input_mode} in {time.time() - t0:.1f}s")
    print(teacher.mapping_report())
    print("abstain notes:")
    for k, v in ADE_ABSTAIN_NOTES.items():
        print(f"  {k:<12s} {v}")

    frames = [f for _, f in io_utils.read_frames("clip_01", max_frames=8)]
    p, lab = teacher.probs(frames[0])
    print(f"prob {p.shape} {p.dtype} sum/px {p.sum(0).mean():.4f}  "
          f"label {lab.shape} uniq={np.unique(lab)}")

    # timing
    for _ in range(3):
        teacher.probs(frames[0])
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for f in frames:
        teacher.probs(f)
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / len(frames) * 1000
    print(f"[teacher] {dt:.1f} ms/frame on {dev}")
    if dev == "cuda":
        print(f"[teacher] peak VRAM {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")

    # class balance over the 8 frames
    counts = np.zeros(N_TERRAIN + 1, np.int64)
    for f in frames:
        _, l = teacher.probs(f)
        for c in range(N_TERRAIN):
            counts[c] += int((l == c).sum())
        counts[-1] += int((l == IGNORE_INDEX).sum())
    tot = counts.sum()
    for c, name in enumerate(TERRAIN_CLASSES):
        print(f"  {name:<10s} {100*counts[c]/tot:5.2f}%")
    print(f"  {'ignore':<10s} {100*counts[-1]/tot:5.2f}%")

    # visual sanity check over one frame per clip
    from ..viz_common import colorize_terrain, overlay
    from ..config import WORK_DIR
    rows = []
    for cid in ["clip_01", "clip_02", "clip_03", "clip_04", "clip_05"]:
        fr = [f for _, f in io_utils.read_frames(cid, max_frames=150)][-1]
        _, l = teacher.probs(fr)
        rows.append(np.hstack([fr, overlay(fr, colorize_terrain(l), 0.6)]))
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(WORK_DIR / "_teacher_check.png"), np.vstack(rows))
    print(f"wrote {WORK_DIR / '_teacher_check.png'}")
    teacher.close()
