"""Quality gate over the rendered demo videos.

Checks every expected output exists, is the right length and resolution, decodes cleanly,
and is not visually dead (blank, frozen, or mostly empty placeholder panels). Prints a
table and exits non-zero if anything fails, so it can gate a rebuild.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prism.config import OUT_DIR, OUTPUT_STAGES, CLIP_IDS, CLIP_FPS

EXPECT_FRAMES = 300
SAMPLES = 9


def check(path: Path) -> dict:
    r = {"path": str(path.relative_to(OUT_DIR)), "ok": False, "notes": []}
    if not path.exists():
        r["notes"].append("missing")
        return r
    r["mb"] = round(path.stat().st_size / 1e6, 2)
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    r.update(frames=n, size=f"{w}x{h}", fps=round(fps, 1))
    if abs(n - EXPECT_FRAMES) > 2:
        r["notes"].append(f"frames {n} != {EXPECT_FRAMES}")
    if abs(fps - CLIP_FPS) > 0.6:
        r["notes"].append(f"fps {fps:.1f}")
    if w < 1280 or h < 720:
        r["notes"].append(f"small {w}x{h}")

    idxs = np.linspace(0, max(n - 1, 0), SAMPLES).astype(int)
    frames, means, stds = [], [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            r["notes"].append(f"decode fail @{i}")
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(g, (160, 90)))
        means.append(float(g.mean()))
        stds.append(float(g.std()))
    cap.release()
    if not frames:
        return r

    r["luma"] = round(float(np.mean(means)), 1)
    r["detail"] = round(float(np.mean(stds)), 1)
    if r["detail"] < 12:
        r["notes"].append("near-blank frames")
    # frozen check: consecutive samples spread over the clip should differ
    diffs = [float(np.abs(frames[i].astype(np.float32) - frames[i + 1].astype(np.float32)).mean())
             for i in range(len(frames) - 1)]
    r["motion"] = round(float(np.mean(diffs)), 2)
    if r["motion"] < 0.8:
        r["notes"].append("static across the whole clip")
    r["ok"] = not r["notes"]
    return r


def main() -> None:
    rows, missing_stage = [], []
    for stage_dir, desc in OUTPUT_STAGES:
        present = 0
        for cid in CLIP_IDS:
            res = check(OUT_DIR / stage_dir / f"{cid}.mp4")
            rows.append(res)
            present += int(res["ok"])
        if present < len(CLIP_IDS):
            missing_stage.append((stage_dir, present, desc))

    print(f"{'output':44s} {'frames':>6s} {'size':>10s} {'MB':>6s} {'detail':>6s} {'motion':>6s}  notes")
    print("-" * 110)
    for r in rows:
        print(f"{r['path']:44s} {r.get('frames','-'):>6} {r.get('size','-'):>10s} "
              f"{r.get('mb','-'):>6} {r.get('detail','-'):>6} {r.get('motion','-'):>6}  "
              f"{'; '.join(r['notes']) if r['notes'] else 'ok'}")

    n_ok = sum(r["ok"] for r in rows)
    print("-" * 110)
    print(f"{n_ok}/{len(rows)} videos pass")
    (OUT_DIR / "verify.json").write_text(json.dumps(rows, indent=2))
    if missing_stage:
        print("\nincomplete stages:")
        for d, p, desc in missing_stage:
            print(f"  {d:28s} {p}/{len(CLIP_IDS)}  {desc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
