"""
Build zoom_in / zoom_out variants for every (image, query, sam_phrase) row in
the in-the-wild benchmark.

Both variants start from the *object_only* image (= object pixels kept, gray
fill outside the SAM mask, same as context_object_only). Only the object's
apparent pixel size changes — no scene, no reference cues. This isolates
apparent-size bias as a clean causal test.

Definitions (zoom factor f = 1.20 in / 0.80 out; canvas size preserved):
  - zoom_in  : object_only image scaled by 1.20x, centered crop back to (H, W)
               on the object's mask centroid. Object pixels ~20% larger.
  - zoom_out : object_only image scaled by 0.80x, placed on a gray (127) canvas
               of (H, W) with the object's centroid at canvas center.

For each row we write:
  images_zoom_in/<imgstem>__<phrase_slug>.png
  images_zoom_out/<imgstem>__<phrase_slug>.png

And derived CSVs (one per zoom direction) with the same columns as
questions_v2.csv plus an updated `image` column pointing to the per-row file:
  questions_zoom_in.csv
  questions_zoom_out.csv

Rows without a SAM phrase or whose mask is missing are skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from build_context_variants import load_mask, slug, apply_keep_mask, GRAY  # noqa: E402


def centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    return float(ys.mean()), float(xs.mean())


def object_only(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Object pixels kept, gray fill outside the SAM mask."""
    return apply_keep_mask(img, mask, feather_px=0, gray=GRAY)


def zoom_in(img: np.ndarray, mask: np.ndarray, factor: float = 1.40) -> np.ndarray:
    """Scale object_only by factor; center crop to (H, W) on object's centroid."""
    H, W, _ = img.shape
    obj = object_only(img, mask)
    new_H = int(round(H * factor))
    new_W = int(round(W * factor))
    big = np.array(Image.fromarray(obj).resize((new_W, new_H), Image.LANCZOS))

    cy, cx = centroid(mask)
    cy_s, cx_s = cy * factor, cx * factor

    y0 = int(round(cy_s - H / 2))
    x0 = int(round(cx_s - W / 2))
    y0 = max(0, min(y0, new_H - H))
    x0 = max(0, min(x0, new_W - W))

    return big[y0:y0 + H, x0:x0 + W]


def zoom_out(img: np.ndarray, mask: np.ndarray, factor: float = 0.60) -> np.ndarray:
    """Scale object_only by factor; paste on gray (H, W) canvas, object centered."""
    H, W, _ = img.shape
    obj = object_only(img, mask)
    new_H = int(round(H * factor))
    new_W = int(round(W * factor))
    small = np.array(Image.fromarray(obj).resize((new_W, new_H), Image.LANCZOS))

    canvas = np.full((H, W, 3), GRAY, dtype=np.uint8)

    # place the scaled image so the object's (scaled) centroid lands at canvas center
    cy, cx = centroid(mask)
    cy_s, cx_s = cy * factor, cx * factor

    y0 = int(round(H / 2 - cy_s))
    x0 = int(round(W / 2 - cx_s))
    y0 = max(0, min(y0, H - new_H))
    x0 = max(0, min(x0, W - new_W))

    canvas[y0:y0 + new_H, x0:x0 + new_W] = small
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REPO / "questions_v2.csv"))
    ap.add_argument("--phrases", default=str(REPO / "sam_phrases" / "sam_phrases_v1.csv"))
    ap.add_argument("--images-dir", default=str(REPO / "images"))
    ap.add_argument("--masks-dir", default=str(REPO / "sam_phrases" / "masks"))
    ap.add_argument("--out-root", default=str(REPO))
    ap.add_argument("--in-factor", type=float, default=1.40)
    ap.add_argument("--out-factor", type=float, default=0.60)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    zi_dir = out_root / "images_zoom_in"
    zo_dir = out_root / "images_zoom_out"
    zi_dir.mkdir(exist_ok=True)
    zo_dir.mkdir(exist_ok=True)

    questions = pd.read_csv(args.csv)
    phrases = pd.read_csv(args.phrases)
    df = questions.merge(phrases[["image", "query", "sam_phrase"]],
                         on=["image", "query"], how="left")

    rows_in, rows_out = [], []
    skipped_no_phrase = skipped_no_mask = 0
    img_cache = {}

    for _, row in df.iterrows():
        if pd.isna(row.get("sam_phrase")):
            skipped_no_phrase += 1
            continue
        mask = load_mask(row["image"], row["sam_phrase"], Path(args.masks_dir))
        if mask is None:
            skipped_no_mask += 1
            continue
        if row["image"] not in img_cache:
            img_cache[row["image"]] = np.array(
                Image.open(Path(args.images_dir) / row["image"]).convert("RGB")
            )
        img = img_cache[row["image"]]

        stem = Path(row["image"]).stem
        slug_s = slug(row["sam_phrase"])
        zi_name = f"{stem}__{slug_s}.png"
        zo_name = f"{stem}__{slug_s}.png"

        Image.fromarray(zoom_in(img, mask, args.in_factor)).save(zi_dir / zi_name)
        Image.fromarray(zoom_out(img, mask, args.out_factor)).save(zo_dir / zo_name)

        rec_in = dict(row)
        rec_in["image"] = zi_name
        rec_out = dict(row)
        rec_out["image"] = zo_name
        rows_in.append(rec_in)
        rows_out.append(rec_out)

    pd.DataFrame(rows_in).to_csv(out_root / "questions_zoom_in.csv", index=False)
    pd.DataFrame(rows_out).to_csv(out_root / "questions_zoom_out.csv", index=False)

    print(f"built {len(rows_in)} pairs of zoom_in/zoom_out images")
    print(f"  skipped: {skipped_no_phrase} no_phrase, {skipped_no_mask} no_mask")
    print(f"  -> {zi_dir}")
    print(f"  -> {zo_dir}")


if __name__ == "__main__":
    main()
