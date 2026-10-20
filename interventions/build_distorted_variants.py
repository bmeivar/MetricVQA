"""
Build barrel-distorted (fish-eye) variant of every in-the-wild image.

Distortion model: r' = r * (1 + k * r^2), where r is the normalized radius
from image center. k > 0 is barrel (fish-eye); k < 0 is pincushion.

The whole image is warped — straight lines bend, vanishing points are
disturbed, but object identity and approximate apparent size are preserved.

Writes:
  images_distorted/<imgstem>__<phrase_slug>.png
  questions_distorted.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from build_context_variants import load_mask, slug  # noqa: E402


def barrel_maps(H: int, W: int, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Precompute remap arrays for barrel distortion."""
    y, x = np.meshgrid(np.arange(H, dtype=np.float32),
                       np.arange(W, dtype=np.float32), indexing="ij")
    # Normalize to [-1, 1] using min-dim so distortion is isotropic
    cx, cy = W / 2.0, H / 2.0
    half = min(W, H) / 2.0
    xn = (x - cx) / half
    yn = (y - cy) / half
    r2 = xn ** 2 + yn ** 2
    factor = 1.0 + k * r2
    # Source coordinates: pull from r/factor so output has barrel look
    # For barrel (k>0): src_r < dst_r → image looks "pushed out".
    # Pre-divide: src_n = dst_n / factor → straight lines bend outward.
    src_x = xn / factor * half + cx
    src_y = yn / factor * half + cy
    return src_x.astype(np.float32), src_y.astype(np.float32)


def warp_image(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray,
               border_color: int = 127) -> np.ndarray:
    return cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border_color, border_color, border_color),
    )


def shrink_and_pad(img: np.ndarray, k: float, gray: int = 127) -> np.ndarray:
    """Shrink input by 1/(1+|k|) so that after barrel distortion, ALL input
    content fits inside the output frame. Pad to original H,W with gray."""
    H, W, _ = img.shape
    scale = 1.0 / (1.0 + abs(k))
    new_H = max(1, int(round(H * scale)))
    new_W = max(1, int(round(W * scale)))
    small = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_AREA)
    canvas = np.full((H, W, 3), gray, dtype=np.uint8)
    y0 = (H - new_H) // 2
    x0 = (W - new_W) // 2
    canvas[y0:y0 + new_H, x0:x0 + new_W] = small
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REPO / "questions_v3.csv"))
    ap.add_argument("--phrases", default=str(REPO / "sam_phrases" / "sam_phrases_v1.csv"))
    ap.add_argument("--images-dir", default=str(REPO / "images"))
    ap.add_argument("--masks-dir", default=str(REPO / "sam_phrases" / "masks"))
    ap.add_argument("--out-root", default=str(REPO))
    ap.add_argument("-k", "--k", type=float, default=0.15,
                    help="Barrel distortion coefficient (0.10-0.20 subtle, "
                         "0.25-0.35 moderate, 0.4+ strong).")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="Optional suffix on the output dir/csv name, e.g. "
                         "'_k015' to keep multiple k versions side-by-side.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Build only the first N rows (for visual smoke test).")
    ap.add_argument("--preserve-edges", action="store_true",
                    help="Pre-shrink the input by 1/(1+|k|) so all content "
                         "stays inside the output frame after distortion.")
    args = ap.parse_args()

    suf = args.out_suffix
    out_root = Path(args.out_root)
    out_dir = out_root / f"images_distorted{suf}"
    out_dir.mkdir(exist_ok=True)

    questions = pd.read_csv(args.csv)
    phrases = pd.read_csv(args.phrases)
    # case-insensitive merge so v3 (Capitalized) joins to sam_phrases (lowercase)
    questions["_qkey"] = questions["query"].str.strip().str.lower()
    phrases["_qkey"] = phrases["query"].str.strip().str.lower()
    df = questions.merge(phrases[["image", "_qkey", "sam_phrase"]],
                         on=["image", "_qkey"], how="left")
    df = df.drop(columns=["_qkey"])
    if args.limit:
        df = df.head(args.limit)

    rows = []
    img_cache: dict[str, np.ndarray] = {}
    maps_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    for _, row in df.iterrows():
        img_name = row["image"]
        if img_name not in img_cache:
            img_cache[img_name] = np.array(
                Image.open(Path(args.images_dir) / img_name).convert("RGB")
            )
        img = img_cache[img_name]
        H, W, _ = img.shape

        key = (H, W)
        if key not in maps_cache:
            maps_cache[key] = barrel_maps(H, W, args.k)
        map_x, map_y = maps_cache[key]

        if args.preserve_edges:
            padded = shrink_and_pad(img, args.k)
            distorted_img = warp_image(padded, map_x, map_y)
        else:
            distorted_img = warp_image(img, map_x, map_y)

        stem = Path(img_name).stem
        if pd.isna(row.get("sam_phrase")):
            out_name = f"{stem}__no_phrase.png"
        else:
            out_name = f"{stem}__{slug(row['sam_phrase'])}.png"

        Image.fromarray(distorted_img).save(out_dir / out_name)
        rec = dict(row); rec["image"] = out_name
        rows.append(rec)

    out_csv = out_root / f"questions_distorted{suf}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"built {len(rows)} distorted images (k={args.k})")
    print(f"  -> {out_dir}")
    print(f"  -> {out_csv}")


if __name__ == "__main__":
    main()
