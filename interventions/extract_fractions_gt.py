#!/usr/bin/env python3
"""
Compute ground-truth image fractions for each query in the in-the-wild dataset.
Uses the object_only images (gray=127 background) to extract the object mask,
then computes what fraction of the image the object's bounding box spans.

The fraction is computed for the dimension asked about in the query:
- "how long/length/wide/width/diameter" -> fraction of bbox width OR height (whichever is larger for length, width-axis for width)
- "how tall/height/high" -> fraction of bbox height

Output: CSV with columns [image, query, answer_cm, fraction_width, fraction_height, fraction_relevant]
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_context_variants import load_mask  # noqa: E402
from release_data import ITW_CSV, PHRASES_CSV, MASKS_DIR, ROOT, load_itw, require  # noqa: E402

GRAY_BG = 127


def get_object_mask(obj_img: np.ndarray) -> np.ndarray:
    """Extract object mask from object_only image (gray=127 background)."""
    return ~np.all(obj_img == GRAY_BG, axis=2)


def compute_bbox_fractions(mask: np.ndarray):
    """Compute bbox width and height fractions from a binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return None, None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h, w = mask.shape

    frac_width = (cmax - cmin) / w
    frac_height = (rmax - rmin) / h
    return frac_width, frac_height


def get_relevant_fraction(query: str, frac_width: float, frac_height: float) -> float:
    """Determine which fraction is relevant based on the query's dimension."""
    q = query.lower()
    if any(w in q for w in ["how tall", "height", "how high"]):
        return frac_height
    elif any(w in q for w in ["how wide", "width", "diameter"]):
        return frac_width
    elif any(w in q for w in ["how long", "length"]):
        # "length" = the longer dimension of the object
        return max(frac_width, frac_height)
    else:
        return max(frac_width, frac_height)


def main():
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ITW_CSV))
    ap.add_argument("--phrases", default=str(PHRASES_CSV))
    ap.add_argument("--masks-dir", default=str(MASKS_DIR))
    ap.add_argument("--output", default=str(ROOT / "in_the_wild_fractions_gt.csv"))
    args = ap.parse_args()
    output_path = args.output

    # Object masks from interventions/masks_from_release.py; the object_only canvas is the
    # image gray-filled outside this mask, so its bbox is the mask's bbox.
    require(Path(args.phrases), "sam_phrases CSV (run interventions/masks_from_release.py)")
    q = load_itw(args.csv)
    phr = pd.read_csv(args.phrases)[["image", "query", "sam_phrase"]]
    df_obj = q.merge(phr, on=["image", "query"], how="left")

    print(f"Computing fractions for {len(df_obj)} queries from released masks")

    results = []
    for i, row in df_obj.iterrows():
        query = str(row["query"]).strip()
        gt_cm = float(row["answer"])
        mask = (load_mask(row["image"], row["sam_phrase"], Path(args.masks_dir))
                if isinstance(row["sam_phrase"], str) else None)

        if mask is None:
            print(f"[{i+1}/{len(df_obj)}] No mask for: {query}")
            results.append({
                "image": row["image"], "query": query, "answer_cm": gt_cm,
                "fraction_width": None, "fraction_height": None, "fraction_relevant": None,
            })
            continue

        frac_w, frac_h = compute_bbox_fractions(mask)
        if frac_w is None:
            print(f"[{i+1}/{len(df_obj)}] No object found in mask")
            frac_rel = None
        else:
            frac_rel = get_relevant_fraction(query, frac_w, frac_h)
            print(f"[{i+1}/{len(df_obj)}] {query[:50]:50s} | w={frac_w:.3f} h={frac_h:.3f} rel={frac_rel:.3f} | GT={gt_cm}cm")

        results.append({
            "image": row["image"], "query": query, "answer_cm": gt_cm,
            "fraction_width": frac_w, "fraction_height": frac_h,
            "fraction_relevant": frac_rel,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    valid = results_df.dropna(subset=["fraction_relevant"])
    print(f"\nDone. {len(valid)}/{len(results_df)} valid fractions.")
    print(f"Saved to {output_path}")
    print(f"\nFraction stats:")
    print(f"  Mean: {valid['fraction_relevant'].mean():.3f}")
    print(f"  Median: {valid['fraction_relevant'].median():.3f}")
    print(f"  Range: [{valid['fraction_relevant'].min():.3f}, {valid['fraction_relevant'].max():.3f}]")


if __name__ == "__main__":
    main()
