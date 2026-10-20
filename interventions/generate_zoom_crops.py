#!/usr/bin/env python3
"""
Generate zoom crops from full scene images for the apparent-size experiment.

Three conditions:
- zoom_out: the original full image (widest view)
- base: moderate crop centered on the object
- zoom_in: tighter crop centered on the object

No rescaling, no gray canvas — pure field-of-view change on real scene content.
All crops are resized back to the original image dimensions for consistent input.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image

# Crop factors relative to object bounding box
# base = 3x the object bbox, zoom_in = 1.8x the object bbox
# zoom_out = original image (no crop)
BASE_FACTOR = 3.0
ZOOM_IN_FACTOR = 1.8

OUTPUT_DIR = "qwen3_235b_bundle/images/scene_zoom"
MANIFEST_DIR = "qwen3_235b_bundle/manifests"


def get_object_bbox(obj_img_path):
    """Get object bounding box from object_only image (gray=127 background)."""
    img = np.array(Image.open(obj_img_path).convert('RGB'))
    mask = ~np.all(img == 127, axis=2)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(cmin), int(rmin), int(cmax), int(rmax)


def crop_around_center(img, cx, cy, crop_w, crop_h):
    """Crop image centered at (cx, cy) with given width/height, clamped to image bounds."""
    W, H = img.size  # PIL: (width, height)

    x1 = max(0, int(cx - crop_w / 2))
    y1 = max(0, int(cy - crop_h / 2))
    x2 = min(W, int(cx + crop_w / 2))
    y2 = min(H, int(cy + crop_h / 2))

    # Adjust if clamping moved one side
    if x2 - x1 < crop_w:
        if x1 == 0:
            x2 = min(W, int(crop_w))
        else:
            x1 = max(0, int(x2 - crop_w))
    if y2 - y1 < crop_h:
        if y1 == 0:
            y2 = min(H, int(crop_h))
        else:
            y1 = max(0, int(y2 - crop_h))

    return img.crop((x1, y1, x2, y2))


def main():
    sys.stdout.reconfigure(line_buffering=True)

    gt_df = pd.read_csv('in_the_wild_fractions_gt.csv')

    # Map to original images and object_only paths
    gt_df["original_image"] = gt_df["image"].str.replace(
        "images/context/", "images/itw_vision/"
    ).str.replace(r"__.*$", ".png", regex=True)

    # Create output directories
    for variant in ["scene_zoom_in", "scene_zoom_base", "scene_zoom_out"]:
        os.makedirs(os.path.join(OUTPUT_DIR.replace("scene_zoom", variant)), exist_ok=True)

    os.makedirs(os.path.join("qwen3_235b_bundle/images/scene_zoom_in"), exist_ok=True)
    os.makedirs(os.path.join("qwen3_235b_bundle/images/scene_zoom_base"), exist_ok=True)
    os.makedirs(os.path.join("qwen3_235b_bundle/images/scene_zoom_out"), exist_ok=True)

    manifest_rows_base = []
    manifest_rows_zi = []
    manifest_rows_zo = []

    print(f"Generating scene zoom crops for {len(gt_df)} queries...")

    for i, row in gt_df.iterrows():
        obj_only_path = os.path.join("qwen3_235b_bundle", row["image"])
        orig_path = os.path.join("qwen3_235b_bundle", row["original_image"])

        if not os.path.exists(obj_only_path) or not os.path.exists(orig_path):
            print(f"[{i+1}/{len(gt_df)}] Missing files, skipping")
            continue

        # Get object bbox
        bbox = get_object_bbox(obj_only_path)
        if bbox is None:
            print(f"[{i+1}/{len(gt_df)}] No object found, skipping")
            continue

        x_min, y_min, x_max, y_max = bbox
        obj_w = x_max - x_min
        obj_h = y_max - y_min
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        # Load original image
        orig_img = Image.open(orig_path).convert('RGB')
        orig_W, orig_H = orig_img.size

        # Generate filename from object_only path
        # images/context/image_001__the_remote_control__object_only.png -> image_001__the_remote_control.png
        fname = os.path.basename(row["image"]).replace("__object_only", "")

        # zoom_out = original full image (just copy/symlink)
        zo_path = f"images/scene_zoom_out/{fname}"
        zo_full = os.path.join("qwen3_235b_bundle", zo_path)
        orig_img.save(zo_full)

        # base = crop at BASE_FACTOR × object size, resize back to original dims
        base_crop_w = min(obj_w * BASE_FACTOR, orig_W)
        base_crop_h = min(obj_h * BASE_FACTOR, orig_H)
        base_crop = crop_around_center(orig_img, cx, cy, base_crop_w, base_crop_h)
        base_crop = base_crop.resize((orig_W, orig_H), Image.LANCZOS)
        base_path = f"images/scene_zoom_base/{fname}"
        base_crop.save(os.path.join("qwen3_235b_bundle", base_path))

        # zoom_in = crop at ZOOM_IN_FACTOR × object size, resize back to original dims
        zi_crop_w = min(obj_w * ZOOM_IN_FACTOR, orig_W)
        zi_crop_h = min(obj_h * ZOOM_IN_FACTOR, orig_H)
        zi_crop = crop_around_center(orig_img, cx, cy, zi_crop_w, zi_crop_h)
        zi_crop = zi_crop.resize((orig_W, orig_H), Image.LANCZOS)
        zi_path = f"images/scene_zoom_in/{fname}"
        zi_crop.save(os.path.join("qwen3_235b_bundle", zi_path))

        # Build manifest rows
        query = row["query"]
        answer = row["answer_cm"]

        manifest_rows_zo.append({"image": zo_path, "query": query, "answer": answer})
        manifest_rows_base.append({"image": base_path, "query": query, "answer": answer})
        manifest_rows_zi.append({"image": zi_path, "query": query, "answer": answer})

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(gt_df)}] done")

    # Save manifests
    pd.DataFrame(manifest_rows_zo).to_csv(
        os.path.join(MANIFEST_DIR, "in_the_wild_scene_zoom_out.csv"), index=False)
    pd.DataFrame(manifest_rows_base).to_csv(
        os.path.join(MANIFEST_DIR, "in_the_wild_scene_zoom_base.csv"), index=False)
    pd.DataFrame(manifest_rows_zi).to_csv(
        os.path.join(MANIFEST_DIR, "in_the_wild_scene_zoom_in.csv"), index=False)

    print(f"\nDone. Generated {len(manifest_rows_base)} image sets.")
    print(f"Manifests saved to {MANIFEST_DIR}/in_the_wild_scene_zoom_*.csv")
    print(f"Images saved to qwen3_235b_bundle/images/scene_zoom_*/")
    print(f"\nCrop factors: zoom_in={ZOOM_IN_FACTOR}x, base={BASE_FACTOR}x, zoom_out=full image")
    print(f"Expected fraction ratio (zi vs zo) varies per object (depends on object size relative to image)")


if __name__ == "__main__":
    main()
