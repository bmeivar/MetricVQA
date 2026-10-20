#!/usr/bin/env python3
"""
Well-posed fraction estimation using local HF models (GPU required).
Asks VLMs to estimate what fraction of the image width/height an object occupies.
"""

import argparse
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image

from handler_utils import load_vlm

FRACTION_PROMPT = """Look at the image carefully. {query_text}
Answer with a single decimal number between 0 and 1 representing the fraction. For example, 0.25 means the object spans 25% of the image {dimension}."""


def parse_query_for_fraction(query: str) -> tuple:
    """Convert a cm query to a fraction query + determine width/height."""
    q = query.lower().strip().rstrip("?")

    if any(w in q for w in ["how tall", "height", "how high"]):
        dim = "height"
    elif any(w in q for w in ["how wide", "width", "diameter"]):
        dim = "width"
    elif any(w in q for w in ["how long", "length"]):
        dim = "max"
    else:
        dim = "max"

    obj = q
    for prefix in ["what is the length of the ", "what is the width of the ",
                   "what is the height of the ", "what is the diameter of the ",
                   "what is diameter of the ",
                   "how long is the ", "how wide is the ", "how tall is the ",
                   "how high is the ", "how long is ", "how wide is ",
                   "how tall is ", "how high is "]:
        if obj.startswith(prefix):
            obj = obj[len(prefix):]
            break

    if dim == "height":
        query_text = f"What fraction of the image height does the {obj} span?"
        dim_word = "height"
    elif dim == "width":
        query_text = f"What fraction of the image width does the {obj} span?"
        dim_word = "width"
    else:
        query_text = f"What fraction of the image does the {obj} span along its longest dimension (width or height)?"
        dim_word = "width or height"

    return query_text, dim_word, dim


def parse_fraction(text: str):
    """Parse a fraction (0-1) from model response."""
    text = text.strip()
    matches = re.findall(r"0?\.\d+|1\.0|0|1", text)
    for m in matches:
        val = float(m)
        if 0 <= val <= 1:
            return val
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        return float(m.group(1)) / 100.0
    return None


def main():
    parser = argparse.ArgumentParser(description="Well-posed fraction estimation (local models)")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--fractions-gt", default="in_the_wild_fractions_gt.csv")
    parser.add_argument("--image-dir", default="qwen3_235b_bundle")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--manifest", default=None, help="Use images from this manifest CSV instead of fractions-gt original images")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load GT fractions
    gt_df = pd.read_csv(args.fractions_gt)
    gt_df["original_image"] = gt_df["image"].str.replace(
        "images/context/", "images/itw_vision/"
    ).str.replace(r"__.*$", ".png", regex=True)

    # If manifest provided, override images with the manifest's images
    if args.manifest:
        manifest_df = pd.read_csv(args.manifest)
        # Match by query to pair zoom images with GT fractions
        gt_df = gt_df.merge(manifest_df[["image", "query"]].rename(columns={"image": "manifest_image"}), on="query", how="inner")
        gt_df["original_image"] = gt_df["manifest_image"]

    if args.max_rows:
        gt_df = gt_df.head(args.max_rows)

    print(f"Well-posed fraction estimation: {len(gt_df)} queries")
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")

    # Load model
    print(f"Loading model on device {args.device}...")
    vlm = load_vlm(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        use_flash_attn=args.use_flash_attn,
    )
    print("Model loaded.")

    results = []
    for i, row in gt_df.iterrows():
        img_path = os.path.join(args.image_dir, row["original_image"])
        query = row["query"]
        gt_fraction = row["fraction_relevant"]

        query_text, dim_word, dim = parse_query_for_fraction(query)
        prompt = FRACTION_PROMPT.format(query_text=query_text, dimension=dim_word)

        if not os.path.exists(img_path):
            print(f"[{i+1}/{len(gt_df)}] Image not found: {img_path}")
            results.append({"query": query, "gt_fraction": gt_fraction, "pred_fraction": None, "pred_raw": ""})
            continue

        image = Image.open(img_path).convert("RGB")

        try:
            response = vlm.generate(image, prompt, max_new_tokens=64, temperature=0.0, do_sample=False)
            pred = parse_fraction(response)
            print(f"[{i+1}/{len(gt_df)}] GT={gt_fraction:.3f} Pred={pred} | {response[:60]}")
        except Exception as e:
            print(f"[{i+1}/{len(gt_df)}] Error: {e}")
            response = ""
            pred = None

        results.append({
            "image": row["original_image"],
            "query": query,
            "gt_fraction": gt_fraction,
            "pred_fraction": pred,
            "pred_raw": response,
        })

    # Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    # Metrics
    valid = results_df.dropna(subset=["pred_fraction"])
    if len(valid) == 0:
        print("No valid predictions!")
        metrics = {"model": args.model, "task": "well_posed_fraction", "total": len(results_df), "valid_predictions": 0}
    else:
        mae = (valid["pred_fraction"] - valid["gt_fraction"]).abs().mean()
        within_10pct = ((valid["pred_fraction"] - valid["gt_fraction"]).abs() / valid["gt_fraction"].clip(lower=0.01) <= 0.10).mean()
        correlation = valid["pred_fraction"].corr(valid["gt_fraction"])

        metrics = {
            "model": args.model,
            "task": "well_posed_fraction",
            "total": len(results_df),
            "valid_predictions": len(valid),
            "mae_fraction": round(mae, 4),
            "accuracy_within_10pct": round(within_10pct, 4),
            "correlation": round(correlation, 4),
        }
        print(f"\n{'='*50}")
        print(f"Results: {len(valid)}/{len(results_df)} valid")
        print(f"MAE (fraction): {mae:.4f}")
        print(f"Accuracy (within 10% relative): {within_10pct*100:.1f}%")
        print(f"Correlation: {correlation:.4f}")

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
