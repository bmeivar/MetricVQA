#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vlm_eval.py

Evaluation script for absolute object dimension prediction using
vision-language models (VLMs) from Hugging Face.

Supports multiple VLM families via a registry-based handler architecture:
- LLaVA (llava-hf/*)
- Qwen-VL (Qwen/Qwen2-VL-*, Qwen/Qwen2.5-VL-*)
- BLIP-2 (Salesforce/blip2-*)
- SmolVLM (HuggingFaceTB/SmolVLM-*)
- Idefics2/3 (HuggingFaceM4/idefics*)
- PaliGemma (google/paligemma-*)
- Generic fallback for other models

Features:
- Robust image resolver (recursive search, case-insensitive, HEIC/HEIF support)
- Automatic model handler selection based on model name
- Standardized chat template interface
- MAE and ±10% accuracy metrics

Recommended env additions for HEIC:
  conda install -c conda-forge libheif
  pip install pillow-heif
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List

import pandas as pd
from PIL import Image

from handler_utils import (
    RitsAPIHandler,
    load_vlm,
)

# Optional: HEIC/HEIF support via pillow-heif
_HAS_PILLOW_HEIF = False
try:
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
    _HAS_PILLOW_HEIF = True
except Exception:
    _HAS_PILLOW_HEIF = False

# Hugging Face token retrieval
if "HF_TOKEN" not in os.environ:
    print("WARNING: HF_TOKEN environment variable not set.")
    print("Please run `export HF_TOKEN='your_token'` before executing or use `huggingface-cli login`.")


# -----------------------------
# Parsing / metrics
# -----------------------------

_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def parse_numeric_from_text(text: str) -> Optional[float]:
    """Extract the first float/int appearing in text."""
    if not text:
        return None
    m = _NUM_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def compute_metrics(preds: List[Optional[float]], gts: List[Optional[float]]) -> Dict[str, Any]:
    """MAE (cm) and accuracy within ±10% relative error (skip missing GT/pred)."""
    pairs: List[Tuple[float, float]] = [
        (p, gt) for p, gt in zip(preds, gts) if p is not None and gt is not None
    ]
    if not pairs:
        return {"mae": None, "acc10": None, "count": 0}

    abs_errors = [abs(p - gt) for p, gt in pairs]
    mae = sum(abs_errors) / len(abs_errors)

    within = []
    for p, gt in pairs:
        denom = max(abs(gt), 1e-6)  # avoid divide-by-zero
        within.append(abs(p - gt) <= 0.10 * denom)
    acc10 = sum(within) / len(within)

    return {"mae": mae, "acc10": acc10, "count": len(pairs)}


# -----------------------------
# Image resolution / loading
# -----------------------------

DEFAULT_EXTS = [
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"
]


def resolve_image_path(image_dir: str, image_identifier: str, recursive: bool = True) -> Optional[str]:
    """
    Resolve an image path from CSV `image` column.
    Supports:
      - direct relative paths (with extension)
      - stem-only IDs (try common extensions)
      - recursive lookup under image_dir
      - case-insensitive extensions
    """
    img_dir = Path(image_dir)
    ident = str(image_identifier).strip()

    # 1) If CSV already provides a relative path
    p = img_dir / ident
    if p.is_file():
        return str(p)

    # 2) Try by stem + common extensions in the root
    stem = Path(ident).stem
    for ext in DEFAULT_EXTS:
        cand = img_dir / f"{stem}{ext}"
        if cand.is_file():
            return str(cand)
        cand2 = img_dir / f"{stem}{ext.upper()}"
        if cand2.is_file():
            return str(cand2)

    if not recursive:
        return None

    # 3) Fast recursive search: check common image subdirs first
    for subdir in ["images", "image", "img", "pics"]:
        for ext in DEFAULT_EXTS:
            cand = img_dir / subdir / f"{stem}{ext}"
            if cand.is_file():
                return str(cand)
            cand2 = img_dir / subdir / f"{stem}{ext.upper()}"
            if cand2.is_file():
                return str(cand2)

    # 4) Full recursive search as last resort (may be slow)
    patterns = []
    for ext in DEFAULT_EXTS:
        patterns.append(f"{stem}{ext}")
        patterns.append(f"{stem}{ext.upper()}")

    for pat in patterns:
        hits = list(img_dir.rglob(pat))
        if hits:
            return str(hits[0])

    # 5) As a final last resort, match by stem anywhere regardless of extension
    stem_hits = []
    for f in img_dir.rglob("*"):
        if f.is_file() and f.stem == stem:
            stem_hits.append(f)
    if stem_hits:
        return str(stem_hits[0])

    return None


def maybe_convert_heic_to_jpg(image_path: str, cache_dir: Optional[str]) -> str:
    """
    If image is HEIC/HEIF and cache_dir is provided, convert to JPG once and return JPG path.
    Otherwise return original path.
    """
    p = Path(image_path)
    suf = p.suffix.lower()
    if suf not in [".heic", ".heif"]:
        return image_path
    if cache_dir is None:
        return image_path

    if not _HAS_PILLOW_HEIF:
        return image_path

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out_path = cache / (p.stem + ".jpg")

    if out_path.is_file():
        return str(out_path)

    img = Image.open(p).convert("RGB")
    img.save(out_path, quality=95)
    return str(out_path)


def open_image_as_rgb(image_path: str) -> Image.Image:
    """
    Open image into RGB PIL.Image.
    If HEIC and pillow-heif not installed, raise with a clear message.
    """
    p = Path(image_path)
    if p.suffix.lower() in [".heic", ".heif"] and not _HAS_PILLOW_HEIF:
        raise RuntimeError(
            "HEIC/HEIF file detected but pillow-heif is not installed.\n"
            "Install:\n"
            "  conda install -c conda-forge libheif\n"
            "  pip install pillow-heif\n"
        )
    img = Image.open(image_path)
    return img.convert("RGB")

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a VLM on absolute dimension questions (cm)."
    )
    parser.add_argument("--model", type=str, required=True, help="HF model repo id")
    parser.add_argument(
        "--task",
        type=str,
        default="image-text-to-text",
        choices=["image-to-text", "image-text-to-text"],
        help="(Legacy) Transformers pipeline task. Ignored for direct model loading.",
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="HuggingFace dataset repo id (e.g. bmeivar/metric-vqa). "
                             "If set, --csv and --image-dir are auto-resolved from the downloaded dataset.")
    parser.add_argument("--csv", type=str, default=None, help="CSV with columns: image, query, answer")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory containing images")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for predictions/metrics")
    parser.add_argument("--device", type=int, default=-1, help="-1 CPU, 0 first GPU, etc.")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"],
                        help="torch_dtype for model weights (auto is fine).")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling (otherwise greedy)")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for images under --image-dir")
    parser.add_argument(
        "--instruction",
        type=str,
        default="Answer with a single float in centimeters.",
        help="Prepended instruction to encourage numeric-only answers.",
    )
    parser.add_argument(
        "--blank-image",
        action="store_true",
        help="Replace the real image with a solid 448x448 canvas. Used to "
             "measure language-only (no-vision) performance as a lower bound.",
    )
    parser.add_argument(
        "--blank-color",
        type=int,
        default=0,
        help="Gray value (0-255) for the --blank-image canvas. Default 0 (black). "
             "Use 127 to match the gray fill of context_object_only.",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Skip image input entirely and call the handler's "
             "generate_text_only(). Probes the pure LM prior with no vision "
             "tokens. Stronger probe than --blank-image.",
    )
    parser.add_argument(
        "--heic-cache-dir",
        type=str,
        default=None,
        help="Optional directory to cache HEIC/HEIF converted JPGs (speeds up repeated runs).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading models with custom code (required for some models).",
    )
    parser.add_argument(
        "--use-flash-attn",
        action="store_true",
        help="Use Flash Attention 2 if available (requires flash-attn package).",
    )
    parser.add_argument("--use_lora", action="store_true", help="Whether to apply LoRA weights (if available)")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA weights (if --use_lora is set)")
    parser.add_argument(
        "--cleanup-cache",
        action="store_true",
        help="Delete the HuggingFace cached model after evaluation to free disk space.",
    )
    parser.add_argument(
        "--use-rits",
        action="store_true",
        help="Use RITS API instead of local HuggingFace model.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL (RITS or LiteLLM proxy)",
    )
    parser.add_argument(
        "--auth-style",
        type=str,
        default="rits",
        choices=["rits", "bearer"],
        help="Auth style: 'rits' (RITS_API_KEY header) or 'bearer' (Authorization: Bearer, for LiteLLM)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit to first N rows (useful for testing with --use-rits)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force fresh download of dataset, ignore cache",
    )

    args = parser.parse_args()

    # Resolve dataset from HuggingFace if --dataset is provided
    # Use snapshot_download only - avoid load_dataset() which makes extra API calls
    if args.dataset:
        from huggingface_hub import snapshot_download
        import time

        print(f"Downloading/caching dataset: {args.dataset}")
        try:
            if args.force_download:
                # Resume download with retries on rate limit
                print(f"Resuming/downloading dataset...")
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        local_dir = snapshot_download(
                            repo_id=args.dataset,
                            repo_type="dataset",
                            local_files_only=False,
                            force_download=False  # Allow resume, don't force re-download
                        )
                        print(f"Dataset ready at: {local_dir}")
                        break
                    except Exception as e:
                        if "429" in str(e) or "Too Many Requests" in str(e):
                            wait_time = 60 * (attempt + 1)  # 60s, 120s, 180s
                            print(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                            time.sleep(wait_time)
                            if attempt == max_retries - 1:
                                raise
                        else:
                            raise
            else:
                # First attempt: try with local files only to avoid rate limits if already cached
                try:
                    local_dir = snapshot_download(
                        repo_id=args.dataset,
                        repo_type="dataset",
                        local_files_only=True
                    )
                    print(f"Using cached dataset at: {local_dir}")
                except Exception:
                    
                    # If not cached locally, download from HF
                    print(f"Dataset not cached, downloading from HuggingFace...")
                    local_dir = snapshot_download(
                        repo_id=args.dataset,
                        repo_type="dataset",
                        local_files_only=False
                    )
                    print(f"Dataset cached at: {local_dir}")

            if args.csv is None:
                # Look for any CSV file in the dataset directory
                csv_candidates = []
                for filename in os.listdir(local_dir):
                    if filename.endswith('.csv'):
                        csv_candidates.append(os.path.join(local_dir, filename))

                if csv_candidates:
                    args.csv = csv_candidates[0]  # Use first CSV found
                    print(f"Found CSV: {os.path.basename(args.csv)}")
                else:
                    raise ValueError(f"No CSV file found in {local_dir}")

            if args.image_dir is None:
                args.image_dir = os.path.join(local_dir, "images")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            parser.error(f"Failed to download dataset {args.dataset}: {e}")
    elif args.csv is None or (args.image_dir is None and not args.blank_image and not args.text_only):
        parser.error("Either --dataset, or --csv plus --image-dir (or --blank-image or --text-only) are required.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load CSV
    print(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    # Accept the released Metric VQA schema (question / answer_cm) as well as
    # the legacy working schema (query / answer).
    df = df.rename(columns={"question": "query", "answer_cm": "answer"})
    required_cols = {"image", "query", "answer"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")
    print(f"Loaded {len(df)} rows from CSV")

    if args.max_rows:
        df = df.head(args.max_rows)
        print(f"Limited to first {args.max_rows} rows")

    # Ground truths
    def to_float(x):
        try:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return None
            s = str(x).strip()
            if s == "":
                return None
            return float(s)
        except Exception:
            return None

    ground_truths: List[Optional[float]] = [to_float(a) for a in df["answer"]]

    # Load model
    if args.use_rits:
        print(f"Connecting to API: {args.api_base} (auth_style={args.auth_style})")
        vlm = RitsAPIHandler(
            model_name=args.model,
            api_base=args.api_base,
            auth_style=args.auth_style,
        )
        vlm.load()
        print("Connected to API. Starting inference...")
    else:
        print(f"Loading model {args.model} on device {args.device} ...")
        vlm = load_vlm(
            model_name=args.model,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            use_flash_attn=args.use_flash_attn,
            use_lora=args.use_lora,
            lora_path=args.lora_path
        )
        print("Model loaded. Starting inference...")

    total = len(df)
    predicted_texts: List[str] = []
    predicted_vals: List[Optional[float]] = []
    resolved_paths: List[Optional[str]] = []

    blank_canvas = Image.new("RGB", (448, 448), (args.blank_color,) * 3) if args.blank_image else None

    for i, (img_id, q) in enumerate(zip(df["image"], df["query"]), start=1):
        if args.text_only:
            img_path_used = "<text-only>"
            resolved_paths.append(img_path_used)
            image = None
        elif args.blank_image:
            img_path_used = "<blank>"
            resolved_paths.append(img_path_used)
            image = blank_canvas
        else:
            img_path = resolve_image_path(args.image_dir, str(img_id), recursive=args.recursive)
            if img_path is None:
                print(f"[Row {i}] Image not found: {img_id}")
                predicted_texts.append("")
                predicted_vals.append(None)
                resolved_paths.append(None)
                continue

            # Optional HEIC -> JPG caching
            img_path_used = maybe_convert_heic_to_jpg(img_path, args.heic_cache_dir)
            resolved_paths.append(img_path_used)

            try:
                image = open_image_as_rgb(img_path_used)
            except Exception as e:
                print(f"[Row {i}] Error loading image: {e}")
                predicted_texts.append("")
                predicted_vals.append(None)
                continue

        # Build prompt
        question = str(q).strip()
        prompt = vlm.format_prompt(args.instruction, question)

        try:
            if args.text_only:
                out_text = vlm.generate_text_only(
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=args.do_sample,
                )
            else:
                out_text = vlm.generate(
                    image=image,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=args.do_sample,
                )
        except Exception as e:
            print(f"[Row {i}] Error during inference: {e}")
            out_text = ""

        predicted_texts.append(out_text)
        predicted_vals.append(parse_numeric_from_text(out_text))

        # Print progress every sample or every 10 samples for large datasets
        progress_interval = 1 if (args.max_rows and args.max_rows <= 10) else 10
        if i % progress_interval == 0 or i == total:
            print(f"Processed {i}/{total}")

    # Metrics
    metrics = compute_metrics(predicted_vals, ground_truths)

    # Save outputs
    out_df = df.copy()
    out_df["resolved_path"] = resolved_paths
    out_df["predicted_text"] = predicted_texts
    out_df["predicted_value_cm"] = predicted_vals

    pred_csv_path = os.path.join(args.output_dir, "predictions.csv")
    out_df.to_csv(pred_csv_path, index=False)

    pred_jsonl_path = os.path.join(args.output_dir, "predictions.jsonl")
    with open(pred_jsonl_path, "w", encoding="utf-8") as f:
        for _, row in out_df.iterrows():
            rec = {
                "image": row["image"],
                "resolved_path": row["resolved_path"],
                "query": row["query"],
                "answer": row["answer"],
                "predicted_text": row["predicted_text"],
                "predicted_value_cm": row["predicted_value_cm"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. Metrics: {metrics}")
    print(f"Saved: {pred_csv_path}")
    print(f"Saved: {pred_jsonl_path}")
    print(f"Saved: {metrics_path}")

    # Optional: clean up cached model to free disk space
    if args.cleanup_cache:
        import shutil
        model_dir = "models--" + args.model.replace("/", "--")
        candidates = []
        # Honor HF env vars first, then fall back to ~/.cache.
        for env in ("HF_HUB_CACHE", "HF_HOME"):
            base = os.environ.get(env)
            if not base:
                continue
            if env == "HF_HOME":
                candidates.append(os.path.join(base, "hub", model_dir))
            else:
                candidates.append(os.path.join(base, model_dir))
        candidates.append(os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub", model_dir
        ))
        cleaned = False
        for cache_dir in candidates:
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir)
                print(f"Cleaned up cached model: {cache_dir}")
                cleaned = True
        if not cleaned:
            print(f"Cache directory not found (already clean?): {candidates[0]}")


if __name__ == "__main__":
    main()
