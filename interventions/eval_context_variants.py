"""
Evaluate one VLM on the 5 context-counterfactual variants of every
(image, query) in the in-the-wild benchmark.

Variants (per row, generated in-memory; see scripts/build_context_variants.py):
  full / object_only / object_ring / wrong_object / random_mask

For each variant we write a separate output dir:
  results/context_<variant>/<model_short>/predictions.csv

Rows where the SAM mask is missing (e.g. image_005 the table) are skipped
across all five variants so the comparison stays paired.

Reuses metric_vqa's existing handler registry (handler_utils.load_vlm).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

REPO = Path(os.environ.get("METRICVQA_ROOT",
                            Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling builders
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # vlm_eval, handler_utils, release_data
from release_data import ITW_CSV, ITW_IMAGES, PHRASES_CSV, MASKS_DIR, RESULTS, load_itw  # noqa: E402

from handler_utils import load_vlm  # noqa: E402
from vlm_eval import compute_metrics, parse_numeric_from_text  # noqa: E402

from build_context_variants import build_variants  # noqa: E402

VARIANTS = ["full", "object_only", "object_ring", "wrong_object", "random_mask"]


def model_short(model_id: str) -> str:
    return model_id.split("/")[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id (e.g. Qwen/Qwen2.5-VL-7B-Instruct)")
    ap.add_argument("--csv", default=str(ITW_CSV))
    ap.add_argument("--phrases", default=str(PHRASES_CSV))
    ap.add_argument("--images-dir", default=str(ITW_IMAGES))
    ap.add_argument("--masks-dir", default=str(MASKS_DIR))
    ap.add_argument("--out-root", default=str(RESULTS))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--dtype", default="fp16", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--instruction", default="Answer with a single float in centimeters.")
    ap.add_argument("--max-rows", type=int, default=None, help="for smoke testing")
    ap.add_argument("--variants", nargs="*", default=VARIANTS,
                    help="subset of variants to run (default: all 5)")
    args = ap.parse_args()

    short = model_short(args.model)
    print(f"model: {args.model}  short: {short}")
    print(f"variants: {args.variants}")

    # ----- load queries and phrases -----
    df = load_itw(args.csv)
    phr = pd.read_csv(args.phrases)
    if "sam_phrase" not in df.columns:
        df = df.merge(phr[["image", "query", "sam_phrase"]],
                      on=["image", "query"], how="left")

    if args.max_rows:
        df = df.head(args.max_rows)

    n_with_phrase = df["sam_phrase"].notna().sum()
    print(f"rows: {len(df)}  rows with phrase: {n_with_phrase}")

    # ----- precompute mask availability so we have a paired set -----
    paired_ok = []
    skipped_no_phrase = 0
    skipped_no_mask = 0
    for _, row in df.iterrows():
        if pd.isna(row["sam_phrase"]):
            skipped_no_phrase += 1
            paired_ok.append(False)
            continue
        v = build_variants(row["image"], row["sam_phrase"],
                           phrases_df=phr, images_dir=Path(args.images_dir),
                           masks_dir=Path(args.masks_dir))
        if v is None:
            skipped_no_mask += 1
            paired_ok.append(False)
            continue
        paired_ok.append(True)

    df["paired_ok"] = paired_ok
    print(f"paired_ok: {sum(paired_ok)} / {len(df)}  "
          f"(skipped: {skipped_no_phrase} no_phrase, {skipped_no_mask} no_mask)")

    # ----- prepare output dirs -----
    out_dirs: dict[str, Path] = {}
    for v in args.variants:
        d = Path(args.out_root) / f"context_{v}" / short
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[v] = d
        print(f"  → {d}")

    # ----- load model once -----
    print(f"loading {args.model} (device={args.device}, dtype={args.dtype}) ...", flush=True)
    t0 = time.time()
    vlm = load_vlm(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        use_flash_attn=False,
        use_lora=False,
        lora_path=None,
    )
    print(f"loaded in {time.time() - t0:.1f}s")

    # ----- iterate rows × variants -----
    # buffer one row per variant
    per_variant_rows: dict[str, list[dict]] = {v: [] for v in args.variants}

    pbar = tqdm(df.itertuples(index=False), total=len(df),
                desc=f"eval {short}", dynamic_ncols=True)
    for row in pbar:
        image = row.image
        query = row.query
        gt = row.answer
        phrase = row.sam_phrase if isinstance(row.sam_phrase, str) else None

        # build variants (uses image cache internally)
        if not row.paired_ok:
            variants = None
        else:
            variants = build_variants(image, phrase, phrases_df=phr,
                                      images_dir=Path(args.images_dir),
                                      masks_dir=Path(args.masks_dir))

        wrong_alt = (variants or {}).get("_wrong_object_phrase", "")

        prompt = vlm.format_prompt(args.instruction, str(query).strip())

        for v_name in args.variants:
            rec = {
                "image": image,
                "query": query,
                "answer": gt,
                "sam_phrase": phrase or "",
                "variant": v_name,
                "wrong_object_alt": wrong_alt if v_name == "wrong_object" else "",
                "predicted_text": "",
                "predicted_value_cm": None,
                "skipped": False,
            }

            if variants is None or v_name not in variants:
                rec["skipped"] = True
                per_variant_rows[v_name].append(rec)
                continue

            img = variants[v_name]
            try:
                out_text = vlm.generate(
                    image=img,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                )
            except Exception as exc:
                print(f"[err] {image} / {query} / {v_name}: {exc}", flush=True)
                out_text = ""

            rec["predicted_text"] = out_text
            rec["predicted_value_cm"] = parse_numeric_from_text(out_text)
            per_variant_rows[v_name].append(rec)

    # ----- save per-variant CSVs and metrics -----
    for v_name, rows in per_variant_rows.items():
        odf = pd.DataFrame(rows)
        odf.to_csv(out_dirs[v_name] / "predictions.csv", index=False)

        # only evaluate metrics on rows we actually ran (skipped=False)
        ran = odf[~odf["skipped"]]
        gts = ran["answer"].tolist()
        preds = ran["predicted_value_cm"].tolist()
        metrics = compute_metrics(preds, gts)
        metrics["n_total"] = int(len(odf))
        metrics["n_ran"] = int(len(ran))
        metrics["n_skipped"] = int(odf["skipped"].sum())
        with open(out_dirs[v_name] / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  [{v_name}] mae={metrics.get('mae'):.2f}  acc10={metrics.get('acc10'):.3f}  "
              f"ran={metrics['n_ran']}/{metrics['n_total']}")


if __name__ == "__main__":
    main()
