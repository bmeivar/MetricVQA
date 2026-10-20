#!/usr/bin/env python3
"""Text-only baseline: run VLM queries with a blank (uninformative) image.

Tests whether VLMs extract any visual information or just rely on language priors.
For Objectron, there are only 27 unique queries — we run each once and map to all rows.
For in-the-wild, we run all 331 queries individually.

Usage:
    python text_only_baseline.py --model "Qwen/Qwen2.5-VL-7B-Instruct" \
        --csv questions_objectron.csv --output-dir results/textonly_objectron_Qwen2.5-VL-7B \
        --device 0 --dtype bf16
"""

import argparse, csv, json, os, re, sys
import numpy as np
from pathlib import Path
from PIL import Image

# Blank gray image (provides no visual information)
BLANK_IMAGE = Image.new("RGB", (384, 384), (128, 128, 128))

# Reuse vlm_eval infrastructure
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_eval import load_vlm, parse_numeric_from_text


def main():
    parser = argparse.ArgumentParser(description="Text-only VLM baseline")
    parser.add_argument("--model", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    # Load CSV
    rows = []
    with open(args.csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("image", "").strip():
                rows.append(r)
    print(f"Loaded {len(rows)} rows from {args.csv}")

    # Find unique queries
    unique_queries = sorted(set(r["query"] for r in rows))
    print(f"Unique queries: {len(unique_queries)}")

    # Load model
    vlm = load_vlm(args.model, device=args.device, dtype=args.dtype,
                    trust_remote_code=args.trust_remote_code)

    # Run each unique query with blank image
    query_to_response = {}
    for i, q in enumerate(unique_queries, 1):
        prompt = f"{q} Answer with a single float number in centimeters."
        try:
            text = vlm.generate(
                image=BLANK_IMAGE,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )
        except Exception as e:
            print(f"  [{i}] Error: {e}")
            text = ""
        val = parse_numeric_from_text(text)
        query_to_response[q] = (text, val)
        print(f"  [{i}/{len(unique_queries)}] {q[:50]:50s} -> {text!r:20s} -> {val}")

    # Map to all rows and save
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "predictions.csv")
    out_jsonl = os.path.join(args.output_dir, "predictions.jsonl")

    preds = []
    with open(out_csv, "w", newline="") as fc, open(out_jsonl, "w") as fj:
        writer = csv.writer(fc)
        writer.writerow(["image", "query", "answer", "resolved_path",
                         "predicted_text", "predicted_value_cm"])
        for r in rows:
            q = r["query"]
            text, val = query_to_response[q]
            gt = float(r["answer"])
            writer.writerow([r["image"], q, gt, "", text, val if val is not None else ""])
            preds.append({"gt": gt, "pred": val})
            fj.write(json.dumps({
                "image": r["image"], "query": q, "answer": gt,
                "predicted_text": text, "predicted_value_cm": val
            }) + "\n")

    # Compute metrics
    valid = [(p["gt"], p["pred"]) for p in preds if p["pred"] is not None]
    if valid:
        gts, ps = zip(*valid)
        gts, ps = np.array(gts), np.array(ps)
        errs = np.abs(ps - gts)
        rel = errs / gts
        metrics = {
            "mae": float(errs.mean()),
            "median_ae": float(np.median(errs)),
            "acc10": float((rel <= 0.10).mean()),
            "acc20": float((rel <= 0.20).mean()),
            "count": len(valid),
            "n_failed": len(preds) - len(valid),
        }
    else:
        metrics = {"mae": None, "acc10": None, "count": 0, "n_failed": len(preds)}

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults: MAE={metrics['mae']:.2f}, Acc@10%={metrics['acc10']*100:.1f}%, "
          f"valid={metrics['count']}/{len(preds)}")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
