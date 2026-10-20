#!/usr/bin/env python3
"""Convert the released Metric VQA masks into the layout the intervention
builders expect.

The HuggingFace release ships one binary PNG mask per query
(itw/masks/itw_NNNN.png) with grounding fields in itw/questions.jsonl. The
intervention builders predate that layout and read npz masks keyed by
(image stem, slugified target phrase) plus a phrases CSV. This script maps
one onto the other so the full canvas pipeline runs on the public release:

  $METRICVQA_ROOT/itw/questions.jsonl + masks/*.png
      -> $METRICVQA_ROOT/sam_phrases/masks/<stem>__<slug>.npz
      -> $METRICVQA_ROOT/sam_phrases/sam_phrases_v1.csv

Usage:
  METRICVQA_ROOT=/path/to/data python interventions/masks_from_release.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(os.environ.get("METRICVQA_ROOT", "."))
_SLUG = re.compile(r"[^a-z0-9]+")


def slug(s: str, maxlen: int = 48) -> str:
    return _SLUG.sub("_", s.lower()).strip("_")[:maxlen]


def main() -> None:
    itw = ROOT / "itw"
    out_masks = ROOT / "sam_phrases" / "masks"
    out_masks.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in open(itw / "questions.jsonl")]
    rows, written = [], 0
    for r in recs:
        image_name = Path(r["image"]).name
        rows.append({"image": image_name, "query": r["question"],
                     "answer": r["answer_cm"], "sam_phrase": r["target_phrase"]})
        if not r.get("mask") or not r.get("target_phrase"):
            continue
        npz = out_masks / f"{Path(image_name).stem}__{slug(r['target_phrase'])}.npz"
        if npz.exists():
            continue  # several queries share one object's mask
        m = np.asarray(Image.open(itw / r["mask"])) > 0
        x0, y0, x1, y1 = r["bbox_xyxy"]
        np.savez_compressed(npz, mask=m, score=np.float32(1.0),
                            bbox=np.asarray([x0, y0, x1, y1], np.float32),
                            phrase=r["target_phrase"], image=image_name)
        written += 1

    csv = ROOT / "sam_phrases" / "sam_phrases_v1.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"{written} npz masks -> {out_masks}")
    print(f"{len(rows)} rows -> {csv}")


if __name__ == "__main__":
    main()
