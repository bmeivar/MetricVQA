#!/usr/bin/env python3
"""Build the Metric VQA public release package.

Converts the working files (flat CSVs + npz masks) into the layout VLM
benchmark releases are expected to have in 2026: one JSONL record per query
with structured grounding fields, binary PNG masks, a dataset card, and a
directory layout HuggingFace's dataset viewer can render directly. CSVs are
kept alongside -- the format was never the problem, the missing fields were.

Record schema (superset; Objectron rows omit mask/bbox):
  query_id        itw_0001 / obj_00001
  split           itw | objectron
  image           relative path inside the release
  question        verbatim prompt text (before the float-format instruction)
  answer_cm       ground-truth dimension in centimeters
  axis            length | height | width | depth | diameter  (from query text)
  target_phrase   referring expression used for mask extraction (ITW)
  mask            relative path to binary PNG mask (ITW)
  bbox_xyxy       tight bbox of the mask, pixels (ITW)
  mask_source     sam3_auto_verified | manual_box_prompt  (ITW)
  gt_method       tape_measure | objectron_3d_box
  category        Objectron category (objectron split only)
"""
from __future__ import annotations

import json
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

R = Path(os.environ.get("METRICVQA_ROOT", "."))
OUT = R / "release"

AXIS_PAT = [
    ("diameter", r"diameter"),
    ("height", r"\bheight\b|\btall\b|\bhigh\b"),
    ("width", r"\bwidth\b|\bwide\b"),
    ("depth", r"\bdepth\b|\bdeep\b"),
    ("length", r"\blength\b|\blong\b"),
]

_SLUG = re.compile(r"[^a-z0-9]+")


def slug(s: str) -> str:
    return _SLUG.sub("_", s.lower()).strip("_")


def axis_of(q: str) -> str:
    ql = q.lower()
    for name, pat in AXIS_PAT:
        if re.search(pat, ql):
            return name
    return "size"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def main() -> None:
    (OUT / "itw" / "masks").mkdir(parents=True, exist_ok=True)
    (OUT / "itw" / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "objectron").mkdir(parents=True, exist_ok=True)

    # ── ITW ────────────────────────────────────────────────────────────────
    q = pd.read_csv(R / "questions_v3.csv")
    ph = pd.read_csv(R / "sam_phrases" / "sam_phrases_v1.csv")
    ph["k"] = ph.image + "||" + ph["query"].map(norm)
    phrase_of = dict(zip(ph.k, ph.sam_phrase))

    mb = pd.read_csv(R / "sam_phrases" / "manual_bboxes.csv")
    mb["k"] = mb.image + "||" + mb.sam_phrase.astype(str)
    status_of = dict(zip(mb.k, mb.status))

    rows, missing_mask = [], []
    for i, r in q.iterrows():
        qid = f"itw_{i+1:04d}"
        phrase = phrase_of.get(r.image + "||" + norm(r["query"]))
        rec = {
            "query_id": qid, "split": "itw",
            "image": f"images/{r.image}",
            "question": r["query"].strip(),
            "answer_cm": float(r.answer),
            "axis": axis_of(r["query"]),
            "target_phrase": phrase,
            "mask": None, "bbox_xyxy": None, "mask_source": None,
            "gt_method": "tape_measure",
        }
        if phrase is not None:
            npz = R / "sam_phrases" / "masks" / f"{Path(r.image).stem}__{slug(phrase)}.npz"
            if npz.exists():
                d = np.load(npz, allow_pickle=True)
                m = d["mask"].astype(bool)
                Image.fromarray((m * 255).astype(np.uint8)).save(
                    OUT / "itw" / "masks" / f"{qid}.png", optimize=True)
                ys, xs = np.where(m)
                rec["mask"] = f"masks/{qid}.png"
                rec["bbox_xyxy"] = [int(xs.min()), int(ys.min()),
                                    int(xs.max()) + 1, int(ys.max()) + 1]
                st = status_of.get(r.image + "||" + str(phrase), "")
                rec["mask_source"] = ("manual_box_prompt" if st == "annotated"
                                      else "sam3_auto_verified")
            else:
                missing_mask.append((qid, r.image, phrase))
        else:
            missing_mask.append((qid, r.image, None))
        rows.append(rec)

    with open(OUT / "itw" / "questions.jsonl", "w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")
    pd.DataFrame(rows).to_csv(OUT / "itw" / "questions.csv", index=False)

    for img in sorted((R / "images").glob("image_*.png")):
        tgt = OUT / "itw" / "images" / img.name
        if not tgt.exists():
            tgt.write_bytes(img.read_bytes())

    # ── Objectron ──────────────────────────────────────────────────────────
    oq = pd.read_csv(R / "questions_objectron.csv")
    orecs = []
    for i, r in oq.iterrows():
        orecs.append({
            "query_id": f"obj_{i+1:05d}", "split": "objectron",
            "image": f"images/{r.image}",
            "question": r["query"].strip(),
            "answer_cm": float(r.answer),
            "axis": axis_of(r["query"]),
            "category": str(r.image).split("__")[0],
            "gt_method": "objectron_3d_box",
        })
    with open(OUT / "objectron" / "questions.jsonl", "w") as f:
        for rec in orecs:
            f.write(json.dumps(rec) + "\n")
    pd.DataFrame(orecs).to_csv(OUT / "objectron" / "questions.csv", index=False)

    # ── report ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    print(f"ITW: {len(df)} records | masks {df['mask'].notna().sum()} | "
          f"manual {len(df[df.mask_source == 'manual_box_prompt'])} | "
          f"axes {df.axis.value_counts().to_dict()}")
    print(f"Objectron: {len(orecs)} records | "
          f"{pd.DataFrame(orecs).category.nunique()} categories")
    if missing_mask:
        print(f"MISSING masks ({len(missing_mask)}):")
        for t in missing_mask[:10]:
            print("   ", t)


if __name__ == "__main__":
    main()
