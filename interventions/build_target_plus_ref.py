"""
Builder for the `target_plus_ref` context variant.

Composition:
  - Load the original image.
  - Load the SAM mask for the TARGET query.
  - Load the SAM mask for the REFERENCE query (another queried object on the
    same image).
  - Combined mask = target_mask OR reference_mask.
  - Gray-fill all pixels outside combined_mask (value 127, optional 5px feather).

Both objects stay in their **original spatial positions** — preserving relative
size and scene geometry, which are necessary cues for metric estimation. Only
the rest of the scene context is removed.

A (target, ref) pair is constructed for every ordered (image, target, ref)
where ref is another queried object on the same image. So an image with k
query objects produces k*(k-1) pairs.

For analysis, the comparison set is:
  - context_full         (Phase C baseline, all context visible)
  - context_object_only  (Phase C, only target visible)
  - target_plus_ref      (this experiment, target + one specific reference visible)
"""

from __future__ import annotations

import re
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

REPO = Path(os.environ.get("METRICVQA_ROOT",
                            Path(__file__).resolve().parent.parent))

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(s: str, maxlen: int = 48) -> str:
    return _SLUG_RE.sub("_", s.lower()).strip("_")[:maxlen]


def mask_path_for(image_name: str, phrase: str, masks_dir: Path) -> Path:
    stem = Path(image_name).stem
    return masks_dir / f"{stem}__{slug(phrase)}.npz"


def load_mask(image_name: str, phrase: str,
              masks_dir: Path | None = None) -> np.ndarray | None:
    masks_dir = masks_dir or (REPO / "sam_phrases" / "masks")
    p = mask_path_for(image_name, phrase, masks_dir)
    if not p.exists():
        return None
    data = np.load(p, allow_pickle=True)
    m = data["mask"]
    if not m.any():
        return None
    return m.astype(bool)


def build_target_plus_ref(image_name: str, target_phrase: str,
                          ref_phrase: str,
                          images_dir: Path,
                          masks_dir: Path,
                          fill_value: int = 127,
                          feather_px: int = 5) -> Optional[Image.Image]:
    """Returns a PIL image with both target and reference visible (in their
    original positions) and everything else gray-filled.

    Returns None if either mask is missing or empty.
    """
    target = load_mask(image_name, target_phrase, masks_dir)
    if target is None:
        return None
    ref = load_mask(image_name, ref_phrase, masks_dir)
    if ref is None:
        return None

    img_path = images_dir / image_name
    if not img_path.exists():
        return None
    img = np.array(Image.open(img_path).convert("RGB"))

    combined = target | ref   # union
    if combined.shape != img.shape[:2]:
        from scipy.ndimage import zoom
        zh = img.shape[0] / combined.shape[0]
        zw = img.shape[1] / combined.shape[1]
        combined = zoom(combined.astype(np.float32), (zh, zw), order=0) > 0.5

    if feather_px > 0:
        alpha = gaussian_filter(combined.astype(np.float32), sigma=feather_px)
        alpha = np.clip(alpha, 0.0, 1.0)
        gray = np.full_like(img, fill_value)
        out = (alpha[..., None] * img + (1 - alpha[..., None]) * gray).astype(np.uint8)
    else:
        out = np.full_like(img, fill_value)
        out[combined] = img[combined]
    return Image.fromarray(out)


def enumerate_pairs(queries_df,
                    phrases_df,
                    masks_dir: Path,
                    target_query_col: str = "query",
                    target_answer_col: str = "answer",
                    sam_phrase_col: str = "sam_phrase",
                    note_col: str | None = "note") -> List[dict]:
    """All ordered (target, ref) pairs where the target is a real query and the
    ref is any object with a valid SAM mask on the same image.

    Rows in `phrases_df` tagged with `note == "reference"` are reference-only —
    they share the query/answer columns with the actual target but their
    sam_phrase points at a different object. We use them as REFERENCES but
    never as TARGETS.
    """
    import pandas as pd
    df = phrases_df.copy()
    df = df[df[sam_phrase_col].notna()]
    if note_col not in df.columns:
        df[note_col] = ""
    df[note_col] = df[note_col].fillna("")

    # Build per-image lists of valid (phrase, is_reference_only) entries with masks
    pairs = []
    for img, g in df.groupby("image"):
        # All phrases in this image that have a mask
        entries = []
        for _, r in g.iterrows():
            if not mask_path_for(img, r[sam_phrase_col], masks_dir).exists():
                continue
            entries.append({
                "phrase": r[sam_phrase_col],
                "query": r[target_query_col],
                "answer": r[target_answer_col],
                "is_reference_only": (str(r[note_col]).strip().lower() == "reference"),
            })
        if len(entries) < 2:
            continue
        # Targets: only NOT is_reference_only rows. Refs: any other entry.
        # Dedup target phrases (same phrase may appear once as target, once as ref).
        target_phrases_seen = set()
        targets = []
        for e in entries:
            if e["is_reference_only"]:
                continue
            if e["phrase"] in target_phrases_seen:
                continue
            target_phrases_seen.add(e["phrase"])
            targets.append(e)
        # Ref pool = all entries with this image (including reference-only)
        for t in targets:
            for r in entries:
                if r["phrase"] == t["phrase"]:
                    continue
                pairs.append({
                    "image": img,
                    "target_query": t["query"],
                    "target_answer": t["answer"],
                    "target_phrase": t["phrase"],
                    "ref_query": r["query"],
                    "ref_answer": r["answer"],
                    "ref_phrase": r["phrase"],
                    "ref_is_reference_only": r["is_reference_only"],
                })
    return pairs
