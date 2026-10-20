"""
SAM3 text-grounded masks for metric-VQA in-the-wild queries.

For each unique (image, sam_phrase) pair in sam_phrases_v[1|2].csv:
  - load image, ask SAM3 to localize the phrase
  - keep the top-scoring mask (above threshold)
  - save mask (bool HxW) + score + bbox in a .npz

Output layout:
  sam_phrases/masks/<image_stem>__<phrase_slug>.npz
    keys: mask (bool HxW), score (float), bbox (xyxy float32),
          phrase (str), image (str)

Resumable: skips already-existing .npz files.

Usage:
  PY=python  # point at an env with transformers>=4.58 (Sam3Model)
  $PY scripts/run_sam3_masks.py --phrases sam_phrases/sam_phrases_v1.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
IMAGES_DIR_DEFAULT = REPO / "images"
PHRASES_DEFAULT = REPO / "sam_phrases" / "sam_phrases_v1.csv"
MASKS_OUT_DEFAULT = REPO / "sam_phrases" / "masks"

MODEL_ID = "facebook/sam3"
THRESHOLD = 0.3
MASK_THRESHOLD = 0.5
MAX_TOKENS = 32

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(s: str, maxlen: int = 48) -> str:
    s = _SLUG_RE.sub("_", s.lower()).strip("_")
    return s[:maxlen]


def truncate_text(processor, text: str) -> str:
    tok = processor.tokenizer
    ids = tok(text, add_special_tokens=False, truncation=True,
              max_length=MAX_TOKENS - 2)["input_ids"]
    return tok.decode(ids, skip_special_tokens=True)


@torch.no_grad()
def run_sam3(processor, model, img: Image.Image, text: str, device: str):
    text = truncate_text(processor, text)
    inp = processor(images=img, text=text, return_tensors="pt").to(device)
    out = model(**inp)
    H, W = img.size[1], img.size[0]
    results = processor.post_process_instance_segmentation(
        out, threshold=THRESHOLD, mask_threshold=MASK_THRESHOLD,
        target_sizes=[(H, W)],
    )[0]
    scores = results.get("scores")
    masks = results.get("masks")
    if scores is None or masks is None or len(scores) == 0:
        return None, None
    masks_np = masks.detach().cpu().numpy().astype(bool)
    scores_np = scores.detach().cpu().float().numpy()
    return masks_np, scores_np


def mask_bbox_xyxy(m: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(m)
    if ys.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()),
            float(xs.max() + 1), float(ys.max() + 1))


def out_path_for(out_root: Path, image_name: str, phrase: str) -> Path:
    stem = Path(image_name).stem
    return out_root / f"{stem}__{slug(phrase)}.npz"


def save_mask(out_path: Path, mask: np.ndarray, score: float, phrase: str,
              image_name: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bbox = np.asarray(mask_bbox_xyxy(mask), dtype=np.float32)
    np.savez_compressed(
        str(out_path),
        mask=mask.astype(bool),
        score=np.float32(score),
        bbox=bbox,
        phrase=phrase,
        image=image_name,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrases", type=Path, default=PHRASES_DEFAULT)
    ap.add_argument("--images-dir", type=Path, default=IMAGES_DIR_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=MASKS_OUT_DEFAULT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N pairs (smoke test)")
    args = ap.parse_args()

    phr = pd.read_csv(args.phrases)
    if "sam_phrase" not in phr.columns:
        sys.exit("phrases csv must have a 'sam_phrase' column")
    pairs = (phr[["image", "sam_phrase"]]
             .dropna()
             .drop_duplicates()
             .reset_index(drop=True))
    if args.limit:
        pairs = pairs.head(args.limit)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # filter already-done
    todo = []
    skipped = 0
    for _, row in pairs.iterrows():
        op = out_path_for(args.out_dir, row.image, row.sam_phrase)
        if op.exists():
            skipped += 1
            continue
        todo.append((row.image, row.sam_phrase, op))
    print(f"total pairs: {len(pairs)}  skipped (exists): {skipped}  todo: {len(todo)}",
          flush=True)
    if not todo:
        return

    print(f"loading {MODEL_ID} on {args.device} (bf16) ...", flush=True)
    from transformers import Sam3Model, Sam3Processor
    processor = Sam3Processor.from_pretrained(MODEL_ID)
    model = Sam3Model.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
    ).to(args.device).eval()

    n_empty = 0
    pbar = tqdm(todo, desc="SAM3", dynamic_ncols=True)
    for image_name, phrase, out_path in pbar:
        img_path = args.images_dir / image_name
        if not img_path.exists():
            print(f"[warn] missing image: {img_path}", flush=True)
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"[err] cannot open {img_path}: {exc}", flush=True)
            continue
        try:
            masks, scores = run_sam3(processor, model, img, phrase, args.device)
        except Exception as exc:
            print(f"[err] {image_name} '{phrase}': {exc}", flush=True)
            continue

        if masks is None:
            n_empty += 1
            # save an empty mask with score=0 so downstream knows this pair was tried
            save_mask(out_path,
                      np.zeros((img.size[1], img.size[0]), dtype=bool),
                      0.0, phrase, image_name)
            pbar.set_postfix(empty=n_empty)
            continue

        # pick the highest-scoring mask
        top = int(np.argmax(scores))
        save_mask(out_path, masks[top], float(scores[top]), phrase, image_name)
        pbar.set_postfix(empty=n_empty, last=f"{scores[top]:.2f}")


if __name__ == "__main__":
    main()
