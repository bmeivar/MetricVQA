"""
Build the 5 context-counterfactual image variants for a single (image, query).

Variants (all returned at the original image size):
  - full          : original.
  - object_only   : gray fill (127) outside SAM mask, hard edge, no dilation.
  - object_ring   : gray fill outside dilate(mask, RING_PX), with a 5px feather.
  - wrong_object  : gray fill outside a DIFFERENT object's mask on the same image
                    (drawn from another query of the same image).
                    If the image only has one phrase, falls back to random_mask.
  - random_mask   : gray fill outside a random axis-aligned rectangle that has
                    the same pixel area as the object mask.

Indexed by (image, sam_phrase) through `sam_phrases/masks/<stem>__<slug>.npz`.

The slug must match `run_sam3_masks.slug(phrase)`.

Returns a dict of {variant_name: PIL.Image.Image}.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter, generate_binary_structure

REPO = Path(__file__).resolve().parent.parent
IMAGES_DIR = REPO / "images"
MASKS_DIR = REPO / "sam_phrases" / "masks"
PHRASES_CSV = REPO / "sam_phrases" / "sam_phrases_v1.csv"

# allow callers to override at module import time:
#   import build_context_variants as bcv; bcv.MASKS_DIR = Path("...")
# OR pass masks_dir into build_variants() per-call (added below).

GRAY = 127
RING_PX = 50
FEATHER_PX = 5
RANDOM_SEED_BASE = 1337

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(s: str, maxlen: int = 48) -> str:
    s = _SLUG_RE.sub("_", s.lower()).strip("_")
    return s[:maxlen]


def mask_path_for(image_name: str, phrase: str, masks_dir: Path = None) -> Path:
    stem = Path(image_name).stem
    return (masks_dir or MASKS_DIR) / f"{stem}__{slug(phrase)}.npz"


def load_mask(image_name: str, phrase: str,
              masks_dir: Path | None = None) -> np.ndarray | None:
    p = mask_path_for(image_name, phrase, masks_dir)
    if not p.exists():
        return None
    data = np.load(p, allow_pickle=True)
    m = data["mask"]
    if not m.any():
        return None
    return m.astype(bool)


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask.copy()
    struct = generate_binary_structure(2, 1)
    return binary_dilation(mask, structure=struct, iterations=px)


def build_alpha(mask: np.ndarray, feather_px: int) -> np.ndarray:
    if feather_px <= 0:
        return mask.astype(np.float32)
    struct = generate_binary_structure(2, 1)
    dilated = binary_dilation(mask, structure=struct, iterations=feather_px)
    alpha = dilated.astype(np.float32)
    alpha = gaussian_filter(alpha, sigma=feather_px / 2.0)
    alpha[mask] = 1.0
    return alpha.clip(0, 1)


def apply_keep_mask(img_np: np.ndarray, keep: np.ndarray,
                    feather_px: int = 0, gray: int = GRAY) -> np.ndarray:
    """Outside keep -> gray; supports soft feather edge."""
    alpha = build_alpha(keep, feather_px)[:, :, None]
    img_f = img_np.astype(np.float32)
    fill = np.full_like(img_f, float(gray))
    out = img_f * alpha + fill * (1.0 - alpha)
    return out.clip(0, 255).astype(np.uint8)


def random_rectangle(H: int, W: int, target_area: int,
                     rng: np.random.Generator) -> np.ndarray:
    """A random axis-aligned rectangle whose pixel area equals target_area
    (within ±1px due to rounding). Aspect ratio drawn uniformly in [0.5, 2.0]."""
    target_area = max(target_area, 4)
    target_area = min(target_area, H * W)
    ar = float(rng.uniform(0.5, 2.0))  # h/w
    # h*w = area; h = ar*w -> w = sqrt(area / ar)
    w = int(np.sqrt(target_area / ar))
    h = int(ar * w)
    w = max(1, min(w, W))
    h = max(1, min(h, H))
    x0 = int(rng.integers(0, max(1, W - w + 1)))
    y0 = int(rng.integers(0, max(1, H - h + 1)))
    m = np.zeros((H, W), dtype=bool)
    m[y0:y0 + h, x0:x0 + w] = True
    return m


def pick_alternate_phrase(image_name: str, this_phrase: str,
                          all_phrases_for_image: list[str],
                          rng: np.random.Generator,
                          masks_dir: Path | None = None) -> str | None:
    """Return another phrase from the same image whose mask exists and is
    non-empty AND does not overlap heavily with this_phrase's mask."""
    others = [p for p in all_phrases_for_image if p != this_phrase]
    rng.shuffle(others)
    this_mask = load_mask(image_name, this_phrase, masks_dir)
    for cand in others:
        m = load_mask(image_name, cand, masks_dir)
        if m is None:
            continue
        if this_mask is None:
            return cand
        # IoU
        inter = (m & this_mask).sum()
        union = (m | this_mask).sum()
        iou = inter / union if union > 0 else 0.0
        if iou < 0.3:
            return cand
    return None


# in-memory cache: image_name -> {phrases: [..], img_np: np.ndarray}
_image_cache: dict[str, dict] = {}


def _get_image_entry(image_name: str, phrases_df: pd.DataFrame,
                     images_dir: Path) -> dict:
    if image_name in _image_cache:
        return _image_cache[image_name]
    img = np.array(Image.open(images_dir / image_name).convert("RGB"))
    phrases = sorted(
        phrases_df.loc[phrases_df["image"] == image_name, "sam_phrase"]
                  .dropna().unique().tolist()
    )
    entry = {"img_np": img, "phrases": phrases}
    _image_cache[image_name] = entry
    return entry


def build_variants(image_name: str, sam_phrase: str,
                   phrases_df: pd.DataFrame | None = None,
                   images_dir: Path = IMAGES_DIR,
                   masks_dir: Path | None = None,
                   ring_px: int = RING_PX,
                   feather_px: int = FEATHER_PX,
                   seed: int | None = None) -> dict[str, Image.Image] | None:
    """Returns dict {variant: PIL Image} or None if the SAM mask is missing.

    The dict always contains 'full' if the image exists; the other four
    variants require the mask. If wrong_object cannot find a suitable
    alternate phrase, that key is replaced with the random_mask variant
    (with a sentinel value of None? no — fill with random_mask).
    """
    phrases_df = pd.read_csv(PHRASES_CSV) if phrases_df is None else phrases_df
    entry = _get_image_entry(image_name, phrases_df, images_dir)
    img_np = entry["img_np"]
    H, W, _ = img_np.shape

    mask = load_mask(image_name, sam_phrase, masks_dir)
    if mask is None:
        return None

    rng = np.random.default_rng(
        RANDOM_SEED_BASE if seed is None else seed
        + hash((image_name, sam_phrase)) % (2**31 - 1)
    )

    variants: dict[str, Image.Image] = {}
    variants["full"] = Image.fromarray(img_np)

    # object_only — hard edge, no dilation
    variants["object_only"] = Image.fromarray(
        apply_keep_mask(img_np, mask, feather_px=0)
    )

    # object_ring — dilation + soft feather
    ring_mask = dilate(mask, ring_px)
    variants["object_ring"] = Image.fromarray(
        apply_keep_mask(img_np, ring_mask, feather_px=feather_px)
    )

    # wrong_object — another query's mask on the same image
    alt_phrase = pick_alternate_phrase(
        image_name, sam_phrase, entry["phrases"], rng, masks_dir)
    if alt_phrase is not None:
        wrong_mask = load_mask(image_name, alt_phrase, masks_dir)
        variants["wrong_object"] = Image.fromarray(
            apply_keep_mask(img_np, wrong_mask, feather_px=feather_px)
        )
        variants["_wrong_object_phrase"] = alt_phrase  # for logging
    else:
        # fall back to a random rectangle
        rm = random_rectangle(H, W, int(mask.sum()), rng)
        variants["wrong_object"] = Image.fromarray(
            apply_keep_mask(img_np, rm, feather_px=feather_px)
        )
        variants["_wrong_object_phrase"] = "<random_fallback>"

    # random_mask — random rectangle with same area as object mask
    rm = random_rectangle(H, W, int(mask.sum()), rng)
    variants["random_mask"] = Image.fromarray(
        apply_keep_mask(img_np, rm, feather_px=feather_px)
    )

    return variants


def main() -> None:
    """Smoke test: dump 5 variants for a single (image, query) to disk."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="image_001.png")
    ap.add_argument("--phrase", default="the remote control")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "sam_phrases" / "variant_examples")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    variants = build_variants(args.image, args.phrase)
    if variants is None:
        print(f"no SAM mask for ({args.image}, '{args.phrase}')")
        return
    for name, img in variants.items():
        if name.startswith("_"):
            print(f"  {name}: {img}")
            continue
        out = args.out_dir / f"{Path(args.image).stem}__{slug(args.phrase)}__{name}.png"
        img.save(out)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
