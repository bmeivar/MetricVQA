#!/usr/bin/env python3
"""Prediction-level sensitivity, independent of Acc@10%.

Every reviewer made the same point: our channel tests operate on the binary
correct@10% indicator, so a null says "no reliable effect on that threshold",
not "the model does not use the cue". A prediction can move a lot and stay
inside the band, or move and cross it in both directions and cancel.

This measures the predictions themselves. For each (model, contrast) we pair
rows and compute the log ratio r = log(y_cf / y_base):

  med|log r|   median absolute displacement (0 = predictions identical)
  %moved       fraction of rows whose prediction changes by more than 10%
  dir          signed median log ratio -- systematic push up (+) or down (-)

Reading: identity/pixels should show large displacement (the model is
reacting), lens distortion should show near-zero if geometry genuinely goes
unused. A large displacement with a near-zero direction is the apparent-size
signature: predictions move, but not predictably.

Writes results/prediction_displacement.csv and a LaTeX table.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("METRICVQA_ROOT", "."))
sys.path.insert(0, str(REPO / "paper/emnlp_scripts"))
import fdr_analysis as base  # loaders, MODELS, CTX_MAP, CONDITIONS

MOVE_THRESH = np.log(1.10)   # "materially changed" = >10% shift in the number
B_BOOT = 2000

CONTRASTS = [
    ("identity",   "wrong_object", "object_only"),
    ("pixels",     "object_only",  "random_mask"),
    ("ring",       "object_ring",  "object_only"),
    ("lens barrel", "dist_k015",   "full"),
    ("lens pin",   "dist_pin015",  "full"),
]


def load_preds_values(path: Path) -> pd.DataFrame | None:
    """Like base.load_preds but keeps the predicted VALUE, not correctness."""
    p = path / "predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    col = "predicted_value_cm" if "predicted_value_cm" in df.columns else "prediction"
    out = pd.DataFrame({
        "key": df["image"].map(base.base_key) + "||" + df["query"].astype(str).str.strip(),
        "pred": pd.to_numeric(df[col], errors="coerce"),
    })
    # positive, finite predictions only -- log ratio is undefined otherwise
    out = out[(out["pred"] > 0) & np.isfinite(out["pred"])]
    return out.drop_duplicates("key")


def scene_of(key: str) -> str:
    return key.split("||")[0]


def cluster_ci(vals: np.ndarray, scenes: np.ndarray, stat, rng) -> tuple[float, float]:
    uniq = np.unique(scenes)
    idx = {s: np.where(scenes == s)[0] for s in uniq}
    reps = []
    for _ in range(B_BOOT):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([idx[s] for s in pick])
        reps.append(stat(vals[sel]))
    return tuple(np.percentile(reps, [2.5, 97.5]))


def main():
    rng = np.random.default_rng(7)
    rows = []
    for slug, label in base.MODELS:
        cache = {}
        for cond, pf in base.CONDITIONS.items():
            d = load_preds_values(pf(slug))
            if d is not None:
                cache[cond] = d
        for cname, a, b in CONTRASTS:
            if a not in cache or b not in cache:
                continue
            j = cache[a].merge(cache[b], on="key", suffixes=("_cf", "_base"))
            if len(j) < 50:
                continue
            lr = np.log(j["pred_cf"].values / j["pred_base"].values)
            scenes = np.array([scene_of(k) for k in j["key"]])
            med_abs = float(np.median(np.abs(lr)))
            moved = float((np.abs(lr) > MOVE_THRESH).mean())
            direction = float(np.median(lr))
            lo, hi = cluster_ci(np.abs(lr), scenes, np.median, rng)
            rows.append({
                "model": label, "contrast": cname, "n": len(j),
                "med_abs_logratio": round(med_abs, 4),
                "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                "pct_moved": round(100 * moved, 1),
                "median_logratio": round(direction, 4),
                "median_ratio": round(float(np.exp(direction)), 3),
            })

    df = pd.DataFrame(rows)
    out = REPO / "results" / "prediction_displacement.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} model x contrast cells)\n")

    piv = df.pivot_table(index="contrast", values=["med_abs_logratio", "pct_moved"],
                         aggfunc="median").reindex([c[0] for c in CONTRASTS])
    print("pooled medians across models:")
    print("contrast        med|log r|   %moved(>10%)")
    for c in [c[0] for c in CONTRASTS]:
        s = df[df.contrast == c]
        if len(s):
            print(f"  {c:13s} {s.med_abs_logratio.median():8.3f} "
                  f"{s.pct_moved.median():10.1f}%")
    return df


if __name__ == "__main__":
    main()
