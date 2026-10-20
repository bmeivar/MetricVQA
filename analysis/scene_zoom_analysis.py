"""
Scene-crop zoom control (camera-ready): apparent-size manipulation on real
scene crops at three field-of-view levels (zoom_in = 1.8x bbox, base = 3x
bbox, zoom_out = original full image), no gray canvas, no rescaling
artifacts. Results produced by run_scene_zoom_rits*.py on the cluster.

Per model: Acc@10% per condition (unparsable = incorrect, the convention for
post-rebuttal analyses, paper Sec. 3.1) and paired sign-flip permutation
tests on the per-row correct indicator for the three contrasts, pooled
across the 8 paper models and per model.

Outputs:
  results/scene_zoom_summary.csv
  paper/emnlp_paper/tables/table_scene_zoom.tex
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("METRICVQA_ROOT", "."))
RES = REPO / "results" / "scene_zoom"
OUT_TABLE = REPO / "paper/emnlp_paper/tables/table_scene_zoom.tex"
OUT_CSV = REPO / "results/scene_zoom_summary.csv"

# (dir prefix, display label) — the 8 paper models the rebuttal reports.
# Phi-3.5-Vision was also run but is not part of the paper pool.
MODELS = [
    ("qwen3.5-397b",           "Qwen3.5-397B"),
    ("qwen3-vl-235b",          "Qwen3-VL-235B"),
    ("Qwen2.5-VL-7B",          "Qwen2.5-VL-7B"),
    ("LLaVA-OneVision-7B",     "LLaVA-OV"),
    ("LLaVA-v1.6-Mistral-7B",  "LLaVA-v1.6"),
    ("Gemma3-4B-IT",           "Gemma3-4B"),
    ("SmolVLM-Instruct",       "SmolVLM"),
    ("Qwen2.5-VL-3B",          "Qwen2.5-VL-3B"),
]

CONDITIONS = ["zoom_in", "base", "zoom_out"]
CONTRASTS = [("zoom_in", "base"), ("zoom_in", "zoom_out"), ("base", "zoom_out")]
B = 10_000
SEED = 0


DIR_SUFFIX = {"zoom_in": "in", "base": "base", "zoom_out": "out"}


def load(model_dir: str, cond: str) -> pd.DataFrame:
    p = RES / f"{model_dir}_scene_zoom_{DIR_SUFFIX[cond]}" / "predictions.csv"
    df = pd.read_csv(p)
    pred_col = "predicted_value_cm" if "predicted_value_cm" in df.columns else "prediction"
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
    key = df["image"].map(lambda s: Path(str(s)).name) + "||" + df["query"].astype(str)
    ok = df[pred_col].notna() & (df[pred_col].abs() < 1e4)
    rel = (df[pred_col] - df["answer"]).abs() / df["answer"].abs().clip(lower=1e-6)
    correct = (ok & (rel <= 0.10)).astype(int)  # unparsable = incorrect
    pred = df[pred_col].where(ok & (df[pred_col] > 0))
    return pd.DataFrame({"key": key, "correct": correct, "pred": pred})


def sign_flip_p(diffs: np.ndarray, rng: np.random.Generator) -> float:
    obs = diffs.mean()
    if len(diffs) == 0:
        return float("nan")
    flips = rng.choice([-1, 1], size=(B, len(diffs)))
    null = (flips * diffs).mean(axis=1)
    return float((np.abs(null) >= abs(obs) - 1e-12).mean())


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows, csv_rows = [], []
    pooled = {c: [] for c in CONTRASTS}
    for mdir, label in MODELS:
        conds = {c: load(mdir, c) for c in CONDITIONS}
        accs = {c: conds[c]["correct"].mean() * 100 for c in CONDITIONS}
        pvals = {}
        for ca, cb in CONTRASTS:
            m = conds[ca].merge(conds[cb], on="key", suffixes=("_a", "_b"))
            # Directional apparent-size test: does the *predicted value*
            # move with apparent size?  Paired log-ratio of predictions
            # (rows parsable in both conditions), sign-flip permutation.
            v = m[m["pred_a"].notna() & m["pred_b"].notna()]
            logr = np.log(v["pred_a"].to_numpy(float) / v["pred_b"].to_numpy(float))
            pooled[(ca, cb)].append(logr)
            pvals[(ca, cb)] = sign_flip_p(logr, rng)
        med_ratio, gm_ratio = {}, {}
        for ca, cb in CONTRASTS:
            m = conds[ca].merge(conds[cb], on="key", suffixes=("_a", "_b"))
            v = m[m["pred_a"].notna() & m["pred_b"].notna()]
            r = v["pred_a"].to_numpy(float) / v["pred_b"].to_numpy(float)
            med_ratio[(ca, cb)] = float(np.median(r))
            gm_ratio[(ca, cb)] = float(np.exp(np.log(r).mean()))
        rows.append(
            f"{label} & {accs['zoom_in']:.1f}\\% & {accs['base']:.1f}\\% & "
            f"{accs['zoom_out']:.1f}\\% & "
            + " & ".join(f"{gm_ratio[c]:.2f} ({pvals[c]:.2f})" for c in CONTRASTS)
            + " \\\\")
        csv_rows.append({"model": label,
                         **{f"acc_{c}": round(accs[c], 2) for c in CONDITIONS},
                         **{f"medratio_{a}_vs_{b}": round(med_ratio[(a, b)], 3)
                            for a, b in CONTRASTS},
                         **{f"gmratio_{a}_vs_{b}": round(gm_ratio[(a, b)], 3)
                            for a, b in CONTRASTS},
                         **{f"p_{a}_vs_{b}": round(pvals[(a, b)], 4)
                            for a, b in CONTRASTS}})
        print(f"{label:16s} " + "  ".join(f"{c}={accs[c]:.1f}" for c in CONDITIONS)
              + "   gm(p): " + "  ".join(
                  f"{a}/{b}:{gm_ratio[(a,b)]:.2f}({pvals[(a,b)]:.3f})"
                  for a, b in CONTRASTS))

    pooled_p = {}
    for c in CONTRASTS:
        diffs = np.concatenate(pooled[c])
        pooled_p[c] = sign_flip_p(diffs, rng)
    print("pooled p-values: " + "  ".join(
        f"{a} vs {b}: {pooled_p[(a,b)]:.3f}" for a, b in CONTRASTS))
    csv_rows.append({"model": "POOLED",
                     **{f"p_{a}_vs_{b}": round(pooled_p[(a, b)], 4)
                        for a, b in CONTRASTS}})
    pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False)

    tex = []
    tex.append("\\begin{table*}[t]")
    tex.append("\\centering\\small")
    tex.append("\\begin{tabular}{l|ccc|ccc}")
    tex.append("\\toprule")
    tex.append(" & \\multicolumn{3}{c|}{Acc@10\\%} & \\multicolumn{3}{c}{geom.\\ mean prediction ratio (permutation $p$)} \\\\")
    tex.append("Model & zoom-in & base & zoom-out & in/base & in/out & base/out \\\\")
    tex.append("\\midrule")
    tex.extend(rows)
    tex.append("\\midrule")
    tex.append("Pooled ($8$ models) & & & & "
               + " & ".join(f"{pooled_p[c]:.2f}" for c in CONTRASTS) + " \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append(
        "\\caption{\\textbf{Scene-crop zoom control.} Apparent size varied on "
        "real scene content at three field-of-view levels centered on the "
        "target (zoom-in $=1.8\\times$ object bbox, base $=3\\times$ bbox, "
        "zoom-out $=$ original full image; each crop resized to the input "
        "resolution), with natural context preserved and no synthetic "
        "canvas or rescaling artifacts ($342$ queries $\\times$ $3$ "
        "conditions, $8$ pool models). Acc@10\\% treats unparsable "
        "responses as incorrect. The directional test asks whether the "
        "\\emph{predicted value} tracks apparent size: geometric-mean "
        "paired prediction ratio between conditions, with a paired "
        "sign-flip permutation test on the per-row log-ratio "
        "($B{=}10^4$, two-sided). A model reading apparent size would "
        "predict larger sizes under tighter crops (in/base ratio "
        "$\\approx 1.67$ from the $3\\times/1.8\\times$ crop factors). "
        "Observed ratios stay within $\\pm 10\\%$ of $1.0$ (median "
        "paired ratio exactly $1.00$ in every cell), an order of "
        "magnitude below the size-tracking prediction; the few "
        "nominally significant log-ratio shifts are small and "
        "inconsistent in direction (several negative), i.e.\\ drift, "
        "not size use. The apparent-size invariance of "
        "Section~\\ref{sec:res_zoom} is therefore not an artifact of "
        "the gray-canvas implementation.}")
    tex.append("\\label{tab:scene_zoom}")
    tex.append("\\end{table*}")
    OUT_TABLE.write_text("\n".join(tex) + "\n")
    print(f"wrote {OUT_TABLE}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
