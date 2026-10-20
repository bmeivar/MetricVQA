"""
Cumulative-accuracy curves: Acc@k as the relative-error tolerance k sweeps
from 0 to 50%. Two panels: in-the-wild and Objectron. Reference baselines
(ChatGPT 5.5 text-only, human) plotted as dashed lines on top.

Output: paper/emnlp/figs/acc_at_k.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from build_tables import (  # noqa: E402
    MODELS, REPO, RES, vision_itw, vision_obj, baseline,
)


def rel_errors(d: Path):
    """Return |y_hat - y| / max(|y|, eps) for valid numeric predictions."""
    p = d / "predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "predicted_value_cm" not in df.columns or "answer" not in df.columns:
        return None
    v = df[df["predicted_value_cm"].notna() & df["answer"].notna()].copy()
    v = v[v["predicted_value_cm"].abs() < 1e4]
    if len(v) == 0:
        return None
    e = (v["predicted_value_cm"] - v["answer"]).abs()
    return (e / v["answer"].abs().clip(lower=1e-6)).values


def llm_itw_rel_errors():
    """ChatGPT 5.5 on in-the-wild: stored as a flat CSV, not a model dir."""
    gt = pd.read_csv(REPO / "questions_v2.csv")
    df = pd.read_csv(REPO / "in-the-wild_chatgpt5.5.csv",
                     header=0, names=["image", "query", "pred"])
    m = df.merge(gt, on=["image", "query"])
    e = (m["pred"] - m["answer"]).abs()
    return (e / m["answer"].abs().clip(lower=1e-6)).values


def curve(rel, thresholds):
    return np.array([(rel <= t).mean() * 100 for t in thresholds])


# ─── data ──────────────────────────────────────────────────────────────────
THRESHOLDS = np.linspace(0, 0.50, 200)

itw_rel = {label: rel_errors(RES / slug) for slug, label in MODELS}
obj_rel = {}
for slug, label in MODELS:
    # mirror build_tables.vision_obj's path fallbacks
    for cand in (RES / f"objectron_{slug}",
                 RES / f"objectron_{slug.replace('-Instruct','')}",
                 RES / f"objectron_{slug.split('-Llama3')[0]}"):
        r = rel_errors(cand)
        if r is not None:
            obj_rel[label] = r
            break

# References: ChatGPT 5.5 (both splits), human (ITW only)
llm_itw = llm_itw_rel_errors()
llm_obj = rel_errors(RES / "objectron_chatgpt5.5_prior")
human_itw = rel_errors(RES / "human_baseline")

# Order VLMs by Acc@10% on ITW (matches Table 1)
order = sorted(
    [(label, vision_itw(slug)["acc10"] if vision_itw(slug) else -1)
     for slug, label in MODELS],
    key=lambda x: -x[1],
)
vlm_labels = [lbl for lbl, _ in order]

# ─── plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

cmap = plt.colormaps["viridis"]
colors = {lbl: cmap(i / max(len(vlm_labels) - 1, 1))
          for i, lbl in enumerate(vlm_labels)}

x = THRESHOLDS * 100

for ax, title, rel_map, llm_rel, human_rel in [
    (axes[0], "In-the-wild (n=331)",   itw_rel, llm_itw, human_itw),
    (axes[1], "Objectron (n=10,482)",  obj_rel, llm_obj, None),
]:
    # VLMs
    for lbl in vlm_labels:
        rel = rel_map.get(lbl)
        if rel is None:
            continue
        ax.plot(x, curve(rel, THRESHOLDS), linewidth=1.6,
                color=colors[lbl], label=lbl)

    # Reference baselines
    if llm_rel is not None:
        ax.plot(x, curve(llm_rel, THRESHOLDS), linewidth=2.2,
                color="#444", linestyle="--", label="ChatGPT 5.5 (no image)",
                zorder=10)
    if human_rel is not None:
        ax.plot(x, curve(human_rel, THRESHOLDS), linewidth=2.2,
                color="#b53737", linestyle="--", label="Human reference",
                zorder=10)

    # Acc@10% guide
    ax.axvline(10, color="#888", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Relative-error tolerance $k$ (%)")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    # panel title removed: LaTeX caption distinguishes the two panels

axes[0].set_ylabel("Acc@$k$ (cumulative %)")

# Single legend below
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, -0.02))

# suptitle removed: LaTeX caption is the figure's only title
fig.tight_layout(rect=(0, 0.10, 1, 1.0))

OUT = HERE.parent / "emnlp_paper" / "figs" / "acc_at_k.pdf"
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
