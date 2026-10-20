"""
Per-VLM intervention decomposition figure.

3x4 grid of small multiples, one panel per VLM. Each panel: bar chart of
Δ MAE (cm) = MAE(intervention) − MAE(full-image baseline), in-the-wild split.

Positive bars = intervention made predictions worse (model was using that
evidence). Bars near zero = model wasn't using that evidence.

Output: paper/emnlp/figs/intervention_decomposition.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import sys
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from build_tables import (  # noqa: E402
    MODELS, vision_itw, gray_itw, zoom_in, zoom_out,
    context_variant, distorted,
)

# ─── intervention spec ──────────────────────────────────────────────────────
# (label, color, loader)
def zoom_avg(slug):
    a, b = zoom_in(slug), zoom_out(slug)
    if a is None or b is None:
        return None
    return {"mae": 0.5 * (a["mae"] + b["mae"])}

INTERVENTIONS = [
    ("blank",        "#7a3b8a", lambda s: gray_itw(s)),
    ("object\nonly", "#3a7a3a", lambda s: context_variant(s, "object_only")),
    ("object\nring", "#5fa05f", lambda s: context_variant(s, "object_ring")),
    ("wrong\nobject","#b53737", lambda s: context_variant(s, "wrong_object")),
    ("zoom\n(avg)",  "#3a6aa8", lambda s: zoom_avg(s)),
    ("lens\ndist.",  "#a87a3a", lambda s: distorted(s)),
]

# ─── collect data ──────────────────────────────────────────────────────────
records = []  # list of (model_label, base_mae, [delta or NaN per intervention])
for slug, label in MODELS:
    base = vision_itw(slug)
    base_mae = base["mae"] if base else np.nan
    row = [base_mae]
    for _, _, fn in INTERVENTIONS:
        m = fn(slug)
        if m is None or base is None:
            row.append(np.nan)
        else:
            row.append(m["mae"] - base_mae)
    records.append((label, base_mae, row[1:]))

# ─── plot ──────────────────────────────────────────────────────────────────
n_cols, n_rows = 4, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.5, 7.0), sharey=True)
axes = axes.flatten()

# Determine symmetric y limits with a small headroom.
all_deltas = np.array(
    [d for _, _, ds in records for d in ds if not np.isnan(d)]
)
if len(all_deltas) == 0:
    ymax = 1.0
else:
    ymax = float(np.nanmax(np.abs(all_deltas))) * 1.15
ymax = max(ymax, 5.0)  # avoid empty-looking axes

xpos = np.arange(len(INTERVENTIONS))
xlabels = [name for name, _, _ in INTERVENTIONS]
colors = [c for _, c, _ in INTERVENTIONS]

for ax, (label, base_mae, deltas) in zip(axes, records):
    deltas_arr = np.array(deltas, dtype=float)
    finite = ~np.isnan(deltas_arr)
    ax.bar(xpos[finite], deltas_arr[finite],
           color=[colors[i] for i in np.where(finite)[0]],
           edgecolor="#222", linewidth=0.4, width=0.78)
    # missing bars: hatched gray placeholder so layout doesn't shift
    for i, ok in enumerate(finite):
        if not ok:
            ax.bar(xpos[i], 0.01 * ymax, color="#eee", hatch="//",
                   edgecolor="#bbb", linewidth=0.4, width=0.78)
    ax.axhline(0, color="#333", linewidth=0.7)
    ax.set_ylim(-ymax * 0.35, ymax)
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels, fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7.5)
    title = f"{label}"
    if not np.isnan(base_mae):
        title += f"  (MAE={base_mae:.1f} cm)"
    ax.set_title(title, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

# Hide any unused panels
for ax in axes[len(records):]:
    ax.set_visible(False)

# Shared y-axis label on the leftmost panels
for r in range(n_rows):
    axes[r * n_cols].set_ylabel(r"$\Delta$ MAE (cm) vs full image", fontsize=8)

# Footer legend explaining categories
legend_handles = [
    mpatches.Patch(color="#7a3b8a", label="language-only prior"),
    mpatches.Patch(color="#3a7a3a", label="target only / target+ring"),
    mpatches.Patch(color="#b53737", label="identity swap"),
    mpatches.Patch(color="#3a6aa8", label="apparent size"),
    mpatches.Patch(color="#a87a3a", label="scene geometry"),
    mpatches.Patch(facecolor="#eee", hatch="//", edgecolor="#bbb",
                   label="missing / parse-failed"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=6,
           frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))

fig.suptitle(
    "Per-VLM intervention decomposition — positive bars indicate the "
    "intervention increased MAE (i.e., the model was using that evidence)",
    fontsize=9.5, y=0.995,
)

fig.tight_layout(rect=(0, 0.03, 1, 0.97))

OUT = HERE / "figs" / "intervention_decomposition.pdf"
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
