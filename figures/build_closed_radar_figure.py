"""Paper-Figure-2-style radar panels for the frontier closed models.

Adapted from paper/rebuttal_arr_jul2026/scripts/closed_model_radar_paper.py
with the camera-ready conventions fixed:
  - Acc@10% over parsable responses (unparsable excluded), matching
    frontier_deltas.py / the paper tables.
  - Values are read from results/frontier_deltas.csv (single source of
    truth, written by paper/emnlp_scripts/frontier_deltas.py).

Panels: GPT-5.5 and Gemini 3.1 Pro, both no-reasoning (the protocol-matched
configs; the Gemini re-run of 2026-08-26 covers all manifests), and
Qwen3-VL-235B (best open model) recomputed identically as reference.

Axis layout, colors, and reference circles match
paper/emnlp_scripts/build_radar_figure.py (the paper's Figure 2).

Output: paper/emnlp_paper/figs/closed_radar.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
REPO = HERE.parent.parent
CSV = REPO / "results" / "frontier_deltas.csv"
OUT = HERE.parent / "emnlp_paper" / "figs" / "closed_radar.pdf"


def load_accs(bundle: str) -> dict[str, float | None]:
    df = pd.read_csv(CSV)
    df = df[(df["bundle"] == bundle) & (df["kind"] == "acc")]
    acc = dict(zip(df["name"], df["value"] / 100.0))
    dist = [acc.get(k) for k in ("distorted_k015", "distorted_pin015")]
    dist = [d for d in dist if d is not None]
    return dict(
        full=acc.get("vision"),
        no_image=acc.get("text_only"),
        object_only=acc.get("object_only"),
        object_ring=acc.get("object_ring"),
        wrong_object=acc.get("wrong_object"),
        zoom_in=acc.get("zoom_in"),
        zoom_out=acc.get("zoom_out"),
        lens=sum(dist) / len(dist) if dist else None,
    )


PANELS = [
    ("GPT-5.5 (no reasoning)",       load_accs("gpt-5.5-no-reasoning_bundle")),
    ("Gemini 3.1 Pro (no reasoning)", load_accs("gemini-3.1-pro-no-reasoning_bundle")),
    ("Qwen3-VL-235B (best open)",    load_accs("qwen3-vl-235b_open")),
]

for label, d in PANELS:
    print(label, {k: (None if v is None else round(v * 100, 1))
                  for k, v in d.items()})

# paper Figure 2 axes: order and colors from build_radar_figure.py
AXES = [
    ("no\nimage",              "#5a2563", "no_image"),
    ("object\nonly",           "#3a7a3a", "object_only"),
    ("object\nring",           "#5fa05f", "object_ring"),
    ("wrong\nobject",          "#b53737", "wrong_object"),
    ("zoom\nin 1.4$\\times$",  "#3a6aa8", "zoom_in"),
    ("zoom\nout 0.6$\\times$", "#5a8acc", "zoom_out"),
    ("lens\ndist.",            "#a87a3a", "lens"),
]
N = len(AXES)
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
theta_closed = np.concatenate([theta, theta[:1]])

ymax = 0.62
yticks = [0.10, 0.20, 0.30, 0.40, 0.50]

fig, axes_grid = plt.subplots(1, 3, figsize=(9.6, 3.8),
                              subplot_kw=dict(projection="polar"))

for ax, (label, d) in zip(axes_grid, PANELS):
    vals = np.array([np.nan if d[k] is None else d[k] for _, _, k in AXES])
    v_closed = np.concatenate([vals, vals[:1]])
    ax.plot(theta_closed, v_closed, color="#1a1a1a", linewidth=1.5)
    ax.fill(theta_closed, v_closed, color="#1a1a1a", alpha=0.18)

    if d["full"] is not None:
        ax.plot(theta_closed, np.full(N + 1, d["full"]), color="#e67e22",
                linewidth=1.1, linestyle="--", alpha=0.85)
    if d["object_only"] is not None:
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(t, np.full_like(t, d["object_only"]), color="#2c7a3e",
                linewidth=1.2, linestyle="--", alpha=0.9, zorder=3)

    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.plot(theta[i], v, "o", color=AXES[i][1], markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5)

    ax.set_xticks(theta)
    ax.set_xticklabels([n for n, _, _ in AXES], fontsize=7.5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(y*100)}%" for y in yticks], fontsize=6.5,
                       color="#888")
    ax.set_ylim(0, ymax)
    ax.tick_params(axis="x", pad=2)
    ax.set_rlabel_position(95)
    ax.grid(color="#ddd", linewidth=0.6)
    ax.spines["polar"].set_color("#bbb")
    full_s = "" if d["full"] is None else f"  (full={d['full']*100:.1f}%)"
    ax.set_title(label + full_s, fontsize=9.5, pad=12)

legend_handles = [
    Line2D([0], [0], color="#1a1a1a", linewidth=1.5,
           label="Acc@10% under intervention"),
    Line2D([0], [0], color="#e67e22", linewidth=1.1, linestyle="--",
           label="full-image Acc@10% (reference for blank / ring / lens)"),
    Line2D([0], [0], color="#2c7a3e", linewidth=1.2, linestyle="--",
           label="object_only Acc@10% (reference for wrong_object / zoom)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 0.98))

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
