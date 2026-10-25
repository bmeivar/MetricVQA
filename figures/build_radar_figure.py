"""
Per-VLM radar (spider) chart of intervention Acc@10%.

3x4 grid of small multiples. Each panel: 6-axis radar with
Acc@10% values under each intervention (blank, object_only, object_ring,
wrong_object, zoom_avg, lens distortion). A dashed hexagon overlay marks
the model's full-image Acc@10% as a baseline reference.

Output: paper/emnlp/figs/intervention_radar.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "analysis"))  # build_tables
sys.path.insert(0, str(HERE.parent))  # release_data
from release_data import FIGS, load_itw  # noqa: E402
FIGS.mkdir(parents=True, exist_ok=True)
from build_tables import (  # noqa: E402
    MODELS, vision_itw, gray_itw, zoom_in, zoom_out,
    context_variant, distorted, metrics_from, RES,
)


def text_only_itw(slug):
    """Image omitted entirely from the prompt (vs blank which sends gray RGB)."""
    return metrics_from(RES / f"text_only_{slug}")


def _no_image_acc(slug):
    """Unified 'no image' probe. Prefer text-only where the model produces
    numeric answers without an image; fall back to gray-blank otherwise."""
    m = text_only_itw(slug)
    if m is not None:
        return m["acc10"]
    m = gray_itw(slug)
    return m["acc10"] if m is not None else None


def _dist_avg_acc(slug):
    """Average Acc@10% across the two |k|=0.15 distortion conditions
    (barrel and pincushion). The stronger k=0.25 conditions are not used in
    the paper (overly extreme)."""
    vals = []
    for tag in ("distorted", "distorted_pin015"):
        m = metrics_from(RES / f"{tag}_{slug}")
        if m is not None:
            vals.append(m["acc10"])
    if not vals:
        return None
    return sum(vals) / len(vals)


# axis_name, color, loader returning Acc@10 (0..1) or None
INTERVENTIONS = [
    ("no\nimage",     "#5a2563", _no_image_acc),
    ("object\nonly",  "#3a7a3a", lambda s: context_variant(s, "object_only")["acc10"] if context_variant(s, "object_only") else None),
    ("object\nring",  "#5fa05f", lambda s: context_variant(s, "object_ring")["acc10"] if context_variant(s, "object_ring") else None),
    ("wrong\nobject", "#b53737", lambda s: context_variant(s, "wrong_object")["acc10"] if context_variant(s, "wrong_object") else None),
    ("zoom\nin 1.4$\\times$",  "#3a6aa8", lambda s: zoom_in(s)["acc10"]  if zoom_in(s)  else None),
    ("zoom\nout 0.6$\\times$", "#5a8acc", lambda s: zoom_out(s)["acc10"] if zoom_out(s) else None),
    ("lens\ndist.",   "#a87a3a", lambda s: _dist_avg_acc(s)),
]

N = len(INTERVENTIONS)
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
theta_closed = np.concatenate([theta, theta[:1]])

# ─── collect data ──────────────────────────────────────────────────────────
all_records = []
all_vals = []
for slug, label in MODELS:
    base = vision_itw(slug)
    base_acc = base["acc10"] if base else np.nan
    vals = []
    for _, _, fn in INTERVENTIONS:
        v = fn(slug)
        vals.append(np.nan if v is None else v)
    all_records.append((slug, label, base_acc, vals))
    all_vals.extend([v for v in vals if not np.isnan(v)])
    if not np.isnan(base_acc):
        all_vals.append(base_acc)

# Body figure: drop the 3 LoRA-shown models (avoid overlap with Figure~3)
# plus SmolVLM, so the body radar is smaller and the LoRA figure can fit on
# the same printed spread. Full 12-model radar stays in the appendix.
DROP_SLUGS = {
    "Qwen2.5-VL-3B-Instruct",
    "Qwen2.5-VL-7B-Instruct",
    "LLaVA-v1.6-Mistral-7B",
    "SmolVLM-Instruct",
}
SUBSET_SLUGS = {slug for slug, _ in MODELS if slug not in DROP_SLUGS}

ymax = max(all_vals) * 1.10 if all_vals else 0.3
yticks = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
yticks = [y for y in yticks if y <= ymax]


def render_panels(records, n_cols, n_rows, figsize, output_path):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             subplot_kw=dict(projection="polar"))
    axes = np.array(axes).flatten() if hasattr(axes, "flatten") else np.array([axes])

    axis_labels = [name for name, _, _ in INTERVENTIONS]

    for ax, (label, base_acc, vals) in zip(axes, records):
        vals_arr = np.array(vals, dtype=float)
        finite = ~np.isnan(vals_arr)
        if finite.all():
            v_closed = np.concatenate([vals_arr, vals_arr[:1]])
            ax.plot(theta_closed, v_closed, color="#1a1a1a", linewidth=1.5)
            ax.fill(theta_closed, v_closed, color="#1a1a1a", alpha=0.18)
        else:
            v_plot = vals_arr.copy()
            v_plot[~finite] = 0
            v_closed = np.concatenate([v_plot, v_plot[:1]])
            ax.plot(theta_closed, v_closed, color="#1a1a1a",
                    linewidth=1.5, alpha=0.5, linestyle="--")
            ax.fill(theta_closed, v_closed, color="#1a1a1a", alpha=0.10)

        if not np.isnan(base_acc):
            b_closed = np.full(N + 1, base_acc)
            ax.plot(theta_closed, b_closed, color="#e67e22",
                    linewidth=1.1, linestyle="--", alpha=0.85)

        obj_only_idx = next(
            i for i, (name, _, _) in enumerate(INTERVENTIONS) if "object\nonly" in name
        )
        obj_only_acc = vals_arr[obj_only_idx]
        if not np.isnan(obj_only_acc):
            theta_smooth = np.linspace(0, 2 * np.pi, 200)
            ax.plot(theta_smooth, np.full_like(theta_smooth, obj_only_acc),
                    color="#2c7a3e", linewidth=1.2, linestyle="--", alpha=0.9,
                    zorder=3)

        for i, v in enumerate(vals_arr):
            if not np.isnan(v):
                color = INTERVENTIONS[i][1]
                ax.plot(theta[i], v, "o", color=color, markersize=4.5,
                        markeredgecolor="white", markeredgewidth=0.5, zorder=5)

        ax.set_xticks(theta)
        ax.set_xticklabels(axis_labels, fontsize=7.5)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{int(y*100)}%" for y in yticks], fontsize=6.5,
                           color="#888")
        ax.set_ylim(0, ymax)
        ax.tick_params(axis="x", pad=2)
        ax.set_rlabel_position(95)
        ax.grid(color="#ddd", linewidth=0.6)
        ax.spines["polar"].set_color("#bbb")

        title = f"{label}"
        if not np.isnan(base_acc):
            title += f"  (full={base_acc*100:.1f}%)"
        ax.set_title(title, fontsize=9, pad=10)

    for ax in axes[len(records):]:
        ax.set_visible(False)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#1a1a1a", linewidth=1.5, label="Acc@10% under intervention"),
        Line2D([0], [0], color="#e67e22", linewidth=1.1, linestyle="--",
               label="full-image Acc@10% (reference for blank / ring / lens)"),
        Line2D([0], [0], color="#2c7a3e", linewidth=1.2, linestyle="--",
               label="object_only Acc@10% (reference for wrong_object / zoom)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))

    fig.tight_layout(rect=(0, 0.025, 1, 0.99))
    fig.savefig(output_path, bbox_inches="tight")
    print(f"wrote {output_path}")
    plt.close(fig)


# Sort by full-image Acc@10% descending. NaN baselines last.
all_records.sort(key=lambda r: (-r[2] if not np.isnan(r[2]) else 1e9))

# Body: all 12 models in a 4x3 grid (test whether the body still fits 8pp)
subset_records = [(label, base_acc, vals) for slug, label, base_acc, vals in all_records]
render_panels(subset_records, n_cols=4, n_rows=3, figsize=(11.5, 8.4),
              output_path=FIGS / "intervention_radar.pdf")

# Appendix: all 12, 4x3 grid
full_records = [(label, base_acc, vals) for slug, label, base_acc, vals in all_records]
render_panels(full_records, n_cols=4, n_rows=3, figsize=(11.5, 8.4),
              output_path=FIGS / "intervention_radar_full.pdf")
