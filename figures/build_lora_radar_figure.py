"""
Overlaid base-vs-LoRA radar plots for the two Qwen2.5-VL models.

Two panels (left: Qwen2.5-VL-3B, right: Qwen2.5-VL-7B). Each panel shows the
same 7-axis radar as build_radar_figure.py, with two overlaid polygons:
  - base model (red)
  - LoRA-finetuned model (blue)

Axes:
  no image (text-only, gray fallback)
  object_only
  object_ring
  wrong_object
  zoom-in 1.4x
  zoom-out 0.6x
  lens dist. (avg of barrel + pincushion, |k|=0.15)

Output: paper/emnlp/figs/lora_radar.pdf
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from build_tables import (  # noqa: E402
    REPO, RES, vision_itw, gray_itw, zoom_in, zoom_out,
    context_variant, metrics_from,
)


# ─── base-model loaders (existing) ─────────────────────────────────────────
def base_text_only(slug):
    return metrics_from(RES / f"text_only_{slug}")


def base_no_image_acc(slug):
    m = base_text_only(slug)
    if m is not None:
        return m["acc10"]
    m = gray_itw(slug)
    return m["acc10"] if m is not None else None


def base_ctx_acc(slug, variant):
    m = context_variant(slug, variant)
    return m["acc10"] if m is not None else None


def _median_paired_ratio_from_dirs(oo_pred_path, zoom_pred_path):
    """Compute median(predicted_zoom / predicted_object_only) on the paired
    (query, answer) intersection. Returns None if either path missing."""
    if not Path(oo_pred_path).exists() or not Path(zoom_pred_path).exists():
        return None
    oo = pd.read_csv(oo_pred_path)
    zo = pd.read_csv(zoom_pred_path)
    for df in (oo, zo):
        df["answer"] = pd.to_numeric(df["answer"], errors="coerce")
        df["predicted_value_cm"] = pd.to_numeric(df["predicted_value_cm"], errors="coerce")
    oo = oo.dropna(subset=["predicted_value_cm", "answer"])
    zo = zo.dropna(subset=["predicted_value_cm", "answer"])
    oo = oo[oo["predicted_value_cm"].abs() < 1e4]
    zo = zo[zo["predicted_value_cm"].abs() < 1e4]
    oo["_k"] = oo["query"].astype(str).str.lower().str.strip() + "||" + oo["answer"].astype(str)
    zo["_k"] = zo["query"].astype(str).str.lower().str.strip() + "||" + zo["answer"].astype(str)
    m = oo.merge(zo, on="_k", suffixes=("_oo", "_z"))
    m = m[m["predicted_value_cm_oo"] > 0]
    if len(m) < 10:
        return None
    return float((m["predicted_value_cm_z"] / m["predicted_value_cm_oo"]).median())


def _invariance_score(r_in, r_out):
    """1.0 = perfectly invariant; deviates as either zoom-direction's median
    paired ratio drifts from 1.0. Worst-of-the-two-directions."""
    if r_in is None and r_out is None:
        return None
    devs = [abs(1.0 - r) for r in (r_in, r_out) if r is not None]
    return max(0.0, 1.0 - max(devs))


def base_apparent_invariance(slug):
    """Median paired ratio (zoom / object_only) for the base, both directions,
    combined into one invariance score. The base zoom variants live at
    results/zoom_{in,out}_{slug}; object_only lives at results/context_object_only/{ctx_short}."""
    ctx_short = CONTEXT_NAME_MAP.get(slug, slug) if "CONTEXT_NAME_MAP" in globals() else slug
    # Resolve context_only path via the same name_map as build_tables
    name_map = {
        "Qwen2.5-VL-3B-Instruct":    "Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL-7B-Instruct":    "Qwen2.5-VL-7B-Instruct",
        "PaliGemma-3B-Mix-448":      "paligemma-3b-mix-448",
        "SmolVLM-Instruct":          "SmolVLM-Instruct",
        "Gemma3-4B-IT":              "gemma-3-4b-it",
        "granite-vision-3.3-2b":     "granite-vision-3.3-2b",
        "LLaVA-v1.6-Mistral-7B":     "llava-v1.6-mistral-7b-hf",
        "LLaVA-OneVision-7B":        "llava-onevision-qwen2-7b-ov-hf",
    }
    ctx_short = name_map.get(slug, slug)
    oo_path = RES / "context_object_only" / ctx_short / "predictions.csv"
    r_in = _median_paired_ratio_from_dirs(oo_path, RES / f"zoom_in_{slug}" / "predictions.csv")
    r_out = _median_paired_ratio_from_dirs(oo_path, RES / f"zoom_out_{slug}" / "predictions.csv")
    return _invariance_score(r_in, r_out)


def lora_apparent_invariance(lora_dir):
    """For a LoRA, the context_5variants CSV holds object_only; the zoom dirs
    hold zoom predictions. Returns the same invariance score."""
    ctx_p = RES / lora_dir / "in_the_wild_context_5variants" / "predictions.csv"
    if not ctx_p.exists():
        return None
    df_all = pd.read_csv(ctx_p)
    df_all["answer"] = pd.to_numeric(df_all["answer"], errors="coerce")
    df_all["predicted_value_cm"] = pd.to_numeric(df_all["predicted_value_cm"], errors="coerce")
    df_all = df_all.dropna(subset=["predicted_value_cm", "answer"])
    df_all = df_all[df_all["predicted_value_cm"].abs() < 1e4]
    oo = df_all[df_all["variant"] == "object_only"].copy()
    oo["_k"] = oo["query"].astype(str).str.lower().str.strip() + "||" + oo["answer"].astype(str)

    def median_ratio(zoom_pred_path):
        if not Path(zoom_pred_path).exists():
            return None
        zo = pd.read_csv(zoom_pred_path)
        zo["answer"] = pd.to_numeric(zo["answer"], errors="coerce")
        zo["predicted_value_cm"] = pd.to_numeric(zo["predicted_value_cm"], errors="coerce")
        zo = zo.dropna(subset=["predicted_value_cm", "answer"])
        zo = zo[zo["predicted_value_cm"].abs() < 1e4]
        zo["_k"] = zo["query"].astype(str).str.lower().str.strip() + "||" + zo["answer"].astype(str)
        m = oo.merge(zo, on="_k", suffixes=("_oo", "_z"))
        m = m[m["predicted_value_cm_oo"] > 0]
        if len(m) < 10:
            return None
        return float((m["predicted_value_cm_z"] / m["predicted_value_cm_oo"]).median())

    r_in = median_ratio(RES / lora_dir / "in_the_wild_zoom_in" / "predictions.csv")
    r_out = median_ratio(RES / lora_dir / "in_the_wild_zoom_out" / "predictions.csv")
    return _invariance_score(r_in, r_out)


def base_dist_avg_acc(slug):
    vals = []
    for tag in ("distorted", "distorted_pin015"):
        m = metrics_from(RES / f"{tag}_{slug}")
        if m is not None:
            vals.append(m["acc10"])
    if not vals:
        return None
    return sum(vals) / len(vals)


# ─── LoRA loaders (new directory layout) ───────────────────────────────────
def _acc10_from_csv(path, threshold=0.10):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "predicted_value_cm" not in df.columns or "answer" not in df.columns:
        return None
    v = df[df["predicted_value_cm"].notna() & df["answer"].notna()].copy()
    v = v[v["predicted_value_cm"].abs() < 1e4]
    if len(v) == 0:
        return None
    rel = (v["predicted_value_cm"] - v["answer"]).abs() / v["answer"].abs().clip(lower=1e-6)
    return float((rel <= threshold).mean())


def lora_acc(lora_dir, condition):
    return _acc10_from_csv(RES / lora_dir / condition / "predictions.csv")


def lora_ctx_acc(lora_dir, variant):
    """LoRA context predictions are in a single 5-variant CSV with a 'variant' column."""
    p = RES / lora_dir / "in_the_wild_context_5variants" / "predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[df["variant"] == variant]
    df["answer"] = pd.to_numeric(df["answer"], errors="coerce")
    df["predicted_value_cm"] = pd.to_numeric(df["predicted_value_cm"], errors="coerce")
    df = df.dropna(subset=["predicted_value_cm", "answer"])
    df = df[df["predicted_value_cm"].abs() < 1e4]
    if len(df) == 0:
        return None
    rel = (df["predicted_value_cm"] - df["answer"]).abs() / df["answer"].abs().clip(lower=1e-6)
    return float((rel <= 0.10).mean())


def lora_no_image_acc(lora_dir, prefer_text_only=True):
    """Mirror the base model's probe choice for an apples-to-apples comparison.
    If the base used text-only (prefer_text_only=True), use text-only here;
    otherwise use gray-blank. This avoids the case where the base refused
    text-only and silently fell back to gray, while the LoRA (which can now
    answer text-only) gets its much worse text-only number on the radar."""
    if prefer_text_only:
        return lora_acc(lora_dir, "in_the_wild_text_only")
    return lora_acc(lora_dir, "in_the_wild_gray")


def lora_dist_avg_acc(lora_dir):
    vals = []
    for tag in ("in_the_wild_distorted_k015", "in_the_wild_distorted_pin015"):
        a = lora_acc(lora_dir, tag)
        if a is not None:
            vals.append(a)
    if not vals:
        return None
    return sum(vals) / len(vals)


# ─── axis spec: (label, base_loader, lora_loader) ──────────────────────────
def base_zoom_in_acc(slug):
    m = zoom_in(slug)
    return m["acc10"] if m is not None else None


def base_zoom_out_acc(slug):
    m = zoom_out(slug)
    return m["acc10"] if m is not None else None


def base_loaders(slug):
    return [
        ("no\nimage",     base_no_image_acc(slug)),
        ("object\nonly",  base_ctx_acc(slug, "object_only")),
        ("object\nring",  base_ctx_acc(slug, "object_ring")),
        ("wrong\nobject", base_ctx_acc(slug, "wrong_object")),
        ("zoom\nin 1.4$\\times$",  base_zoom_in_acc(slug)),
        ("zoom\nout 0.6$\\times$", base_zoom_out_acc(slug)),
        ("lens\ndist.",   base_dist_avg_acc(slug)),
    ]


def lora_loaders(lora_dir, base_prefer_text_only=True):
    return [
        ("no\nimage",     lora_no_image_acc(lora_dir, base_prefer_text_only)),
        ("object\nonly",  lora_ctx_acc(lora_dir, "object_only")),
        ("object\nring",  lora_ctx_acc(lora_dir, "object_ring")),
        ("wrong\nobject", lora_ctx_acc(lora_dir, "wrong_object")),
        ("zoom\nin 1.4$\\times$",  lora_acc(lora_dir, "in_the_wild_zoom_in")),
        ("zoom\nout 0.6$\\times$", lora_acc(lora_dir, "in_the_wild_zoom_out")),
        ("lens\ndist.",   lora_dist_avg_acc(lora_dir)),
    ]


BASE_COLOR = "#e67e22"  # orange
LORA_COLOR = "#2c4a8a"  # blue

N = 7  # axes: no_image, object_only, object_ring, wrong_object, zoom_in, zoom_out, lens_dist
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
theta_closed = np.concatenate([theta, theta[:1]])


def build_panel_data(panels):
    """Return list of (label, axis_names, base_vals, lora_vals,
    base_full_acc, lora_full_acc) for the given (label, base_slug, lora_dir)
    triples. Invariance values stay in raw 0..1; rescaling happens at plot time."""
    data = []
    for label, base_slug, lora_dir in panels:
        # Did the base use text-only or gray-blank on the no-image axis?
        base_prefer_text_only = base_text_only(base_slug) is not None
        base_pts = base_loaders(base_slug)
        lora_pts = lora_loaders(lora_dir, base_prefer_text_only)
        axis_names = [n for n, _ in base_pts]
        base_vals = np.array([v if v is not None else np.nan for _, v in base_pts],
                             dtype=float)
        lora_vals = np.array([v if v is not None else np.nan for _, v in lora_pts],
                             dtype=float)
        base_full = vision_itw(base_slug)
        base_full_acc = base_full["acc10"] if base_full else np.nan
        lora_full_acc = lora_acc(lora_dir, "in_the_wild_vision")
        data.append((label, axis_names, base_vals, lora_vals,
                     base_full_acc, lora_full_acc))
    return data


def render_lora_radar(panels, out_path, n_rows=1, n_cols=None, figsize=None):
    panel_data = build_panel_data(panels)
    all_vals = np.concatenate([
        np.concatenate([bv[~np.isnan(bv)], lv[~np.isnan(lv)]])
        for _, _, bv, lv, _, _ in panel_data
    ])
    ymax = float(np.nanmax(all_vals)) * 1.10 if len(all_vals) else 0.3
    yticks = [y for y in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30) if y <= ymax]

    if n_cols is None:
        n_cols = len(panels)
    if figsize is None:
        figsize = (5.5 * n_cols, 5.0 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             subplot_kw=dict(projection="polar"))
    axes_flat = np.array(axes).flatten()
    _render_panels(axes_flat[:len(panels)], panel_data, ymax, yticks)
    # hide any extra slots
    for ax in axes_flat[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")
    plt.close(fig)


def _render_panels(axes, panel_data, ymax, yticks):
    for ax, (label, axis_names, base_vals, lora_vals, base_full, lora_full) \
            in zip(axes, panel_data):
        # base polygon
        bv = base_vals.copy()
        bv_nan = np.isnan(bv)
        bv[bv_nan] = 0
        b_closed = np.concatenate([bv, bv[:1]])
        ax.plot(theta_closed, b_closed, color=BASE_COLOR, linewidth=1.7,
                label=f"base ({base_full*100:.1f}%)"
                if not np.isnan(base_full) else "base")
        ax.fill(theta_closed, b_closed, color=BASE_COLOR, alpha=0.14)

        # lora polygon
        lv = lora_vals.copy()
        lv_nan = np.isnan(lv)
        lv[lv_nan] = 0
        l_closed = np.concatenate([lv, lv[:1]])
        ax.plot(theta_closed, l_closed, color=LORA_COLOR, linewidth=1.7,
                label=f"+ LoRA ({lora_full*100:.1f}%)"
                if lora_full is not None else "+ LoRA")
        ax.fill(theta_closed, l_closed, color=LORA_COLOR, alpha=0.14)

        # full-image baseline hexagons (one per model, constant radius)
        if not np.isnan(base_full):
            ax.plot(theta_closed, np.full(N + 1, base_full),
                    color=BASE_COLOR, linewidth=1.0, linestyle="--",
                    alpha=0.7, label="base full-image")
        if lora_full is not None:
            ax.plot(theta_closed, np.full(N + 1, lora_full),
                    color=LORA_COLOR, linewidth=1.0, linestyle="--",
                    alpha=0.7, label="+ LoRA full-image")

        # vertex markers
        for i in range(N):
            if not bv_nan[i]:
                ax.plot(theta[i], base_vals[i], "o", color=BASE_COLOR,
                        markersize=4.5, markeredgecolor="white",
                        markeredgewidth=0.5, zorder=5)
            if not lv_nan[i]:
                ax.plot(theta[i], lora_vals[i], "o", color=LORA_COLOR,
                        markersize=4.5, markeredgecolor="white",
                        markeredgewidth=0.5, zorder=5)

        ax.set_xticks(theta)
        ax.set_xticklabels(axis_names, fontsize=8)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{int(y*100)}%" for y in yticks], fontsize=7,
                           color="#888")
        ax.set_ylim(0, ymax)
        ax.tick_params(axis="x", pad=4)
        ax.set_rlabel_position(95)
        ax.grid(color="#ddd", linewidth=0.6)
        ax.spines["polar"].set_color("#bbb")
        ax.set_title(label, fontsize=11, pad=14)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10),
                  fontsize=8, frameon=False)


OUT_DIR = HERE.parent / "emnlp_paper" / "figs"

# ─── The 3 paper-pool LoRA variants, 1x3 horizontal grid ──────────────────
render_lora_radar(
    panels=[
        ("Qwen2.5-VL-3B",        "Qwen2.5-VL-3B-Instruct",
         "lora_v3_lora_Qwen2.5-VL-3B-Instruct"),
        ("Qwen2.5-VL-7B",        "Qwen2.5-VL-7B-Instruct",
         "lora_v3_lora_Qwen2.5-VL-7B-Instruct"),
        ("LLaVA-v1.6-Mistral-7B","LLaVA-v1.6-Mistral-7B",
         "lora_v3_lora_llava-v1.6-mistral-7b"),
    ],
    out_path=OUT_DIR / "lora_radar.pdf",
    n_rows=1, n_cols=3,
    figsize=(15.0, 5.4),
)
