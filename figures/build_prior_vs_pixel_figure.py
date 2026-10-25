"""
"Prior vs Pixel" diagnostic scatter plot.

X-axis: gray-blank MAE  (model with no image — pure internal LM prior)
Y-axis: full-image MAE  (model with the original image)

One point per model on the in-the-wild split. The y=x diagonal separates
models that benefit from vision (below diagonal) from those that don't (above
diagonal). The text-only frontier LLM (ChatGPT 5.5) is shown as a vertical
guideline for what an image-blind frontier LLM achieves.

Output: paper/emnlp/figs/prior_vs_pixel.pdf
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
from build_tables import MODELS, vision_itw, gray_itw, text_only_itw, baseline  # noqa: E402


# For each model, prefer text-only (the clean language-prior probe); fall back
# to gray-blank when the model refuses to operate without an image
# (PaliGemma, Molmo, InternVL2 return empty responses text-only).
# Points are Acc@10% (higher = better, opposite of MAE) — MAE on ITW is
# dominated by outliers over the 0.8-245 cm GT range.
# is_fallback flags which marker style to draw.
points = []  # (label, x_acc, y_acc, is_fallback)
for slug, label in MODELS:
    v = vision_itw(slug)
    if v is None:
        continue
    t = text_only_itw(slug)
    if t is not None:
        points.append((label, t["acc10"], v["acc10"], False))
    else:
        g = gray_itw(slug)
        if g is None:
            continue
        points.append((label, g["acc10"], v["acc10"], True))

# Reference baselines (Acc@10%)
llm = baseline("llm_itw")  # ChatGPT 5.5 text-only on in-the-wild
human = baseline("human_itw")
llm_v = llm["acc10"] if llm else None
human_v = human["acc10"] if human else None

# Plot
fig, ax = plt.subplots(figsize=(6.0, 5.4))

xs = np.array([p[1] for p in points])
ys = np.array([p[2] for p in points])
is_fb = np.array([p[3] for p in points], dtype=bool)

# y=x diagonal — for Acc@10% (higher = better), above diagonal means vision
# helps; below diagonal means vision hurts.
# Range chosen to keep the two largest open VLMs (30.8% ITW) and the
# ChatGPT 5.5 / human references (32.4% / 39.3%) inside the frame.
lo = 0.0
hi = 0.42
ax.plot([lo, hi], [lo, hi], color="#888", linestyle="--", linewidth=1,
        label=r"$y\!=\!x$  (vision adds nothing)")

# Shade: above diagonal = vision helps (higher with-image Acc); below = hurts
ax.fill_between([lo, hi], [lo, hi], hi, color="#e6eef7", alpha=0.6, zorder=0)
ax.fill_between([lo, hi], lo, [lo, hi], color="#fbeede", alpha=0.6, zorder=0)
ax.text(hi * 0.97, hi * 0.03, "vision hurts",
        ha="right", va="bottom", color="#a86a14", fontsize=9, alpha=0.95)

# Reference: frontier LLM (no image) — vertical dashed line, inline label
if llm_v is not None and llm_v <= hi:
    ax.axvline(llm_v, color="#7a3b8a", linewidth=1.2, alpha=0.7,
               linestyle=":")
    ax.text(llm_v + hi * 0.005, hi * 0.50,
            f"LLM reference ({llm_v*100:.1f}%)",
            rotation=90, color="#7a3b8a", fontsize=8.5,
            ha="left", va="center", alpha=0.95)

# Reference: human (with image) — horizontal dashed line, inline label
if human_v is not None and human_v <= hi:
    ax.axhline(human_v, color="#3a6aa8", linewidth=1.2, alpha=0.7,
               linestyle=":")
    ax.text(hi * 0.50, human_v + hi * 0.005,
            f"human reference ({human_v*100:.1f}%)",
            color="#3a6aa8", fontsize=8.5,
            ha="center", va="bottom", alpha=0.95)

# Points + labels — text-only models = filled black; gray-blank fallback = open
ax.scatter(xs[~is_fb], ys[~is_fb], s=64, c="#1a1a1a",
           edgecolors="white", linewidth=1.2, zorder=3,
           label="text-only x-axis")
ax.scatter(xs[is_fb], ys[is_fb], s=72, facecolors="none",
           edgecolors="#1a1a1a", linewidth=1.4, zorder=3,
           label="gray-blank fallback (no text-only)")

# annotate with smart offsets to reduce overlap. Offsets are in display
# points; leader lines connect labels back to their dots.
offsets = {
    "Qwen2.5-VL-3B":  (14, -10),  # down-right
    "PaliGemma-3B":   (-14, -10), # down-left (fallback marker)
    "SmolVLM":        (-14, 10),  # up-left
    "Gemma3-4B":      (14, -10),  # down-right
    "granite-3.3-2b": (-14, 8),   # up-left
    "Qwen2.5-VL-7B":  (-14, 12),  # up-left
    "LLaVA-OV":       (-18, -18), # far-left, below   (vertical stack with Molmo/InternVL2)
    "Molmo-7B-D":     (-18, 12),  # far-left, above   (vertical stack)
    "InternVL2":      (-18, 0),   # far-left, middle  (vertical stack)
    "InternVL3.5":    (14, -8),   # down-right (clear of Qwen2.5-VL-7B above)
    "Qwen3-VL-235B":  (-14, 14),  # up-left (above the dot)
    "Qwen3.5-397B":   (14, -12),  # down-right (below the dot)
}
for label, x, y, _ in points:
    dx, dy = offsets.get(label, (6, 4))
    ha = "left" if dx > 0 else "right"
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8.5,
        ha=ha,
        va="center",
        arrowprops=dict(
            arrowstyle="-",
            color="#888",
            lw=0.6,
            shrinkA=1,    # leave a small gap at the label end
            shrinkB=5,    # leave a gap at the dot end so line doesn't touch the marker
        ),
    )

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal")
ax.set_xlabel("Language-prior Acc@10%", fontsize=10)
ax.set_ylabel("Full-image Acc@10%", fontsize=10)

from matplotlib.ticker import FuncFormatter
fmt = FuncFormatter(lambda v, _: f"{v*100:.0f}")
ax.xaxis.set_major_formatter(fmt)
ax.yaxis.set_major_formatter(fmt)
# title removed: LaTeX caption is the figure's only title
ax.tick_params(axis="both", labelsize=9)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.95)

fig.tight_layout()
OUT = FIGS / "prior_vs_pixel.pdf"
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
