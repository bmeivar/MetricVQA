"""
Side-by-side pie charts comparing object-class composition of the two
benchmark splits.

Objectron: 9 clean product categories (filename prefix).
In-the-wild: free-form queries, bucketed by keyword into 8 broad
domains (furniture, kitchen, toy, decor, electronics, container,
clothing/textile, other).

Output: paper/emnlp_paper/figs/dataset_composition.pdf
"""

from __future__ import annotations

import re
from collections import Counter
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
REPO = Path(os.environ.get("METRICVQA_ROOT", "."))


# ─── ITW bucketing ─────────────────────────────────────────────────────────
# Keyword → bucket. First match wins (order matters: more-specific first).
ITW_BUCKETS = [
    # Toys / playthings
    ("toy",          [r"\btoy\b", r"\bdoll\b", r"\bblock\b", r"\bpuzzle\b",
                      r"\bstuffed\b", r"\bplush\b", r"\blego\b", r"\bgame\b",
                      r"stacking ring", r"piano", r"fire truck", r"pop-up",
                      r"\bdice\b", r"\babacus\b", r"\bjenga\b",
                      r"\baccordion\b", r"slinky", r"play-mat", r"play mat",
                      r"foam mat", r"\bcar\b", r"truck"]),
    # Kitchen / dining
    ("kitchen",      [r"\bkettle\b", r"\bpan\b", r"\bpot\b", r"\bbowl\b",
                      r"\bmug\b", r"\bcup\b", r"\bplate\b", r"\bjar\b",
                      r"\bbottle\b", r"\bglass\b", r"\bfaucet\b", r"\bsink\b",
                      r"\boven\b", r"microwave", r"\bstove\b", r"\bfridge\b",
                      r"cutting board", r"coffee", r"\btea\b", r"teapot",
                      r"\bpitcher\b", r"\bbucket\b", r"coffee maker",
                      r"\bspoon\b", r"\bfork\b", r"\bknife\b", r"\bplate\b",
                      r"colander", r"strainer", r"toaster", r"blender",
                      r"\bsalt\b", r"\bpepper\b", r"\bspice", r"\bcoaster\b",
                      r"napkin", r"cake stand", r"soap dispenser",
                      r"refrigerator", r"refregirator", r"fridge"]),
    # Electronics / appliances
    ("electronics",  [r"remote", r"\bphone\b", r"laptop", r"monitor",
                      r"computer", r"\btv\b", r"\bscreen\b", r"keyboard",
                      r"\bmouse\b", r"camera", r"speaker", r"controller",
                      r"router", r"charger", r"headphone", r"earbud",
                      r"console", r"printer", r"\bcable\b", r"\boutlet\b",
                      r"socket", r"\bplug\b", r"\bbutton\b", r"\bswitch\b",
                      r"television", r"\bac engine\b", r"\bac unit\b",
                      r"\bamplifier\b", r"\badapter\b", r"whiteboard"]),
    # Decor / art / lighting
    ("decor / lighting", [r"painting", r"picture", r"\bframe\b", r"vase",
                          r"\bplant\b", r"planter", r"candle", r"\blamp\b",
                          r"mirror", r"\bart\b", r"sculpture", r"statue",
                          r"clock", r"poster", r"\brug\b", r"\bornament\b",
                          r"\bcurtain\b", r"\bblind\b", r"\bhorse\b",
                          r"\bdeer\b", r"figurine", r"\bglobe\b",
                          r"\btree\b", r"metal cat"]),
    # Furniture / structural
    ("furniture",    [r"\bchair\b", r"\btable\b", r"\bshelf\b", r"\bshelves\b",
                      r"\bstool\b", r"ottoman", r"cabinet", r"\bdesk\b",
                      r"drawer", r"\bbed\b", r"\bsofa\b", r"couch",
                      r"\bbench\b", r"dresser", r"nightstand", r"\bdoor\b",
                      r"\bwindow\b", r"\bwall\b", r"\bfloor\b",
                      r"\bcounter\b", r"\bcountertop\b", r"backrest",
                      r"\bslat\b", r"\barmrest\b", r"\bleg\b",
                      r"\bbasket\b", r"\bcrate\b", r"\borganizer\b",
                      r"holder", r"\brail\b", r"railing", r"\brim of the pool\b",
                      r"\bpool\b", r"step", r"compartment", r"\brack\b",
                      r"drywall", r"\bledge\b"]),
    # Containers / boxes / bins
    ("container",    [r"\bbox\b", r"\bbin\b", r"container",
                      r"\bcan\b", r"\bcanister\b", r"\bjug\b", r"\btub\b",
                      r"\bbag\b", r"backpack", r"\bpurse\b", r"suitcase",
                      r"\bball\b", r"basketball", r"soccer ball",
                      r"aquarium", r"water tank", r"\bbarrel\b"]),
    # Textile / clothing
    ("clothing / textile",   [r"\bshirt\b", r"\bhat\b", r"\bshoe\b",
                              r"\bsock\b", r"scarf", r"\bglove\b",
                              r"\bjacket\b", r"\bpants\b", r"\bdress\b",
                              r"\btowel\b", r"\bblanket\b", r"\bpillow\b",
                              r"flip flop", r"\byoga mat\b", r"\bmat\b"]),
    # Books / paper
    ("book / paper", [r"\bbook\b", r"\bnotebook\b", r"\bmagazine\b",
                      r"\bpaper\b", r"toilet paper", r"\benvelope\b",
                      r"\bcard\b", r"\bnote\b", r"\bbinder\b"]),
    # Small accessories / personal
    ("accessory",    [r"\bkey\b", r"\bwatch\b", r"umbrella",
                      r"\bwallet\b", r"sunglass", r"\bbrush\b",
                      r"\btag\b", r"\bbroom\b", r"\bcomb\b",
                      r"\bband\b", r"\bring\b"]),
    # Musical
    ("musical",      [r"\bdrum", r"\bguitar\b", r"\bukulele\b",
                      r"\bharmonica\b"]),
]

ITW_BUCKET_NAMES = [b for b, _ in ITW_BUCKETS] + ["other"]


def _pat_label(pat: str) -> str:
    """Clean a regex pattern into a human-readable label."""
    s = pat.replace("\\b", "")
    # drop simple regex metachars
    for ch in ("\\", "(", ")", "?", "*", "+", "^", "$", "|", "[", "]"):
        s = s.replace(ch, "")
    return s.strip()


def bucket_itw(query: str) -> tuple[str, str]:
    """Return (coarse_bucket, fine_keyword) for the first matching pattern."""
    q = query.lower()
    for name, patterns in ITW_BUCKETS:
        for pat in patterns:
            if re.search(pat, q):
                return name, _pat_label(pat)
    return "other", "other"


# ─── load each split ───────────────────────────────────────────────────────
itw = pd.read_csv(REPO / "questions_v2.csv")
buckets_fines = itw["query"].astype(str).apply(bucket_itw)
itw["bucket"] = buckets_fines.apply(lambda t: t[0])
itw["fine"]   = buckets_fines.apply(lambda t: t[1])

# Aggregate fine categories: (count, mean GT cm)
itw_fine_stats = (
    itw.groupby("fine")
       .agg(n=("answer", "count"), mean_gt=("answer", "mean"))
       .sort_values("n", ascending=False)
       .reset_index()
)
print(f"ITW: {len(itw_fine_stats)} distinct fine categories")
print(itw_fine_stats.head(15).to_string(index=False))

# Objectron: extract category from filename prefix (bike__XXXX.png → bike)
obj_df = pd.read_csv(REPO / "results/objectron_chatgpt5.5_prior/predictions.csv")
obj_df["category"] = obj_df["image"].str.split("__").str[0]
obj_df["answer"] = pd.to_numeric(obj_df["answer"], errors="coerce")
obj_fine_stats = (
    obj_df.groupby("category")
          .agg(n=("answer", "count"), mean_gt=("answer", "mean"))
          .sort_values("n", ascending=False)
          .reset_index()
)
print(f"\nObjectron: {len(obj_fine_stats)} categories")
print(obj_fine_stats.to_string(index=False))


# ─── plot ──────────────────────────────────────────────────────────────────
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

fig = plt.figure(figsize=(12.5, 5.9))
gs = fig.add_gridspec(2, 2, height_ratios=[28, 1], hspace=0.05,
                      top=0.86, bottom=0.10, left=0.04, right=0.96)
ax_l = fig.add_subplot(gs[0, 0])
ax_r = fig.add_subplot(gs[0, 1])
ax_cb = fig.add_subplot(gs[1, :])

# Shared colour scale: log10(GT cm). Both splits span ~0.8 → ~245 cm.
all_gts = list(itw_fine_stats["mean_gt"]) + list(obj_fine_stats["mean_gt"])
vmin = max(min(all_gts), 0.5)
vmax = max(all_gts)
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
cmap = plt.colormaps["viridis"]


def _single_ring_pie(ax, stats, raw_gts, title,
                     show_slice_labels=True,
                     gt_lists=None,
                     gt_floor=None):
    """One slice per row of `stats` (columns: fine/category, n, mean_gt).

    Slices sized by `n`. Two colouring modes:
      - If `gt_lists` is None (default): one solid colour per slice = mean GT.
      - If `gt_lists` is a dict {name -> array of GTs}: each slice is
        sub-divided into per-query micro-wedges sorted by GT, producing a
        visible within-category gradient (min→max colour).

    `gt_floor` (optional) drops queries with GT below this value when
    computing the centre-annotation range (used to ignore Objectron's
    near-zero degenerate axes).
    """
    sizes = stats["n"].values
    gts = stats["mean_gt"].values
    names = (stats["fine"].values if "fine" in stats.columns
             else stats["category"].values)
    total = sizes.sum()

    if gt_lists is None:
        # one-colour-per-slice
        colors = [cmap(norm(g)) for g in gts]
        labels = []
        if show_slice_labels:
            for i, (name, n) in enumerate(zip(names, sizes)):
                pct = 100.0 * n / total
                if pct >= 1.5:
                    labels.append(f"{name}\n({pct:.0f}%)")
                else:
                    labels.append("")
        else:
            labels = [""] * len(sizes)
        wedges, texts = ax.pie(sizes, labels=labels, colors=colors,
               startangle=90, labeldistance=1.05,
               textprops={"fontsize": 8.5},
               wedgeprops={"edgecolor": "white", "linewidth": 0.6})
        # Nudge specific labels that crowd their neighbour. Use a name
        # lookup so the offsets stay correct if categories are reordered.
        nudge = {"bike": (-0.10, 0.05)}  # (dx, dy) in axes coords
        for txt, name in zip(texts, names):
            if name in nudge:
                dx, dy = nudge[name]
                x, y = txt.get_position()
                txt.set_position((x + dx, y + dy))
    else:
        # per-query micro-wedges within each category for a visible gradient
        sub_sizes = []
        sub_colors = []
        label_positions = []  # (name, pct) for category labels
        running_count = 0
        for name, n in zip(names, sizes):
            qs = sorted([g for g in gt_lists.get(name, [])
                         if pd.notna(g) and g > 0])
            if not qs:
                continue
            # one sub-wedge per query
            for g in qs:
                sub_sizes.append(1.0)
                sub_colors.append(cmap(norm(g)))
            # capture label info
            label_positions.append((name, 100.0 * n / total))
            running_count += len(qs)
        # draw the gradient pie
        ax.pie(sub_sizes, colors=sub_colors,
               startangle=90,
               wedgeprops={"edgecolor": "none", "linewidth": 0})
        # overlay category-boundary lines and labels
        # We need to redraw with thin separators between categories.
        # Easiest: draw a second transparent pie at category sizes with
        # only edges visible.
        labels = []
        if show_slice_labels:
            for i, (name, pct) in enumerate(label_positions):
                if pct >= 5:
                    labels.append(f"{name}\n({pct:.0f}%)")
                else:
                    labels.append(name)
        else:
            labels = [""] * len(label_positions)
        ax.pie([n for _, n in zip(names, sizes)],
               labels=labels,
               colors=["none"] * len(sizes),
               startangle=90, labeldistance=1.05,
               textprops={"fontsize": 8.5},
               wedgeprops={"edgecolor": "white", "linewidth": 1.4,
                           "fill": False})

    # Annotate per-query GT span inside the donut
    valid = raw_gts.dropna() if hasattr(raw_gts, "dropna") else raw_gts
    valid = valid[valid > (gt_floor if gt_floor is not None else 0)]
    q_min, q_max = valid.min(), valid.max()
    span = f"GT range\n{q_min:.1f} cm $\\to$ {q_max:.0f} cm"
    ax.text(0, 0, f"{len(stats)} distinct\nobjects\n\n{span}",
            ha="center", va="center", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="#bbb",
                      boxstyle="round,pad=0.4", linewidth=0.7))
    # panel title removed: LaTeX caption distinguishes the two panels


# Convert ITW stats DataFrame name to match what _single_ring_pie expects
itw_fine_for_pie = itw_fine_stats.rename(columns={})  # already has 'fine'
obj_fine_for_pie = obj_fine_stats.rename(columns={"category": "fine"})

_single_ring_pie(
    ax_l, itw_fine_for_pie, itw["answer"].astype(float),
    f"In-the-wild ($n{{=}}{sum(itw_fine_stats['n'])}$ queries)",
    show_slice_labels=False)

_single_ring_pie(
    ax_r, obj_fine_for_pie, obj_df["answer"].astype(float),
    f"Objectron ($n{{=}}{sum(obj_fine_stats['n'])}$ queries)",
    show_slice_labels=True,
    gt_floor=1.0)  # ignore degenerate sub-1cm axes from flat 3D boxes

# Shared colour bar
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cb = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
cb.set_label("mean ground-truth dimension per object (cm, log scale)",
             fontsize=9.5)
cb.ax.tick_params(labelsize=8)

# suptitle removed: LaTeX caption is the figure's only title
# gridspec already places everything; tight_layout would re-add padding

OUT = HERE.parent / "emnlp_paper" / "figs" / "dataset_composition.pdf"
fig.savefig(OUT, bbox_inches="tight")
print(f"\nwrote {OUT}")
