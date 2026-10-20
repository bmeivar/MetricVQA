#!/usr/bin/env python3
"""
Analyze VLM size-estimation predictions across all model variants.

Reads predictions.csv from each subfolder, computes accuracy at various
relative-error thresholds (10 %, 15 %, 20 %), and identifies extreme outliers.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(os.environ.get("METRICVQA_ROOT",
                                  Path(__file__).resolve().parent)) / "results"
THRESHOLDS = [0.10, 0.15, 0.20]           # relative-error thresholds
OUTLIER_MULTIPLIER = 3.0                    # IQR multiplier for outlier detection

def display_name(folder_name: str) -> str:
    """Convert folder name to a readable display name."""
    return folder_name.replace("_", " ")

# ──────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────

def load_all_predictions(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load predictions.csv from every model subfolder."""
    data = {}
    for folder in sorted(results_dir.iterdir()):
        csv_path = folder / "predictions.csv"
        if folder.is_dir() and csv_path.exists():
            df = pd.read_csv(csv_path)
            # normalise column name (float_qwen_7 has an extra column)
            if "Number predicted_value_cm" in df.columns:
                df = df.drop(columns=["Number predicted_value_cm"])
            # coerce to numeric; un-parseable predictions become NaN
            df["answer"] = pd.to_numeric(df["answer"], errors="coerce")
            df["predicted_value_cm"] = pd.to_numeric(
                df["predicted_value_cm"], errors="coerce"
            )
            df["model"] = folder.name
            data[folder.name] = df
    return data


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add absolute and relative error columns."""
    df = df.copy()
    df["abs_error"] = (df["predicted_value_cm"] - df["answer"]).abs()
    df["rel_error"] = df["abs_error"] / df["answer"].abs()
    return df

# ──────────────────────────────────────────────────────────────────────
# Accuracy metrics
# ──────────────────────────────────────────────────────────────────────

def compute_accuracy_table(
    datasets: dict[str, pd.DataFrame], thresholds: list[float]
) -> pd.DataFrame:
    """Return a DataFrame with accuracy at each threshold for every model."""
    rows = []
    for model, df in datasets.items():
        valid = df.dropna(subset=["rel_error"])
        n_valid = len(valid)
        n_total = len(df)
        n_failed = n_total - n_valid
        row = {
            "Model": display_name(model),
            "N total": n_total,
            "N valid": n_valid,
            "N failed parse": n_failed,
            "MAE (cm)": valid["abs_error"].mean(),
            "Median AE (cm)": valid["abs_error"].median(),
        }
        for t in thresholds:
            acc = (valid["rel_error"] <= t).mean()
            row[f"Acc@{int(t*100)}%"] = acc
        rows.append(row)
    return pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────────────
# Outlier analysis
# ──────────────────────────────────────────────────────────────────────

def detect_outliers(df: pd.DataFrame, multiplier: float = 3.0) -> pd.DataFrame:
    """Flag extreme outliers using IQR on relative error."""
    valid = df.dropna(subset=["rel_error"]).copy()
    q1 = valid["rel_error"].quantile(0.25)
    q3 = valid["rel_error"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + multiplier * iqr
    outliers = valid[valid["rel_error"] > upper].copy()
    outliers = outliers.sort_values("rel_error", ascending=False)
    return outliers


def outlier_report(datasets: dict[str, pd.DataFrame], multiplier: float = 3.0):
    """Print a detailed outlier report for each model."""
    print("\n" + "=" * 90)
    print(f"EXTREME OUTLIER ANALYSIS  (IQR × {multiplier})")
    print("=" * 90)

    all_outliers = []
    for model, df in datasets.items():
        outliers = detect_outliers(df, multiplier)
        display = display_name(model)
        n_valid = df.dropna(subset=["rel_error"]).shape[0]
        print(f"\n{'─' * 90}")
        print(f"  {display}:  {len(outliers)} outliers out of {n_valid} valid predictions")
        print(f"{'─' * 90}")
        if outliers.empty:
            print("  No extreme outliers detected.")
            continue
        for _, r in outliers.head(10).iterrows():
            print(
                f"  {r['image']:20s} | {r['query'][:45]:45s} | "
                f"GT={r['answer']:>7.1f} cm  Pred={r['predicted_value_cm']:>8.1f} cm  "
                f"RelErr={r['rel_error']:>7.1%}"
            )
        outliers["model_display"] = display
        all_outliers.append(outliers)

    if all_outliers:
        combined = pd.concat(all_outliers)
        print(f"\n{'─' * 90}")
        print(f"  WORST 15 PREDICTIONS ACROSS ALL MODELS")
        print(f"{'─' * 90}")
        worst = combined.sort_values("rel_error", ascending=False).head(15)
        for _, r in worst.iterrows():
            print(
                f"  [{r['model_display']:26s}] {r['image']:16s} | "
                f"{r['query'][:40]:40s} | "
                f"GT={r['answer']:>7.1f}  Pred={r['predicted_value_cm']:>8.1f}  "
                f"RelErr={r['rel_error']:>8.1%}"
            )

        # Summarise common patterns in outliers
        print(f"\n{'─' * 90}")
        print("  OUTLIER PATTERN ANALYSIS")
        print(f"{'─' * 90}")

        # Check if small objects are harder
        median_gt = combined["answer"].median()
        small = combined[combined["answer"] < median_gt]
        large = combined[combined["answer"] >= median_gt]
        print(f"  Outliers on small objects (GT < {median_gt:.1f} cm): {len(small)}")
        print(f"  Outliers on large objects (GT >= {median_gt:.1f} cm): {len(large)}")

        # Over-prediction vs under-prediction
        over = combined[combined["predicted_value_cm"] > combined["answer"]]
        under = combined[combined["predicted_value_cm"] < combined["answer"]]
        print(f"  Over-predictions:  {len(over)}")
        print(f"  Under-predictions: {len(under)}")

        # Most error-prone images
        print("\n  Most error-prone images (appearing in outliers across models):")
        img_counts = combined.groupby("image").size().sort_values(ascending=False)
        for img, cnt in img_counts.head(5).items():
            queries = combined[combined["image"] == img]["query"].unique()
            print(f"    {img}: {cnt} outlier(s)  –  queries: {', '.join(q[:50] for q in queries)}")

        # Most error-prone query types
        print("\n  Most error-prone query keywords:")
        combined["query_lower"] = combined["query"].str.lower()
        keywords = {}
        for kw in ["long", "wide", "tall", "high", "diameter", "thick"]:
            n = combined["query_lower"].str.contains(kw).sum()
            if n > 0:
                keywords[kw] = n
        for kw, n in sorted(keywords.items(), key=lambda x: -x[1]):
            print(f"    '{kw}': {n} outlier(s)")

# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def plot_accuracy_bars(acc_table: pd.DataFrame, save_path: Path):
    """Grouped bar chart of accuracy at each threshold."""
    models = acc_table["Model"]
    acc_cols = [c for c in acc_table.columns if c.startswith("Acc@")]
    x = np.arange(len(models))
    width = 0.22
    offsets = np.arange(len(acc_cols)) - (len(acc_cols) - 1) / 2

    fig, ax = plt.subplots(figsize=(13, 6))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    for i, col in enumerate(acc_cols):
        vals = acc_table[col].values
        bars = ax.bar(x + offsets[i] * width, vals * 100, width,
                      label=col, color=colors[i], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("VLM Size Estimation — Accuracy at Different Error Thresholds",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, min(acc_table[acc_cols].max().max() * 100 + 15, 100))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved accuracy bar chart → {save_path}")


def plot_error_distribution(datasets: dict[str, pd.DataFrame], save_path: Path):
    """Box plot + strip plot of relative errors per model."""
    frames = []
    for model, df in datasets.items():
        valid = df.dropna(subset=["rel_error"]).copy()
        valid["Model"] = display_name(model)
        frames.append(valid)
    combined = pd.concat(frames)

    # Clip for visualisation (keep real data for stats)
    combined["rel_error_clipped"] = combined["rel_error"].clip(upper=5.0)

    fig, ax = plt.subplots(figsize=(13, 6))
    models_order = [display_name(m) for m in datasets]
    positions = range(len(models_order))

    bp_data = [
        combined.loc[combined["Model"] == m, "rel_error_clipped"].values
        for m in models_order
    ]
    bp = ax.boxplot(bp_data, positions=list(positions), widths=0.5,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=3, alpha=0.4))

    n_models = len(models_order)
    cmap = plt.colormaps["tab20"] if n_models > 10 else plt.colormaps["tab10"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor((*cmap(i / max(n_models - 1, 1))[:3], 0.4))

    # Threshold lines
    for t in THRESHOLDS:
        ax.axhline(t, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(len(models_order) - 0.5, t + 0.02, f"{int(t*100)}%",
                color="red", fontsize=9, ha="right")

    ax.set_xticks(list(positions))
    ax.set_xticklabels(models_order, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Relative Error (clipped at 500%)", fontsize=12)
    ax.set_title("Distribution of Relative Errors per Model", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved error distribution plot → {save_path}")


def plot_scatter_pred_vs_gt(datasets: dict[str, pd.DataFrame], save_path: Path):
    """Scatter: predicted vs ground-truth for each model (log scale)."""
    n_models = len(datasets)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows),
                             squeeze=False)
    axes = axes.flatten()

    for idx, (model, df) in enumerate(datasets.items()):
        ax = axes[idx]
        valid = df.dropna(subset=["predicted_value_cm", "answer"])
        valid = valid[(valid["answer"] > 0) & (valid["predicted_value_cm"] > 0)]

        ax.scatter(valid["answer"], valid["predicted_value_cm"],
                   alpha=0.4, s=18, edgecolors="none")
        lo = min(valid["answer"].min(), valid["predicted_value_cm"].min()) * 0.5
        hi = max(valid["answer"].max(), valid["predicted_value_cm"].max()) * 2
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect")
        ax.plot([lo, hi], [lo * 1.2, hi * 1.2], "r:", linewidth=0.7, alpha=0.5, label="+20%")
        ax.plot([lo, hi], [lo * 0.8, hi * 0.8], "r:", linewidth=0.7, alpha=0.5, label="-20%")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Ground Truth (cm)")
        ax.set_ylabel("Predicted (cm)")
        ax.set_title(display_name(model), fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3, which="both")

    for idx in range(len(datasets), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Predicted vs Ground Truth (log-log)", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved scatter plot → {save_path}")


def plot_cumulative_accuracy(datasets: dict[str, pd.DataFrame], save_path: Path):
    """Cumulative accuracy curve: fraction of predictions within X% error."""
    fig, ax = plt.subplots(figsize=(10, 6))
    thresholds_range = np.linspace(0, 1.0, 200)  # 0% to 100%

    n = len(datasets)
    cmap = plt.colormaps["tab20"] if n > 10 else plt.colormaps["tab10"]
    for idx, (model, df) in enumerate(datasets.items()):
        valid = df.dropna(subset=["rel_error"])
        accs = [(valid["rel_error"] <= t).mean() for t in thresholds_range]
        ax.plot(thresholds_range * 100, np.array(accs) * 100,
                label=display_name(model),
                linewidth=2, color=cmap(idx / max(n - 1, 1)))

    for t in THRESHOLDS:
        ax.axvline(t * 100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Relative Error Threshold (%)", fontsize=12)
    ax.set_ylabel("Cumulative Accuracy (%)", fontsize=12)
    ax.set_title("Cumulative Accuracy vs Error Threshold", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved cumulative accuracy curve → {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading predictions …")
    raw = load_all_predictions(RESULTS_DIR)
    datasets = {m: add_error_columns(df) for m, df in raw.items()}

    # ── Accuracy table ────────────────────────────────────────────────
    acc_table = compute_accuracy_table(datasets, THRESHOLDS)
    print("\n" + "=" * 90)
    print("ACCURACY REPORT")
    print("=" * 90)
    print(acc_table.to_string(index=False, float_format="%.3f"))

    # ── Outlier analysis ──────────────────────────────────────────────
    outlier_report(datasets, OUTLIER_MULTIPLIER)

    # ── Plots ─────────────────────────────────────────────────────────
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_accuracy_bars(acc_table, plots_dir / "accuracy_bars.png")
    plot_error_distribution(datasets, plots_dir / "error_distribution.png")
    plot_scatter_pred_vs_gt(datasets, plots_dir / "scatter_pred_vs_gt.png")
    plot_cumulative_accuracy(datasets, plots_dir / "cumulative_accuracy.png")

    # ── Save accuracy table to CSV ────────────────────────────────────
    csv_out = RESULTS_DIR / "accuracy_summary.csv"
    acc_table.to_csv(csv_out, index=False)
    print(f"\nSaved accuracy summary → {csv_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
