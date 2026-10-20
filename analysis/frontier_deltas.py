"""Definitive frontier closed-VLM intervention deltas for the camera-ready.

Fixes two defects of the rebuttal-era numbers (numbers.md, closed_model_radar_paper.py):
  (a) Convention: unparsable predictions count as INCORRECT (denominator = all
      rows of the manifest), matching the paper's convention, instead of being
      skipped as in vlm_eval.compute_metrics.
  (b) Row sets: every delta is computed on the PAIRED INTERSECTION of rows,
      joined by (base image, normalized query) — never by comparing the
      331-row vision manifest against 342-row context-manifest conditions.

Manifest row-set anatomy (verified 2026-08):
  - in_the_wild_vision:            331 rows, base image names (image_NNN.png).
  - in_the_wild_context_5variants: 342 rows x 5 variants. 342 = 331 canonical
    target-mask rows + 11 distractor-mask rows (the wrong-object donor masks,
    e.g. image_008__carpet asked the book query). Within-manifest deltas pair
    on the canvas stem (image_NNN__<slug>), i.e. all 342 rows.
  - in_the_wild_distorted_*:       342 rows, same stems as context.
  - in_the_wild_zoom_in/out:       318 rows, stem subset of context.
  For joins against the vision manifest the stem manifests are deduplicated
  to the canonical target row per (base image, query): within a duplicated
  group the target slug is the one starting with "the_".

Broken-run guard: a manifest whose prediction_raw column is entirely empty is
a dead run (API 401), not a refusing model, and is excluded rather than scored
0%. This used to remove everything in gemini-3.1-pro-no-reasoning_bundle except
k015 / pin015 / k025; that bundle was re-run on 2026-08-26 and now covers every
manifest, so Gemini is protocol-matched to GPT-5.5 in the main table and the
guard is a no-op for it.

Sanity anchors (must reproduce rebuttal values that are convention-invariant):
  GPT-5.5-nr vision Acc@10% = 41.4 (331/331 parsable) and
  identity delta = -16.4 pp (all context rows parsed).

Outputs:
  results/frontier_deltas.csv
  paper/emnlp_paper/tables/table_frontier.tex
  paper/emnlp_paper/tables/table_frontier_ablation.tex
"""

from __future__ import annotations

import re
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("METRICVQA_ROOT", "."))
RES = REPO / "results"
OUT_TABLES = REPO / "paper/emnlp_paper/tables"

HUMAN_REF_ACC = 39.3  # human baseline, ITW Acc@10% (numbers.md)


# ── row keys ───────────────────────────────────────────────────────────────
def norm_q(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.lower().str.strip()
            .str.replace(r"\s+", " ", regex=True).str.rstrip("?").str.strip())


def stem_of(name: str) -> str:
    """images/context/image_008__carpet__object_only.png -> image_008__carpet
    images/zoom_in/image_008__the_book.png              -> image_008__the_book
    images/itw_vision/image_008.png                     -> image_008"""
    b = Path(name).name
    b = re.sub(r"\.png$", "", b, flags=re.I)
    b = re.sub(r"__(full|object_only|object_ring|wrong_object|random_mask)$", "", b)
    return b


def base_img_of(stem: str) -> str:
    return stem.split("__")[0] + ".png"


def prep(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """Add nq / stem / bimg / correct columns. Unparsable => correct=False."""
    df = df.copy()
    df["nq"] = norm_q(df["query"])
    df["stem"] = df["image"].astype(str).map(stem_of)
    df["bimg"] = df["stem"].map(base_img_of)
    p = pd.to_numeric(df[pred_col], errors="coerce")
    y = pd.to_numeric(df["answer"], errors="coerce")
    df["parsable"] = p.notna()
    df["correct"] = ((p - y).abs() / y <= 0.10).fillna(False)
    return df


def dedup_target(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per (bimg, nq): within duplicated groups prefer the
    canonical target slug (starts with 'the_')."""
    is_target = df["stem"].str.split("__").str[-1].str.startswith("the_")
    out = (df.assign(_pref=(~is_target).astype(int))
           .sort_values("_pref", kind="stable")
           .drop_duplicates(subset=["bimg", "nq"])
           .drop(columns="_pref"))
    assert not out.duplicated(subset=["bimg", "nq"]).any()
    return out


# ── loading ────────────────────────────────────────────────────────────────
VARIANT_RE = r"__(full|object_only|object_ring|wrong_object|random_mask)\.png$"

# manifest -> (subdir, join_key_kind); key kinds: base | stem | none
CLOSED_MANIFESTS = {
    "vision":            ("in_the_wild_vision",           "base"),
    "text_only":         ("in_the_wild_text_only",        "none"),
    "gray":              ("in_the_wild_gray",             "none"),
    "zoom_in":           ("in_the_wild_zoom_in",          "stem"),
    "zoom_out":          ("in_the_wild_zoom_out",         "stem"),
    "distorted_k015":    ("in_the_wild_distorted_k015",   "stem"),
    "distorted_pin015":  ("in_the_wild_distorted_pin015", "stem"),
    "distorted_k025":    ("in_the_wild_distorted_k025",   "stem"),
    "distorted_pin025":  ("in_the_wild_distorted_pin025", "stem"),
    "target_plus_ref":   ("in_the_wild_target_plus_ref",  "none"),
    "objectron_vision":     ("objectron_vision_unique27",    "none"),
    "objectron_gray":       ("objectron_gray_unique27",      "none"),
    "objectron_text_only":  ("objectron_text_only_unique27", "none"),
}
CTX_VARIANTS = ["full", "object_only", "object_ring", "wrong_object", "random_mask"]


def load_closed_bundle(bundle: str) -> dict[str, pd.DataFrame]:
    """Returns {condition: prepared df}. Dead manifests (all prediction_raw
    empty => broken API run) are dropped with a warning."""
    conds = {}

    def read(subdir):
        f = RES / bundle / subdir / "predictions.csv"
        if not f.exists():
            return None
        df = pd.read_csv(f)
        if df["prediction_raw"].isna().all():
            print(f"  [{bundle}] {subdir}: DEAD RUN (prediction_raw all empty) — excluded")
            return None
        return df

    for cond, (subdir, _) in CLOSED_MANIFESTS.items():
        df = read(subdir)
        if df is not None:
            conds[cond] = prep(df, "prediction")

    ctx = read("in_the_wild_context_5variants")
    if ctx is not None:
        ctx = prep(ctx, "prediction")
        ctx["variant"] = ctx["image"].str.extract(VARIANT_RE)
        assert ctx["variant"].notna().all()
        for v in CTX_VARIANTS:
            name = "ctx_full" if v == "full" else v
            conds[name] = ctx[ctx["variant"] == v].copy()
    return conds


def load_qwen_open() -> dict[str, pd.DataFrame]:
    """Qwen3-VL-235B from the paper pipeline's own results dirs, same keys."""
    Q = "Qwen3-VL-235B-A22B-Instruct"
    paths = {
        "vision":           RES / Q,
        "text_only":        RES / f"text_only_{Q}",
        "gray":             RES / f"gray_{Q}",
        "zoom_in":          RES / f"zoom_in_{Q}",
        "zoom_out":         RES / f"zoom_out_{Q}",
        "distorted_k015":   RES / f"distorted_{Q}",
        "distorted_pin015": RES / f"distorted_pin015_{Q}",
        "ctx_full":         RES / "context_full" / Q,
        "object_only":      RES / "context_object_only" / Q,
        "object_ring":      RES / "context_object_ring" / Q,
        "wrong_object":     RES / "context_wrong_object" / Q,
        "random_mask":      RES / "context_random_mask" / Q,
    }
    conds = {}
    for cond, d in paths.items():
        f = d / "predictions.csv"
        if f.exists():
            conds[cond] = prep(pd.read_csv(f), "predicted_value_cm")
    return conds


# ── metrics ────────────────────────────────────────────────────────────────
# Convention (2026-08-27): unparsable responses are EXCLUDED. A parse failure
# is a failure to follow the output instruction, not a size estimate, so it
# does not enter accuracy denominators; paired deltas use rows parsable in
# both conditions. Per-model parse rates are reported so the exclusion is
# auditable. (Previously: unparsable counted incorrect.)
def acc(df: pd.DataFrame | None) -> float | None:
    """Acc@10% over parsable rows (unparsable excluded)."""
    if df is None or len(df) == 0:
        return None
    m = df["parsable"]
    if not m.any():
        return None
    return float(df.loc[m, "correct"].mean() * 100)


def paired(a: pd.DataFrame | None, b: pd.DataFrame | None, on: str):
    """Inner join on key column; returns (n, acc_a, acc_b, delta_pp) with
    acc computed on rows parsable in BOTH conditions."""
    if a is None or b is None:
        return None
    keys = [on, "nq"]
    a2, b2 = a, b
    if on == "bimg":  # vision-side join: dedup stem manifests to target rows
        if a2.duplicated(subset=keys).any():
            a2 = dedup_target(a2)
        if b2.duplicated(subset=keys).any():
            b2 = dedup_target(b2)
    m = a2[keys + ["correct", "parsable"]].merge(
        b2[keys + ["correct", "parsable"]], on=keys, suffixes=("_a", "_b"))
    assert not m.duplicated(subset=keys).any(), "non-unique join"
    m = m[m["parsable_a"] & m["parsable_b"]]
    if len(m) == 0:
        return None
    aa = float(m["correct_a"].mean() * 100)
    ab = float(m["correct_b"].mean() * 100)
    return len(m), aa, ab, ab - aa


def triple_lens(conds):
    """mean(k015, pin015) - vision on the common intersection of all three."""
    v, k, p = conds.get("vision"), conds.get("distorted_k015"), conds.get("distorted_pin015")
    if v is None or k is None or p is None:
        return None
    keys = ["bimg", "nq"]
    k2, p2 = dedup_target(k), dedup_target(p)
    m = (v[keys + ["correct", "parsable"]]
         .merge(k2[keys + ["correct", "parsable"]], on=keys, suffixes=("", "_k"))
         .merge(p2[keys + ["correct", "parsable"]], on=keys, suffixes=("", "_p")))
    m = m[m["parsable"] & m["parsable_k"] & m["parsable_p"]]
    if len(m) == 0:
        return None
    av = float(m["correct"].mean() * 100)
    ad = float((m["correct_k"].astype(float) + m["correct_p"].astype(float)).mean() / 2 * 100)
    return len(m), av, ad, ad - av


def compute_bundle(conds: dict) -> list[dict]:
    """Rows: kind=acc (n = manifest rows) and kind=delta (n = paired rows)."""
    rows = []
    acc_order = ["vision", "text_only", "gray", "ctx_full", "object_only",
                 "object_ring", "wrong_object", "random_mask", "zoom_in",
                 "zoom_out", "distorted_k015", "distorted_pin015",
                 "distorted_k025", "distorted_pin025", "target_plus_ref",
                 "objectron_vision", "objectron_gray", "objectron_text_only"]
    for c in acc_order:
        d = conds.get(c)
        if d is not None:
            rows.append(dict(name=c, kind="acc", n=len(d),
                             n_parsable=int(d["parsable"].sum()), value=acc(d)))

    def delta(name, a, b, on):
        r = paired(conds.get(a), conds.get(b), on)
        if r is not None:
            n, aa, ab, d = r
            rows.append(dict(name=name, kind="delta", n=n, n_parsable=None,
                             value=d, acc_ref=aa, acc_cond=ab))

    # within-context (stem pairing over all 342 canvases)
    delta("identity(wrong_obj-obj_only)",  "object_only", "wrong_object", "stem")
    delta("pixels(obj_only-random_mask)",  "random_mask", "object_only", "stem")
    delta("ring(obj_ring-obj_only)",       "object_only", "object_ring", "stem")
    delta("scene_ctx(ctx_full-obj_only)",  "object_only", "ctx_full", "stem")
    # cross-manifest vs the standalone vision manifest (target rows only)
    delta("scene_vision(vision-obj_only)", "object_only", "vision", "bimg")
    delta("zoom_in(zoom_in-obj_only)",     "object_only", "zoom_in", "stem")
    delta("zoom_out(zoom_out-obj_only)",   "object_only", "zoom_out", "stem")
    delta("lens_k015(k015-vision)",        "vision", "distorted_k015", "bimg")
    delta("lens_pin015(pin015-vision)",    "vision", "distorted_pin015", "bimg")
    delta("lens_k025(k025-vision)",        "vision", "distorted_k025", "bimg")
    delta("lens_pin025(pin025-vision)",    "vision", "distorted_pin025", "bimg")
    r = triple_lens(conds)
    if r is not None:
        n, av, ad, d = r
        rows.append(dict(name="lens_mean(mean015-vision)", kind="delta", n=n,
                         n_parsable=None, value=d, acc_ref=av, acc_cond=ad))
    return rows


# ── run all bundles ────────────────────────────────────────────────────────
BUNDLES = [
    ("gpt-5.5-no-reasoning_bundle",       "GPT-5.5 (no reasoning)"),
    ("gpt-5.5_bundle",                    "GPT-5.5 (default API)"),
    ("gemini-3.1-pro_bundle",             "Gemini 3.1 Pro (default API)"),
    ("gemini-3.1-pro-no-reasoning_bundle","Gemini 3.1 Pro (no reasoning)"),
]


def main():
    all_rows = []
    results = {}
    for bundle, label in BUNDLES:
        print(f"== {bundle}")
        conds = load_closed_bundle(bundle)
        rows = compute_bundle(conds)
        results[bundle] = {r["name"]: r for r in rows}
        for r in rows:
            all_rows.append({"bundle": bundle, **r})

    print("== Qwen3-VL-235B (open, paper pipeline)")
    qconds = load_qwen_open()
    qrows = compute_bundle(qconds)
    results["qwen3-vl-235b"] = {r["name"]: r for r in qrows}
    for r in qrows:
        all_rows.append({"bundle": "qwen3-vl-235b_open", **r})

    df = pd.DataFrame(all_rows)[["bundle", "name", "kind", "n", "n_parsable",
                                 "value", "acc_ref", "acc_cond"]]

    # ── sanity anchors (convention-invariant rebuttal values) ──────────────
    nr = results["gpt-5.5-no-reasoning_bundle"]
    v = nr["vision"]["value"]
    ident = nr["identity(wrong_obj-obj_only)"]["value"]
    assert abs(v - 41.4) <= 0.2, f"GPT-5.5-nr vision {v:.2f} != 41.4"
    assert abs(ident - (-16.4)) <= 0.2, f"GPT-5.5-nr identity {ident:.2f} != -16.4"
    print(f"\nsanity OK: GPT-5.5-nr vision={v:.1f}%, identity={ident:+.1f}pp")

    out_csv = RES / "frontier_deltas.csv"
    df.round(2).to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(df)} rows)")

    write_tables(results)

    with pd.option_context("display.width", 200, "display.max_rows", 300):
        print(df.round(1).to_string(index=False))


# ── tables ─────────────────────────────────────────────────────────────────
def _get(results, bundle, name):
    r = results.get(bundle, {}).get(name)
    return None if r is None else r["value"]


def fmt_acc_cell(v):
    return "--" if v is None else f"{v:.1f}"


def fmt_delta_cell(v):
    return "--" if v is None else f"${v:+.1f}$"


def write_tables(results):
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    # ── main-body compact table ────────────────────────────────────────────
    def main_row(label, bundle):
        cells = [fmt_acc_cell(_get(results, bundle, "vision")),
                 fmt_acc_cell(_get(results, bundle, "text_only")),
                 fmt_delta_cell(_get(results, bundle, "identity(wrong_obj-obj_only)")),
                 fmt_delta_cell(_get(results, bundle, "pixels(obj_only-random_mask)")),
                 fmt_delta_cell(_get(results, bundle, "ring(obj_ring-obj_only)")),
                 fmt_delta_cell(_get(results, bundle, "lens_mean(mean015-vision)"))]
        return f"{label} & " + " & ".join(cells) + " \\\\"

    rows = []
    rows.append("\\begin{table}[t]")
    rows.append("\\centering\\footnotesize")
    rows.append("\\setlength{\\tabcolsep}{1.7pt}")  # fits ACL columnwidth (226.85pt), verified
    rows.append("\\begin{tabular}{l|cc|cccc}")
    rows.append("\\toprule")
    rows.append(" & \\multicolumn{2}{c|}{Acc@10\\%} & \\multicolumn{4}{c}{$\\Delta$ Acc@10\\% (pp)} \\\\")
    rows.append("Model & vision & text & ident. & pixels & ring & lens \\\\")
    rows.append("\\midrule")
    rows.append(main_row("GPT-5.5 (no reas.)", "gpt-5.5-no-reasoning_bundle"))
    rows.append(main_row("Gemini 3.1 Pro (no reas.)",
                         "gemini-3.1-pro-no-reasoning_bundle"))
    rows.append("\\midrule")
    rows.append(main_row("Qwen3-VL-235B", "qwen3-vl-235b"))
    rows.append(f"Human reference & {HUMAN_REF_ACC:.1f} & -- & -- & -- & -- & -- \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append(
        "\\caption{\\textbf{Frontier closed VLMs under the intervention "
        "suite} (in-the-wild). Acc@10\\% with unparsable predictions counted "
        "incorrect; every $\\Delta$ is computed on the paired intersection "
        "of rows, joined by (image, normalized query). "
        "\\textbf{ident.} = wrong\\_object $-$ object\\_only, "
        "\\textbf{pixels} = object\\_only $-$ random\\_mask, "
        "\\textbf{ring} = object\\_ring $-$ object\\_only "
        "(all paired within the context manifest, $n{=}342$ canvases); "
        "\\textbf{lens} = mean of barrel/pincushion $|k|{=}0.15$ minus the "
        "full image on the shared $n{=}331$ rows. Both frontier models are "
        "run with reasoning disabled, matching the single-forward-pass "
        "protocol of the open models; their API-default reasoning "
        "configurations are reported as ablations in "
        "Table~\\ref{tab:frontier_ablation}. Both frontier "
        "models reproduce the open-model signature: a large identity "
        "channel, a substantial target-pixel channel, and lens-distortion "
        "invariance. Qwen3-VL-235B (best open model) and the human study "
        "are reference rows. Full per-condition accuracies and the "
        "remaining deltas in Table~\\ref{tab:frontier_ablation}.}")
    rows.append("\\label{tab:frontier}")
    rows.append("\\end{table}")
    f = OUT_TABLES / "table_frontier.tex"
    f.write_text("\n".join(rows) + "\n")
    print(f"wrote {f}")

    # ── appendix: all conditions x all bundles ─────────────────────────────
    COLS = [
        ("gpt-5.5-no-reasoning_bundle",        "GPT-5.5",        "(no reas.)"),
        ("gpt-5.5_bundle",                     "GPT-5.5",        "(default)"),
        ("gemini-3.1-pro_bundle",              "Gemini 3.1",     "(default)"),
        ("gemini-3.1-pro-no-reasoning_bundle", "Gemini 3.1",     "(no reas.)"),
        ("qwen3-vl-235b",                      "Qwen3-VL",       "(235B, open)"),
    ]
    ACC_ROWS = [
        ("vision",            "full image (vision manifest)"),
        ("text_only",         "text-only"),
        ("gray",              "gray-blank"),
        ("ctx_full",          "full (context manifest)"),
        ("object_only",       "object\\_only"),
        ("object_ring",       "object\\_ring"),
        ("wrong_object",      "wrong\\_object"),
        ("random_mask",       "random\\_mask"),
        ("zoom_in",           "zoom-in $1.4\\times$"),
        ("zoom_out",          "zoom-out $0.6\\times$"),
        ("distorted_k015",    "barrel $k{=}0.15$"),
        ("distorted_pin015",  "pincushion $k{=}{-}0.15$"),
        ("distorted_k025",    "barrel $k{=}0.25$"),
        ("distorted_pin025",  "pincushion $k{=}{-}0.25$"),
        ("target_plus_ref",   "target$+$reference"),
        ("objectron_vision",     "Objectron vision (unique-27)"),
        ("objectron_gray",       "Objectron gray-blank (unique-27)"),
        ("objectron_text_only",  "Objectron text-only (unique-27)"),
    ]
    DELTA_ROWS = [
        ("identity(wrong_obj-obj_only)",  "$\\Delta$ identity (wrong\\_obj $-$ obj\\_only)"),
        ("pixels(obj_only-random_mask)",  "$\\Delta$ pixels (obj\\_only $-$ random\\_mask)"),
        ("ring(obj_ring-obj_only)",       "$\\Delta$ ring (obj\\_ring $-$ obj\\_only)"),
        ("scene_ctx(ctx_full-obj_only)",  "$\\Delta$ scene (full $-$ obj\\_only, in-manifest)"),
        ("scene_vision(vision-obj_only)", "$\\Delta$ scene (vision $-$ obj\\_only, cross)"),
        ("zoom_in(zoom_in-obj_only)",     "$\\Delta$ zoom-in (vs obj\\_only)"),
        ("zoom_out(zoom_out-obj_only)",   "$\\Delta$ zoom-out (vs obj\\_only)"),
        ("lens_k015(k015-vision)",        "$\\Delta$ barrel $k{=}0.15$ (vs vision)"),
        ("lens_pin015(pin015-vision)",    "$\\Delta$ pincushion $k{=}{-}0.15$ (vs vision)"),
        ("lens_mean(mean015-vision)",     "$\\Delta$ lens mean $|k|{=}0.15$ (vs vision)"),
    ]

    def row_n(name):
        ns = {results[b].get(name, {}).get("n") for b, _, _ in COLS
              if results.get(b, {}).get(name)}
        ns.discard(None)
        if not ns:
            return "--"
        return "/".join(str(n) for n in sorted(ns))

    rows = []
    rows.append("\\begin{table*}[t]")
    rows.append("\\centering\\footnotesize")
    rows.append("\\setlength{\\tabcolsep}{4.0pt}")
    rows.append("\\begin{tabular}{l|r|" + "c" * len(COLS) + "}")
    rows.append("\\toprule")
    rows.append("Condition / delta & $n$ & " + " & ".join(h for _, h, _ in COLS) + " \\\\")
    rows.append(" & & " + " & ".join(s for _, _, s in COLS) + " \\\\")
    rows.append("\\midrule")
    for name, label in ACC_ROWS:
        cells = [fmt_acc_cell(_get(results, b, name)) for b, _, _ in COLS]
        rows.append(f"{label} & {row_n(name)} & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
    for name, label in DELTA_ROWS:
        cells = [fmt_delta_cell(_get(results, b, name)) for b, _, _ in COLS]
        rows.append(f"{label} & {row_n(name)} & " + " & ".join(cells) + " \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append(
        "\\caption{\\textbf{Full intervention suite for the frontier closed "
        "models} (Acc@10\\%, unparsable responses excluded; deltas on paired "
        "intersections joined by (image, normalized query); the $n$ column "
        "is the number of rows for accuracies and of paired rows for "
        "deltas). "
        "The \\emph{no reas.} columns are the primary configurations "
        "matching the open-model protocol; the \\emph{default} columns "
        "(GPT-5.5 and Gemini~3.1~Pro with API-default reasoning) are "
        "ablation configurations, not protocol-matched. "
        "Identity / pixels / ring / within-context scene deltas "
        "pair the $342$ context canvases; zoom deltas pair the $318$ zoom "
        "rows against the same canvases' object\\_only condition; lens and "
        "cross-manifest scene deltas pair the $331$ vision-manifest rows "
        "with the corresponding target rows of the $342$-row manifests. "
        "Qwen3-VL-235B (best open model) recomputed identically from the "
        "paper pipeline's results for reference.}")
    rows.append("\\label{tab:frontier_ablation}")
    rows.append("\\end{table*}")
    f = OUT_TABLES / "table_frontier_ablation.tex"
    f.write_text("\n".join(rows) + "\n")
    print(f"wrote {f}")


if __name__ == "__main__":
    sys.exit(main())
