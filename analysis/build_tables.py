"""
Build TeX tables for the EMNLP paper from the current results dir.

One table per storyline step:
  table_main_results.tex   — Step 1: vision + LLM/Human/235B baselines
  table_language_prior.tex — Step 2: gray-blank itw + obj
  table_context.tex        — Step 3+5: context-variant ablation
  table_zoom.tex           — Step 4: apparent-size invariance
  table_target_plus_ref.tex — Step 5b: reference-object use (when available)
  table_distortion.tex     — Step 4b: vanishing-point / geometry probe
  table_v3_typos.tex       — small follow-up

Outputs:
  paper/emnlp_paper/tables/<table>.tex
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("METRICVQA_ROOT", "."))
RES = REPO / "results"
OUT = REPO / "paper/emnlp_paper/tables"
OUT.mkdir(parents=True, exist_ok=True)


MIN_VALID = 10  # treat conditions with fewer than this many parseable answers as effective refusal (returns None)


def metrics_from(d: Path) -> dict | None:
    p = d / "predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "predicted_value_cm" not in df.columns:
        return None
    # target_plus_ref uses target_answer; everything else uses answer
    if "answer" not in df.columns and "target_answer" in df.columns:
        df = df.rename(columns={"target_answer": "answer"})
    if "answer" not in df.columns:
        return None
    v = df[df["predicted_value_cm"].notna() & df["answer"].notna()].copy()
    if len(v) < MIN_VALID:
        return None
    # discard absurd parses (> 1e4 cm); they're the granite/SmolVLM tail bug
    v = v[v["predicted_value_cm"].abs() < 1e4]
    if len(v) < MIN_VALID:
        return None
    e = (v["predicted_value_cm"] - v["answer"]).abs()
    rel = e / v["answer"].abs().clip(lower=1e-6)
    return {
        "mae": float(e.mean()),
        "med": float(e.median()),
        "acc10": float((rel <= 0.10).mean()),
        "acc15": float((rel <= 0.15).mean()),
        "acc20": float((rel <= 0.20).mean()),
        "n_valid": int(len(v)),
        "n_total": int(len(df)),
    }


def fmt_mae(r):
    if r is None:
        return "--"
    return f"{r['mae']:.2f}"


def fmt_acc(r, key="acc10"):
    if r is None:
        return "--"
    return f"{r[key]*100:.1f}\\%"


def fmt_pair(r):
    if r is None:
        return "--"
    return f"{r['mae']:.2f} / {r['acc10']*100:.1f}\\%"


def paired_bootstrap_acc_delta(rel_a, rel_b, B=2000, thr=0.10, seed=0):
    """Paired bootstrap 95% CI on Acc@thr(b) - Acc@thr(a) in percentage points.

    rel_a, rel_b: 1-D arrays of relative errors aligned by row index.
    Returns (delta_pp, lo_pp, hi_pp). Resamples paired row indices with
    replacement B times; the binary correctness per row is fixed by the
    threshold thr."""
    rel_a = np.asarray(rel_a, dtype=float)
    rel_b = np.asarray(rel_b, dtype=float)
    assert rel_a.shape == rel_b.shape
    n = len(rel_a)
    if n == 0:
        return (0.0, 0.0, 0.0)
    a_hit = (rel_a <= thr).astype(float)
    b_hit = (rel_b <= thr).astype(float)
    diff = (b_hit - a_hit) * 100.0  # per-row paired diff in pp
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, n, size=(B, n))
    boot = diff[samples].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def paired_bootstrap_acc(rel, B=2000, thr=0.10, seed=0):
    """Bootstrap 95% CI on Acc@thr for a single condition, in percent."""
    rel = np.asarray(rel, dtype=float)
    n = len(rel)
    if n == 0:
        return (0.0, 0.0, 0.0)
    hit = (rel <= thr).astype(float) * 100.0
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, n, size=(B, n))
    boot = hit[samples].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(hit.mean()), float(lo), float(hi)


# 12-model canonical pool spanning 3B-397B open weights, lighter to heavier.
# Replaced Idefics3 / Phi-3.5 / LLaVA-v1.6 with the three large open-weight
# models (Shaked-run on RITS).
MODELS = [
    ("Qwen2.5-VL-3B-Instruct",     "Qwen2.5-VL-3B"),
    ("LLaVA-v1.6-Mistral-7B",      "LLaVA-v1.6"),
    ("SmolVLM-Instruct",           "SmolVLM"),
    ("Gemma3-4B-IT",               "Gemma3-4B"),
    ("granite-vision-3.3-2b",      "granite-3.3-2b"),
    ("Qwen2.5-VL-7B-Instruct",     "Qwen2.5-VL-7B"),
    ("LLaVA-OneVision-7B",         "LLaVA-OV"),
    ("Molmo-7B-D",                 "Molmo-7B-D"),
    ("InternVL2-8B",               "InternVL2"),
    ("InternVL3.5",                "InternVL3.5-241B"),
    ("Qwen3-VL-235B-A22B-Instruct","Qwen3-VL-235B"),
    ("Qwen3.5-397B",               "Qwen3.5-397B"),
]


def vision_itw(slug):
    return metrics_from(RES / slug)


def vision_obj(slug):
    # in the repo, Objectron vision dirs sometimes drop suffixes; try several
    candidates = [
        RES / f"objectron_{slug}",
        RES / f"objectron_{slug.replace('-Instruct','')}",
        RES / f"objectron_{slug.split('-Llama3')[0]}",
    ]
    for c in candidates:
        m = metrics_from(c)
        if m:
            return m
    return None


def gray_itw(slug):
    return metrics_from(RES / f"gray_{slug}")


def gray_obj(slug):
    return metrics_from(RES / f"objectron_gray_{slug}")


def zoom_in(slug):
    return metrics_from(RES / f"zoom_in_{slug}")


def zoom_out(slug):
    return metrics_from(RES / f"zoom_out_{slug}")


def text_only_itw(slug):
    return metrics_from(RES / f"text_only_{slug}")


def text_only_obj(slug):
    """Broadcasted 10,482-row text-only (preferred); falls back to 27-row source."""
    bc = metrics_from(RES / f"objectron_text_only_{slug}")
    if bc is not None:
        return bc
    return metrics_from(RES / f"objectron_text_only_{slug}_27")


CONTEXT_NAME_MAP = {
    "Qwen2.5-VL-3B-Instruct":    "Qwen2.5-VL-3B-Instruct",
    "Qwen2.5-VL-7B-Instruct":    "Qwen2.5-VL-7B-Instruct",
    "PaliGemma-3B-Mix-448":      "paligemma-3b-mix-448",
    "SmolVLM-Instruct":          "SmolVLM-Instruct",
    "Gemma3-4B-IT":              "gemma-3-4b-it",
    "granite-vision-3.3-2b":     "granite-vision-3.3-2b",
    "LLaVA-v1.6-Mistral-7B":     "llava-v1.6-mistral-7b-hf",
    "LLaVA-OneVision-7B":        "llava-onevision-qwen2-7b-ov-hf",
    "Molmo-7B-D":                "Molmo-7B-D-0924",
    "Phi-3.5-Vision":            "Phi-3.5-vision-instruct",
    "InternVL2-8B":              "InternVL2-8B",
    "Idefics3-8B-Llama3":        "Idefics3-8B-Llama3",
}


def context_variant(slug, variant):
    """variant in {full, object_only, object_ring, wrong_object, random_mask}."""
    ctx_short = CONTEXT_NAME_MAP.get(slug, slug)
    return metrics_from(RES / f"context_{variant}" / ctx_short)


def context_variant_rel(slug, variant):
    """Return a DataFrame [_k, rel] of paired keys and relative errors for
    this (model, variant), suitable for merge-on-_k with another variant."""
    ctx_short = CONTEXT_NAME_MAP.get(slug, slug)
    p = RES / f"context_{variant}" / ctx_short / "predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "answer" not in df.columns and "target_answer" in df.columns:
        df = df.rename(columns={"target_answer": "answer"})
    for c in ("answer", "predicted_value_cm"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["predicted_value_cm", "answer"])
    df = df[df["predicted_value_cm"].abs() < 1e4]
    if len(df) == 0:
        return None
    df["_k"] = (df["query"].astype(str).str.strip().str.lower()
                + "||" + df["answer"].astype(str))
    df["_rel"] = ((df["predicted_value_cm"] - df["answer"]).abs()
                  / df["answer"].abs().clip(lower=1e-6))
    return df[["_k", "_rel"]]


def paired_context_delta(slug, var_a, var_b):
    """Returns (n_paired, delta_pp, lo, hi) for Acc@10%(var_b) - Acc@10%(var_a)
    paired on (query, answer); or None if data unavailable."""
    import hashlib
    a = context_variant_rel(slug, var_a)
    b = context_variant_rel(slug, var_b)
    if a is None or b is None:
        return None
    m = a.merge(b, on="_k", suffixes=("_a", "_b"))
    if len(m) < 10:
        return None
    seed = int.from_bytes(
        hashlib.md5(f"{slug}|{var_a}|{var_b}".encode()).digest()[:4], "big")
    d_mean, d_lo, d_hi = paired_bootstrap_acc_delta(
        m["_rel_a"].values, m["_rel_b"].values, seed=seed)
    return len(m), d_mean, d_lo, d_hi


def distorted(slug, tag="distorted"):
    return metrics_from(RES / f"{tag}_{slug}")


def baseline(name):
    if name == "llm_itw":
        # ChatGPT 5.5 on in-the-wild
        gt = pd.read_csv(REPO / "questions_v2.csv")
        df = pd.read_csv(REPO / "in-the-wild_chatgpt5.5.csv",
                         header=0, names=["image", "query", "pred"])
        m = df.merge(gt, on=["image", "query"])
        e = (m["pred"] - m["answer"]).abs()
        rel = e / m["answer"].abs()
        return {"mae": float(e.mean()), "med": float(e.median()),
                "acc10": float((rel <= 0.10).mean()),
                "acc15": float((rel <= 0.15).mean()),
                "acc20": float((rel <= 0.20).mean()),
                "n_valid": len(m), "n_total": 331}
    if name == "llm_obj":
        return metrics_from(RES / "objectron_chatgpt5.5_prior")
    if name == "human_itw":
        return metrics_from(RES / "human_baseline")
    if name == "qwen235b_itw":
        return metrics_from(RES / "Qwen3-VL-235B-A22B-Instruct")
    if name == "qwen235b_obj":
        return metrics_from(RES / "objectron_Qwen3-VL-235B-A22B-Instruct")
    return None


def metrics_incorrect_conv(d: Path) -> dict | None:
    """Acc@10% with unparsable scored as incorrect (denominator = all rows).
    Convention for the text-only LLM baselines and other post-rebuttal
    analyses (see paper §3.1); legacy open-pool tables use metrics_from."""
    p = d / "predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "predicted_value_cm" not in df.columns or "answer" not in df.columns:
        return None
    n_total = len(df)
    v = df[df["predicted_value_cm"].notna() & df["answer"].notna()].copy()
    v = v[v["predicted_value_cm"].abs() < 1e4]
    if n_total == 0 or len(v) < MIN_VALID:
        return None
    e = (v["predicted_value_cm"] - v["answer"]).abs()
    rel = e / v["answer"].abs().clip(lower=1e-6)
    return {
        "mae": float(e.mean()),
        "med": float(e.median()),
        "acc10": float((rel <= 0.10).sum() / n_total),
        "acc15": float((rel <= 0.15).sum() / n_total),
        "acc20": float((rel <= 0.20).sum() / n_total),
        "n_valid": len(v), "n_total": n_total,
    }


OPEN_LLM_BASELINES = [
    ("gpt-oss-120b",      "gpt-oss-120b LLM (no image)"),
    ("deepseek-v4-flash", "DeepSeek-v4-flash LLM (no image)"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 — Main results (Step 1)
# ─────────────────────────────────────────────────────────────────────────────
def table_main_results():
    """Single-column main results: Acc@10% only on both splits. MAE
    counterpart lives in the supplementary (table_main_results_mae)."""
    rows = []
    rows.append("\\begin{table}[t]")
    rows.append("\\centering\\small")
    rows.append("\\begin{tabular}{l|cc}")
    rows.append("\\toprule")
    rows.append("Model & ITW Acc@10\\% & Obj Acc@10\\% \\\\")
    rows.append(" & ($n{=}331$) & ($n{=}10{,}482$) \\\\")
    rows.append("\\midrule")
    rows.append("\\multicolumn{3}{l}{\\emph{Non-VLM reference baselines}} \\\\")
    h = baseline("human_itw")
    rows.append(f"Human reference & {fmt_acc(h)} & -- \\\\")
    li = baseline("llm_itw"); lo = baseline("llm_obj")
    rows.append(f"ChatGPT 5.5 (text) & {fmt_acc(li)} & {fmt_acc(lo)} \\\\")
    rows.append("\\midrule")
    open_rows = []
    for slug, label in MODELS:
        vi = vision_itw(slug); vo = vision_obj(slug)
        acc = vi["acc10"] if vi else None
        open_rows.append((label, vi, vo, acc))
    open_rows.sort(key=lambda r: (-r[3] if r[3] is not None else 1e9))
    for label, vi, vo, _ in open_rows:
        rows.append(f"{label} & {fmt_acc(vi)} & {fmt_acc(vo)} \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\caption{Acc@10\\% on both splits, sorted by in-the-wild. Non-VLM rows are reference baselines, not benchmark contenders. MAE counterpart in Appendix~\\ref{sec:appendix}, Table~\\ref{tab:main_mae}.}")
    rows.append("\\label{tab:main}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


def fmt_med(r):
    if r is None or r.get("med") is None:
        return "--"
    return f"{r['med']:.2f}"


def table_main_results_mae():
    """Supplementary MAE + median-AE counterpart to table_main_results.
    Same row order (ITW Acc@10% descending) for easy cross-reference."""
    rows = []
    rows.append("\\begin{table}[t]")
    rows.append("\\centering\\small")
    rows.append("\\begin{tabular}{l|cc|cc}")
    rows.append("\\toprule")
    rows.append(" & \\multicolumn{2}{c|}{ITW ($n{=}331$)} & \\multicolumn{2}{c}{Obj ($n{=}10{,}482$)} \\\\")
    rows.append("Model & MAE & med.\\,AE & MAE & med.\\,AE \\\\")
    rows.append("\\midrule")
    rows.append("\\multicolumn{5}{l}{\\emph{Non-VLM reference baselines}} \\\\")
    h = baseline("human_itw")
    rows.append(f"Human reference & {fmt_mae(h)} & {fmt_med(h)} & -- & -- \\\\")
    li = baseline("llm_itw"); lo = baseline("llm_obj")
    rows.append(f"ChatGPT 5.5 (text) & {fmt_mae(li)} & {fmt_med(li)} & {fmt_mae(lo)} & {fmt_med(lo)} \\\\")
    rows.append("\\midrule")
    open_rows = []
    for slug, label in MODELS:
        vi = vision_itw(slug); vo = vision_obj(slug)
        acc = vi["acc10"] if vi else None
        open_rows.append((label, vi, vo, acc))
    open_rows.sort(key=lambda r: (-r[3] if r[3] is not None else 1e9))
    for label, vi, vo, _ in open_rows:
        rows.append(f"{label} & {fmt_mae(vi)} & {fmt_med(vi)} & {fmt_mae(vo)} & {fmt_med(vo)} \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\caption{Mean and median absolute error (cm) counterpart to Table~\\ref{tab:main}, rows in the same order (in-the-wild Acc@10\\% descending). MAE over the $0.8$--$245$\\,cm ground-truth range is dominated by rare catastrophic overshoots and only loosely tracks the Acc@10\\% ordering (Spearman $\\rho=0.68$; most extreme, Qwen2.5-VL-7B is 4th by Acc@10\\% but 11th by ITW MAE, driven by a $12\\%$ scale-failure rate with median within-failure error of $91$\\,cm, Table~\\ref{tab:fse_itw}). The outlier-robust median AE tracks the Acc@10\\% ordering more closely ($\\rho=0.80$); we therefore use Acc@10\\% as the primary metric in the main text.}")
    rows.append("\\label{tab:main_mae}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Table 2 — Language-only prior (Step 2)
# ─────────────────────────────────────────────────────────────────────────────
def table_language_prior():
    rows = []
    rows.append("\\begin{table*}[t]")
    rows.append("\\centering\\small")
    rows.append("\\begin{tabular}{l|ccc|ccc}")
    rows.append("\\toprule")
    rows.append(" & \\multicolumn{3}{c|}{In-the-wild Acc@10\\%} & \\multicolumn{3}{c}{Objectron Acc@10\\%} \\\\")
    rows.append("Model & vision & gray-blank & text-only & vision & gray-blank & text-only \\\\")
    rows.append("\\midrule")
    li = baseline("llm_itw"); lo = baseline("llm_obj")
    rows.append(f"ChatGPT 5.5 LLM (no image) & -- & -- & {fmt_acc(li)} & -- & -- & {fmt_acc(lo)} \\\\")
    for llm_slug, llm_label in OPEN_LLM_BASELINES:
        oi = metrics_incorrect_conv(RES / f"text_only_{llm_slug}")
        oo = metrics_incorrect_conv(RES / f"objectron_text_only_{llm_slug}")
        rows.append(f"{llm_label} & -- & -- & {fmt_acc(oi)} & -- & -- & {fmt_acc(oo)} \\\\")
    rows.append("\\midrule")
    for slug, label in MODELS:
        vi = vision_itw(slug); gi = gray_itw(slug); ti = text_only_itw(slug)
        vo = vision_obj(slug); go = gray_obj(slug); to = text_only_obj(slug)
        rows.append(f"{label} & {fmt_acc(vi)} & {fmt_acc(gi)} & {fmt_acc(ti)} & {fmt_acc(vo)} & {fmt_acc(go)} & {fmt_acc(to)} \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\caption{Language-only prior probe on both splits. \\emph{Text-only} sends no image at all; \\emph{gray-blank} sends a uniform gray canvas (RGB 127) in the required image slot. Molmo-7B-D refuses text-only and LLaVA-v1.6 refuses gray-blank (\\textit{--} entries). For the two largest open models a gray canvas suppresses the LM prior far more than removing the image entirely (Qwen3.5-397B on Objectron: $1.2\\%$ gray vs $30.3\\%$ text-only), so we read text-only as the primary anchor and gray-blank as a conservative fallback. Neither probe closes the gap with the frontier text-only LLM. The open-weight LLM rows (unparsable scored incorrect) are fully reproducible language-prior references: below the frontier LLM, yet matching or exceeding the majority of the open VLMs that receive the full image.}")
    rows.append("\\label{tab:language_prior}")
    rows.append("\\end{table*}")
    return "\n".join(rows) + "\n"


def table_language_prior_compact():
    """Single-column main-body summary of the language-prior probes: the
    three text-only LLM references plus the strongest open-VLM text-only
    rows. Full per-model table stays in the appendix."""
    li = baseline("llm_itw"); lo = baseline("llm_obj")
    rows = []
    rows.append("\\begin{table}[t]")
    rows.append("\\centering\\small")
    rows.append("\\begin{tabular}{l|cc}")
    rows.append("\\toprule")
    rows.append(" & \\multicolumn{2}{c}{text-only Acc@10\\%} \\\\")
    rows.append("Model & ITW & Objectron \\\\")
    rows.append("\\midrule")
    rows.append("\\multicolumn{3}{l}{\\emph{Text-only LLM references}} \\\\")
    rows.append(f"ChatGPT 5.5 & {fmt_acc(li)} & {fmt_acc(lo)} \\\\")
    for llm_slug, llm_label in OPEN_LLM_BASELINES:
        oi = metrics_incorrect_conv(RES / f"text_only_{llm_slug}")
        oo = metrics_incorrect_conv(RES / f"objectron_text_only_{llm_slug}")
        label = llm_label.replace(" LLM (no image)", "")
        rows.append(f"{label} & {fmt_acc(oi)} & {fmt_acc(oo)} \\\\")
    rows.append("\\midrule")
    rows.append("\\multicolumn{3}{l}{\\emph{Strongest open-VLM language priors}} \\\\")
    for slug, label in [("Qwen3.5-397B", "Qwen3.5-397B"),
                        ("Qwen3-VL-235B-A22B-Instruct", "Qwen3-VL-235B")]:
        ti = text_only_itw(slug); to = text_only_obj(slug)
        rows.append(f"{label} (text-only) & {fmt_acc(ti)} & {fmt_acc(to)} \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append(
        "\\caption{Language-prior summary. The open-weight LLM rows "
        "(unparsable scored incorrect) are fully reproducible references "
        "for the frontier ChatGPT~5.5 ceiling; even the strongest open-VLM "
        "text-only priors fall below it. Full per-model vision / "
        "gray-blank / text-only accuracies for all $12$ open VLMs are in "
        "Appendix~\\ref{sec:appendix}, Table~\\ref{tab:language_prior}.}"
    )
    rows.append("\\label{tab:language_prior_compact}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Table 3 — Context-variant ablation (Step 3+5)
# ─────────────────────────────────────────────────────────────────────────────
def table_context():
    variants = ["full", "object_only", "object_ring", "wrong_object", "random_mask"]
    rows = []
    rows.append("\\begin{table*}[t]")
    rows.append("\\centering\\small")
    rows.append("\\begin{tabular}{l|c|" + "c" * len(variants) + "}")
    rows.append("\\toprule")
    rows.append("Model & text-only & " +
                " & ".join(v.replace("_", "\\_") for v in variants) + " \\\\")
    rows.append("\\midrule")
    for slug, label in MODELS:
        to = text_only_itw(slug)
        cells = []
        for v in variants:
            m = context_variant(slug, v)
            cells.append(fmt_acc(m))
        rows.append(f"{label} & {fmt_acc(to)} & " + " & ".join(cells) + " \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\caption{Context-variant probe on in-the-wild (Acc@10\\%). "
                "The \\textit{text-only} column is the per-model LM-prior anchor "
                "(no image sent to the model at all). The five context "
                "variants share the \\textit{object\\_only} canvas (gray fill "
                "outside the SAM mask) but differ in what visual evidence is "
                "added: \\emph{full} = full scene; \\emph{object\\_only} = "
                "target mask only; \\emph{object\\_ring} = target + 50px "
                "halo; \\emph{wrong\\_object} = a different object's mask "
                "swapped in; \\emph{random\\_mask} = a random rectangle of "
                "matching area. Reading each variant against text-only tells "
                "us what the added visual evidence buys: target identity "
                "(\\textit{object\\_only} vs text-only), local context "
                "(\\textit{object\\_ring} vs \\textit{object\\_only}), and "
                "the cost of mis-identifying the target "
                "(\\textit{wrong\\_object} vs \\textit{object\\_only}).}")
    rows.append("\\label{tab:context}")
    rows.append("\\end{table*}")
    return "\n".join(rows) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Table 4 — Apparent-size invariance (Step 4)
# ─────────────────────────────────────────────────────────────────────────────
def _zoom_table(direction: str) -> str:
    """direction in {'in', 'out'}."""
    assert direction in ("in", "out")
    factor = "1.4" if direction == "in" else "0.6"
    accessor = zoom_in if direction == "in" else zoom_out
    rows = []
    rows.append("\\begin{table*}[t]")
    rows.append("\\centering\\small")
    rows.append("\\begin{tabular}{l|cc|cc}")
    rows.append("\\toprule")
    rows.append("Model & text-only & object\\_only "
                f"& \\multicolumn{{2}}{{c}}{{zoom-{direction} {factor}$\\times$}} \\\\")
    rows.append(" & Acc@10 & Acc@10 & MAE & Acc@10 \\\\")
    rows.append("\\midrule")
    for slug, label in MODELS:
        z = accessor(slug)
        oo = context_variant(slug, "object_only")
        to = text_only_itw(slug)
        rows.append(f"{label} & {fmt_acc(to)} & {fmt_acc(oo)} & {fmt_mae(z)} & {fmt_acc(z)} \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    if direction == "in":
        rows.append("\\caption{Zoom-in $1.4\\times$ on object-only images. The "
                    "target object is scaled up by 40\\% on the gray canvas; "
                    "no scene or reference cues. Compared against the "
                    "\\textit{object\\_only} anchor (same image, no zoom). Most "
                    "VLMs are invariant to this magnification, consistent with "
                    "category-prior-based estimation rather than pixel-area "
                    "readout.}")
        rows.append("\\label{tab:zoom_in}")
    else:
        rows.append("\\caption{Zoom-out $0.6\\times$ on object-only images. "
                    "The target object is scaled down by 40\\%; remaining "
                    "canvas filled gray. Compared against the "
                    "\\textit{object\\_only} anchor. See "
                    "Table~\\ref{tab:apparent_size_bias} for the paired in/out ratio "
                    "test of apparent-size bias.}")
        rows.append("\\label{tab:zoom_out}")
    rows.append("\\end{table*}")
    return "\n".join(rows) + "\n"


def table_zoom_in():
    return _zoom_table("in")


def table_zoom_out():
    return _zoom_table("out")


def _name_map_context():
    return {
        "Qwen2.5-VL-3B-Instruct":    "Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL-7B-Instruct":    "Qwen2.5-VL-7B-Instruct",
        "PaliGemma-3B-Mix-448":      "paligemma-3b-mix-448",
        "SmolVLM-Instruct":          "SmolVLM-Instruct",
        "Gemma3-4B-IT":              "gemma-3-4b-it",
        "granite-vision-3.3-2b":     "granite-vision-3.3-2b",
        "LLaVA-v1.6-Mistral-7B":     "llava-v1.6-mistral-7b-hf",
        "LLaVA-OneVision-7B":        "llava-onevision-qwen2-7b-ov-hf",
        "Molmo-7B-D":                "Molmo-7B-D-0924",
        "Phi-3.5-Vision":            "Phi-3.5-vision-instruct",
        "InternVL2-8B":              "InternVL2-8B",
        "Idefics3-8B-Llama3":        "Idefics3-8B-Llama3",
    }


def table_apparent_size_bias():
    """Per-model apparent-size bias on the in-the-wild split.

    For each (image, query) we pair the model's prediction at
    object_only (target alone, no scaling) with its prediction at
    zoom-out 0.6x (same target scaled down). The per-row ratio
        r = pred(zoom-out) / pred(object_only)
    is the right scale-free measure of apparent-size bias because GT
    spans 0.8-245 cm and absolute cm shifts are not comparable across
    queries. r = 1 means invariant; r < 1 means the model shrinks its
    prediction when the object is rendered smaller (apparent-size
    bias); r > 1 means the model grows its prediction.

    Diagnostic columns: median paired ratio (scale-free) and the
    percentage of paired rows with r < 1 (consensus indicator).
    """
    name_map = _name_map_context()
    rows = []
    rows.append("\\begin{table*}[t]")
    rows.append("\\centering\\small")
    rows.append("\\begin{tabular}{l|c|c|c}")
    rows.append("\\toprule")
    rows.append("Model & med ratio & \\% rows ratio $<$ 1 & direction \\\\")
    rows.append(" & (zoom-out / obj\\_only) & & \\\\")
    rows.append("\\midrule")
    for slug, label in MODELS:
        ratio_s = pct_s = direction = "--"
        try:
            ctx_short = name_map.get(slug, slug)
            df_oo = pd.read_csv(RES / "context_object_only" / ctx_short / "predictions.csv")
            df_zo = pd.read_csv(RES / f"zoom_out_{slug}/predictions.csv")
            df_oo = df_oo[df_oo["predicted_value_cm"].notna()].copy()
            df_zo = df_zo[df_zo["predicted_value_cm"].notna()].copy()
            df_oo = df_oo[df_oo["predicted_value_cm"].abs() < 1e4]
            df_zo = df_zo[df_zo["predicted_value_cm"].abs() < 1e4]
            paired = df_oo.merge(df_zo, on=["query", "answer"], suffixes=("_oo", "_zo"))
            # drop rows where divisor is non-positive (would blow up the ratio)
            paired = paired[paired["predicted_value_cm_oo"] > 0]
            if len(paired) > 5:
                r = paired["predicted_value_cm_zo"] / paired["predicted_value_cm_oo"]
                med_r = float(r.median())
                pct_below = float((r < 1).mean() * 100)
                ratio_s = f"{med_r:.2f}"
                pct_s = f"{pct_below:.1f}\\%"
                # verdict: scale-free, requires both magnitude (>=10% shift)
                # AND consensus (>=55% of paired rows in that direction).
                if med_r <= 0.90 and pct_below >= 55:
                    direction = "shrinks $\\downarrow$"
                elif med_r >= 1.10 and pct_below <= 45:
                    direction = "grows $\\uparrow$"
                else:
                    direction = "invariant"
        except FileNotFoundError:
            pass
        rows.append(f"{label} & {ratio_s} & {pct_s} & {direction} \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\caption{Apparent-size bias: zoom-out $0.6\\times$ vs "
                "\\textit{object\\_only} (same image, same identity, only "
                "apparent pixel size differs). We report the scale-free "
                "median paired ratio "
                "$r = \\mathrm{med}(\\hat{y}_{\\mathrm{zoom\\text{-}out}} / "
                "\\hat{y}_{\\mathrm{object\\_only}})$ across paired rows. "
                "$r = 1$ is invariance; $r < 1$ means the model shrinks its "
                "prediction when the target is rendered smaller. The "
                "\\textit{direction} column codes the per-model verdict: "
                "\\textbf{shrinks $\\downarrow$} when $r \\leq 0.90$ (at "
                "least 10\\% shrinkage) and $\\geq 55$\\% of paired rows "
                "decrease; \\textbf{grows $\\uparrow$} for the symmetric "
                "case; \\textbf{invariant} otherwise.}")
    rows.append("\\label{tab:apparent_size_bias}")
    rows.append("\\end{table*}")
    return "\n".join(rows) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Table 5 — Target+reference (when available)
# ─────────────────────────────────────────────────────────────────────────────
def table_target_plus_ref():
    rows = []
    rows.append("\\begin{table}[t]")
    rows.append("\\centering\\small")
    rows.append("\\begin{tabular}{l|cc|cc}")
    rows.append("\\toprule")
    rows.append("Model & text-only & object\\_only "
                "& \\multicolumn{2}{c}{target+reference} \\\\")
    rows.append(" & Acc@10 & Acc@10 & MAE (cm) & Acc@10\\% \\\\")
    rows.append("\\midrule")
    for slug, label in MODELS:
        # eval_target_plus_ref stores under HF repo with '--' separator
        short_map = {
            "Qwen2.5-VL-3B-Instruct":  "Qwen--Qwen2.5-VL-3B-Instruct",
            "Qwen2.5-VL-7B-Instruct":  "Qwen--Qwen2.5-VL-7B-Instruct",
            "PaliGemma-3B-Mix-448":    "google--paligemma-3b-mix-448",
            "SmolVLM-Instruct":        "HuggingFaceTB--SmolVLM-Instruct",
            "Gemma3-4B-IT":            "google--gemma-3-4b-it",
            "granite-vision-3.3-2b":   "ibm-granite--granite-vision-3.3-2b",
            "LLaVA-v1.6-Mistral-7B":   "llava-hf--llava-v1.6-mistral-7b-hf",
            "LLaVA-OneVision-7B":      "llava-hf--llava-onevision-qwen2-7b-ov-hf",
            "Molmo-7B-D":              "allenai--Molmo-7B-D-0924",
            "Phi-3.5-Vision":          "microsoft--Phi-3.5-vision-instruct",
            "InternVL2-8B":            "OpenGVLab--InternVL2-8B",
            "Idefics3-8B-Llama3":      "HuggingFaceM4--Idefics3-8B-Llama3",
        }
        sub = short_map.get(slug, slug)
        m = metrics_from(RES / "context_target_plus_ref" / sub / "direct")
        oo = context_variant(slug, "object_only")
        to = text_only_itw(slug)
        rows.append(f"{label} & {fmt_acc(to)} & {fmt_acc(oo)} & {fmt_mae(m)} & {fmt_acc(m)} \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\caption{Target+reference probe. The target object is composited next to an explicit reference object from the same scene; the model is asked the target's dimension. Pairs enumerated over all (target, ref) combinations on the in-the-wild set (1{,}254 pairs, direct prompt). The \\textit{text-only} column is the per-model LM-prior anchor; \\textit{object\\_only} is the bare-object anchor. Gap (target+ref $-$ object\\_only) isolates whether the VLM exploits the reference object's known scale.}")
    rows.append("\\label{tab:target_plus_ref}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Table 6 — Geometry / vanishing-point perturbation (NEW)
# ─────────────────────────────────────────────────────────────────────────────
def table_distortion():
    """Distortion probe across 4 conditions, paired against full image.

    Reports Acc@10% on the paired intersection (matched (query, answer)
    rows present in both full and the distortion variant) so per-model
    parse-failure differences don't confound the comparison.
    """
    rows = []
    rows.append("\\begin{table*}[t]")
    rows.append("\\centering\\small")
    rows.append("\\begin{tabular}{l|c|cc|c}")
    rows.append("\\toprule")
    rows.append("Model & full Acc@10\\% & "
                "$\\Delta$ barrel $k{=}0.15$ "
                "& $\\Delta$ pincushion $k{=}-0.15$ "
                "& verdict \\\\")
    rows.append("\\midrule")

    def paired_rel(model_slug, dist_dir):
        """Return aligned (rel_full, rel_distorted) arrays on the paired
        (query, answer) intersection, or None if too few rows."""
        f = pd.read_csv(RES / model_slug / "predictions.csv")
        d_path = RES / f"{dist_dir}_{model_slug}"
        if not (d_path / "predictions.csv").exists():
            return None
        d = pd.read_csv(d_path / "predictions.csv")
        for df in (f, d):
            df["answer"] = pd.to_numeric(df["answer"], errors="coerce")
            df["predicted_value_cm"] = pd.to_numeric(df["predicted_value_cm"], errors="coerce")
        f = f.dropna(subset=["predicted_value_cm", "answer"])
        d = d.dropna(subset=["predicted_value_cm", "answer"])
        f = f[f["predicted_value_cm"].abs() < 1e4]
        d = d[d["predicted_value_cm"].abs() < 1e4]
        f["_k"] = f["query"].astype(str).str.strip().str.lower() + "||" + f["answer"].astype(str)
        d["_k"] = d["query"].astype(str).str.strip().str.lower() + "||" + d["answer"].astype(str)
        m = f.merge(d, on="_k", suffixes=("_f", "_d"))
        if len(m) < 10:
            return None
        rel_f = ((m["predicted_value_cm_f"] - m["answer_f"]).abs()
                 / m["answer_f"].abs()).values
        rel_d = ((m["predicted_value_cm_d"] - m["answer_d"]).abs()
                 / m["answer_d"].abs()).values
        return rel_f, rel_d

    import hashlib
    def _stable_seed(s):
        return int.from_bytes(hashlib.md5(s.encode()).digest()[:4], "big")

    for slug, label in MODELS:
        deltas = []        # list of (delta, lo, hi) or None
        full_acc = None
        for di, dist_dir in enumerate(["distorted", "distorted_pin015"]):
            r = paired_rel(slug, dist_dir)
            if r is None:
                deltas.append(None)
            else:
                rel_f, rel_d = r
                full_acc = float((rel_f <= 0.10).mean() * 100)
                d_mean, d_lo, d_hi = paired_bootstrap_acc_delta(
                    rel_f, rel_d, seed=_stable_seed(f"{slug}|{dist_dir}"))
                deltas.append((d_mean, d_lo, d_hi))

        non_null = [d for d in deltas if d is not None]
        if not non_null:
            verdict = "--"
        else:
            # "significant" hurts when CI is fully below 0 by > -1 pp
            n_sig_neg = sum(1 for d in non_null if d[2] < 0)
            n_sig_pos = sum(1 for d in non_null if d[1] > 0)
            n_overlap = len(non_null) - n_sig_neg - n_sig_pos
            if n_sig_neg == len(non_null):
                verdict = "all hurt $\\downarrow$"
            elif n_sig_pos == len(non_null):
                verdict = "all help $\\uparrow$"
            elif n_overlap == len(non_null):
                verdict = "invariant"
            else:
                verdict = "mixed"

        def fmt_d(d):
            if d is None:
                return "--"
            m, lo, hi = d
            return f"${m:+.1f}$\\,{{\\scriptsize[{lo:+.1f},\\,{hi:+.1f}]}}"

        full_s = f"{full_acc:.1f}\\%" if full_acc is not None else "--"
        rows.append(f"{label} & {full_s} & " + " & ".join(fmt_d(d) for d in deltas) + f" & {verdict} \\\\")

    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append(
        "\\caption{\\textbf{Radial distortion probe (paired Acc@10\\%).} "
        "Each image is warped with $r' = r(1 + k r^2)$ at $|k| = 0.15$ "
        "in both barrel and pincushion directions, with edge-preserving "
        "pre-shrink by $1/(1+|k|)$. $\\Delta$ columns are Acc@10\\% on "
        "the distorted condition minus Acc@10\\% on the full image, "
        "computed on the paired intersection of (query, answer) rows "
        "present in both runs; the bracketed range is a $95$\\% "
        "paired-bootstrap CI on $\\Delta$ ($B{=}2000$ resamples of "
        "paired row indices). \\textbf{verdict} classifies each model: "
        "\\emph{invariant} when both CIs cover $0$; \\emph{all hurt}/"
        "\\emph{all help} when both CIs lie strictly below/above $0$; "
        "\\emph{mixed} otherwise. Only one model (Qwen2.5-VL-7B) shows "
        "significant degradation under both directions (the closest "
        "signature of partial perspective-cue use); three models are "
        "\\emph{mixed} (significant shift in one direction, CI covering "
        "zero in the other); the remaining eight are CI-invariant. The "
        "sign of the shift does not pattern with the direction of "
        "distortion (barrel vs.\\ pincushion) for any model, ruling out "
        "vanishing-point-specific reasoning.}"
    )
    rows.append("\\label{tab:distortion}")
    rows.append("\\end{table*}")
    return "\n".join(rows) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Appendix — Paired-bootstrap CIs on the key context-variant deltas
# ─────────────────────────────────────────────────────────────────────────────
def table_context_deltas_ci():
    """Per-model paired-bootstrap 95% CIs on three context-variant deltas:
    identity collapse (wrong_object - object_only), local-context lift
    (object_ring - object_only), and target-pixel content (object_only -
    random_mask). Same MD5-seeded bootstrap as the distortion table."""
    rows = []
    rows.append("\\begin{table*}[t]")
    rows.append("\\centering\\small")
    rows.append("\\begin{tabular}{l|ccc}")
    rows.append("\\toprule")
    rows.append("Model & $\\Delta$ wrong\\_obj. & $\\Delta$ ring & $\\Delta$ target pixels \\\\")
    rows.append(" & (vs object\\_only) & (vs object\\_only) & (object\\_only $-$ random\\_mask) \\\\")
    rows.append("\\midrule")

    def fmt(d):
        if d is None:
            return "--"
        _, m, lo, hi = d
        return f"${m:+.1f}$\\,{{\\scriptsize[{lo:+.1f},\\,{hi:+.1f}]}}"

    for slug, label in MODELS:
        d_wrong = paired_context_delta(slug, "object_only", "wrong_object")
        d_ring  = paired_context_delta(slug, "object_only", "object_ring")
        # target-pixel content is object_only - random_mask, i.e. random_mask
        # is the "a" side and object_only the "b" side so positive = pixels help
        d_pix   = paired_context_delta(slug, "random_mask", "object_only")
        rows.append(f"{label} & {fmt(d_wrong)} & {fmt(d_ring)} & {fmt(d_pix)} \\\\")

    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append(
        "\\caption{\\textbf{Paired-bootstrap CIs on the three load-bearing "
        "context deltas} (in-the-wild). Each cell is the per-model "
        "Acc@10\\% $\\Delta$ between two context variants on the paired "
        "intersection of (query, answer) rows, with a $95$\\% paired "
        "bootstrap CI ($B{=}2000$ resamples; MD5-stable seed). "
        "\\textbf{wrong\\_obj.}: replacing the target's mask with a "
        "different object's collapses Acc@10\\% (CI strictly below $0$) "
        "for the models that use target identity. \\textbf{ring}: adding "
        "a $50$\\,px halo lifts Acc@10\\% (CI strictly above $0$) for "
        "the small-to-mid pool. \\textbf{target pixels}: the gap "
        "object\\_only $-$ random\\_mask is positive when the target's "
        "actual pixel content carries information beyond its area; the "
        "CI distinguishes models that genuinely use target pixels from "
        "those tied with a matched-area random patch.}"
    )
    rows.append("\\label{tab:context_deltas_ci}")
    rows.append("\\end{table*}")
    return "\n".join(rows) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# LoRA — PaliGemma-3B base vs LoRA-finetuned (step 4992) across conditions
# ─────────────────────────────────────────────────────────────────────────────
LORA_BASE_DIRS = {
    "ITW full image":        ("PaliGemma-3B-Mix-448",
                              "lora_paligemma-3b-mix-448_step4992/in_the_wild_vision"),
    "ITW text-only":         ("text_only_PaliGemma-3B-Mix-448",
                              "lora_paligemma-3b-mix-448_step4992/in_the_wild_text_only"),
    "ITW gray-blank":        ("gray_PaliGemma-3B-Mix-448",
                              "lora_paligemma-3b-mix-448_step4992/in_the_wild_gray"),
    "ITW zoom-in $1.4\\times$":  ("zoom_in_PaliGemma-3B-Mix-448",
                                  "lora_paligemma-3b-mix-448_step4992/in_the_wild_zoom_in"),
    "ITW zoom-out $0.6\\times$": ("zoom_out_PaliGemma-3B-Mix-448",
                                  "lora_paligemma-3b-mix-448_step4992/in_the_wild_zoom_out"),
    "ITW barrel $k{=}0.15$": ("distorted_PaliGemma-3B-Mix-448",
                              "lora_paligemma-3b-mix-448_step4992/in_the_wild_distorted_k015"),
    "ITW pincushion $k{=}-0.15$": ("distorted_pin015_PaliGemma-3B-Mix-448",
                                   "lora_paligemma-3b-mix-448_step4992/in_the_wild_distorted_pin015"),
    "Obj.\\ unique-27":      ("objectron_PaliGemma-3B-Mix-448",
                              "lora_paligemma-3b-mix-448_step4992/objectron_vision_unique27"),
}


def table_lora_paligemma():
    """Side-by-side MAE/Acc@10% for base PaliGemma-3B vs LoRA-finetuned
    (step 4992) across the intervention conditions."""
    rows = []
    rows.append("\\begin{table}[t]")
    rows.append("\\centering\\small")
    rows.append("\\begin{tabular}{l|cc|cc}")
    rows.append("\\toprule")
    rows.append("Condition & \\multicolumn{2}{c|}{PaliGemma-3B (base)} & \\multicolumn{2}{c}{+ LoRA (step 4992)} \\\\")
    rows.append(" & MAE $\\downarrow$ & Acc@10\\% $\\uparrow$ & MAE $\\downarrow$ & Acc@10\\% $\\uparrow$ \\\\")
    rows.append("\\midrule")
    for label, (base_dir, lora_dir) in LORA_BASE_DIRS.items():
        b = metrics_from(RES / base_dir)
        l = metrics_from(RES / lora_dir)
        rows.append(
            f"{label} & {fmt_mae(b)} & {fmt_acc(b)} & {fmt_mae(l)} & {fmt_acc(l)} \\\\"
        )
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append(
        "\\caption{\\textbf{LoRA fine-tuning of PaliGemma-3B} across "
        "intervention conditions, comparing the released base checkpoint "
        "to a single LoRA adapter at training step $4992$. The base "
        "model refuses text-only (returns empty); the adapter answers "
        "but with very low Acc@10\\%, while \\emph{gray-blank} Acc@10\\% "
        "doubles from $5.1\\%$ to $10.1\\%$---a measurable recovery of "
        "language-prior access. Headline ITW full-image Acc@10\\% moves "
        "only marginally ($14.4\\% \\to 14.8\\%$); the largest gains are "
        "on Objectron unique-$27$ ($15.8\\% \\to 48.1\\%$) and zoom-in "
        "($12.3\\% \\to 19.2\\%$). The pattern is consistent: LoRA "
        "primarily restores the model's category prior, with smaller "
        "secondary gains on perturbations that benefit from prior "
        "stabilisation.}")
    rows.append("\\label{tab:lora_paligemma}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Table 7 — V3 typo follow-up (compact)
# ─────────────────────────────────────────────────────────────────────────────
def table_v3_typos():
    rows = []
    rows.append("\\begin{table}[t]")
    rows.append("\\centering\\small")
    rows.append("\\begin{tabular}{l|cc|cc}")
    rows.append("\\toprule")
    rows.append("Model & \\multicolumn{2}{c|}{v2 (typo'd)} & \\multicolumn{2}{c}{v3 (cleaned)} \\\\")
    rows.append(" & MAE & Acc@10 & MAE & Acc@10 \\\\")
    rows.append("\\midrule")
    # For v2: take metrics from the paper's existing in-the-wild result on the
    # 8 typo'd rows. Not directly available, omit for v2 — only v3 typo metrics
    # are easy to produce.
    for slug, label in MODELS:
        v3 = metrics_from(RES / f"v3_typo_{slug}")
        rows.append(f"{label} & -- & -- & {fmt_mae(v3)} & {fmt_acc(v3)} \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\caption{Re-evaluation on 8 in-the-wild queries whose original wording in v2 contained typos (e.g.\\ ``\\textit{umberalla}'' $\\to$ ``\\textit{umbrella}''). v3 results listed; comparison to v2 left to per-row analysis.}")
    rows.append("\\label{tab:v3_typos}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


def table_ranking():
    """Cross-benchmark validity check: per-model ranking on in-the-wild vs
    Objectron, plus Spearman \\rho of the two ranks.

    Purpose: show the in-the-wild split is a meaningful probe by
    demonstrating that model rankings on it correlate with rankings on
    the established Objectron benchmark. Ranked by Acc@10% (higher =
    better) rather than MAE, because MAE on ITW is dominated by outliers
    over the 0.8-245 cm GT range and is not comparable across splits.
    """
    from scipy.stats import spearmanr
    # collect (label, itw_acc, obj_acc)
    raw = []
    for slug, label in MODELS:
        mi = vision_itw(slug); mo = vision_obj(slug)
        raw.append((label,
                    mi["acc10"] if mi else None,
                    mo["acc10"] if mo else None))

    # rank: descending by Acc@10% (higher = better), missing → no rank
    def rank_of(values):
        items = [(i, v) for i, v in enumerate(values) if v is not None]
        items.sort(key=lambda x: -x[1])  # higher Acc first
        ranks = [None] * len(values)
        for r, (i, _) in enumerate(items, start=1):
            ranks[i] = r
        return ranks

    itw_ranks = rank_of([m[1] for m in raw])
    obj_ranks = rank_of([m[2] for m in raw])

    # build display rows sorted by ITW rank (best first); missing-ITW at bottom
    display = []
    for (label, mi, mo), ri, ro in zip(raw, itw_ranks, obj_ranks):
        sort_key = ri if ri is not None else 999
        display.append((sort_key, label, mi, ri, mo, ro))
    display.sort()

    def fmt_v(v):  return f"{v*100:.1f}\\%" if v is not None else "--"
    def fmt_r(r):  return f"{r}" if r is not None else "--"
    def fmt_d(ri, ro):
        if ri is None or ro is None:
            return "--"
        d = ro - ri
        if d == 0:   return "0"
        if d > 0:    return f"$\\downarrow${d}"  # worse on Obj than ITW
        return f"$\\uparrow${-d}"                 # better on Obj than ITW

    rows = []
    rows.append("\\begin{table}[t]")
    rows.append("\\centering\\small")
    rows.append("\\begin{tabular}{l|cc|cc|c}")
    rows.append("\\toprule")
    rows.append(" & \\multicolumn{2}{c|}{In-the-wild} & "
                "\\multicolumn{2}{c|}{Objectron} & rank shift \\\\")
    rows.append("Model & Acc@10\\% & rank & Acc@10\\% & rank & (Obj vs ITW) \\\\")
    rows.append("\\midrule")
    for _, label, mi, ri, mo, ro in display:
        rows.append(f"{label} & {fmt_v(mi)} & {fmt_r(ri)} & "
                    f"{fmt_v(mo)} & {fmt_r(ro)} & {fmt_d(ri, ro)} \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    # Spearman rank correlation between ITW and Obj ranks (on models with both)
    paired = [(ri, ro) for _, _, _, ri, _, ro in display
              if ri is not None and ro is not None]
    if len(paired) >= 3:
        a, b = zip(*paired)
        rho, pval = spearmanr(a, b)
        rho_s = f"Spearman $\\rho = {rho:.2f}$ ($p = {pval:.3f}$, $n = {len(paired)}$)"
    else:
        rho_s = "Spearman $\\rho$: insufficient data"

    rows.append("\\caption{\\textbf{Cross-benchmark validity check.} "
                "The 12 open VLMs ranked by Acc@10\\% on each split (rank "
                "$1$ = highest Acc@10\\%), sorted by in-the-wild rank. "
                "The \\textit{rank shift} column reports Objectron rank "
                "minus in-the-wild rank: $\\uparrow k$ means $k$ ranks "
                "better on Objectron, $\\downarrow k$ means $k$ ranks "
                "worse. " + rho_s + " between the two rankings indicates "
                "that the in-the-wild split is a meaningful probe rather "
                "than noise---models that rank well on the curated "
                "Objectron benchmark also rank well on uncalibrated "
                "in-the-wild scenes, with most rank shifts of $\\pm 1$--$2$. "
                "We rank by Acc@10\\% rather than MAE because MAE on "
                "in-the-wild is dominated by outliers over the $0.8$--$245$ "
                "cm GT range and is not directly comparable across the two "
                "splits.}")
    rows.append("\\label{tab:ranking}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


def _classify(df):
    """F=parse fail; S=scale fail (|y_hat-y|>y); E=estimation (rel err <=100%)."""
    import numpy as np
    df = df.copy()
    df["answer"] = pd.to_numeric(df["answer"], errors="coerce")
    df["predicted_value_cm"] = pd.to_numeric(df["predicted_value_cm"], errors="coerce")
    ae = (df["predicted_value_cm"] - df["answer"]).abs()
    rel = ae / df["answer"].abs()
    df["abs_error"], df["rel_error"] = ae, rel
    df["bucket"] = np.where(
        df["predicted_value_cm"].isna(), "F",
        np.where(ae > df["answer"], "S", "E"),
    )
    return df


def _fse_row(d):
    p = d / "predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "predicted_value_cm" not in df.columns or "answer" not in df.columns:
        return None
    df = _classify(df)
    n = len(df)
    e_mask = df["bucket"] == "E"
    s_mask = df["bucket"] == "S"
    n_f = int((df["bucket"] == "F").sum())
    n_s = int(s_mask.sum())
    n_e = int(e_mask.sum())
    e_df = df[e_mask]
    s_df = df[s_mask]
    return {
        "N": n,
        "pct_F": 100 * n_f / n,
        "pct_S": 100 * n_s / n,
        "pct_E": 100 * n_e / n,
        "med_AE_S": float(s_df["abs_error"].median()) if n_s else None,
        "MAE_E": float(e_df["abs_error"].mean()) if n_e else None,
        "Acc10_E": 100 * float((e_df["rel_error"] <= 0.10).mean()) if n_e else None,
    }


def table_llm_baselines():
    """Pure-LLM text-only baselines (no vision capability), in-the-wild +
    Objectron (broadcasted from the 27 unique queries to the full 10,482).
    Compares ChatGPT 5.5 (existing) against three additional frontier LLMs
    queried text-only: gpt-5.5, gpt-oss-120b, deepseek-v4-flash."""
    rows = []
    rows.append("\\begin{table}[t]")
    rows.append("\\centering\\small")
    rows.append("\\begin{tabular}{l|cc|cc}")
    rows.append("\\toprule")
    rows.append("LLM (text-only, no image) & "
                "\\multicolumn{2}{c|}{In-the-wild} & "
                "\\multicolumn{2}{c}{Objectron} \\\\")
    rows.append(" & MAE & Acc@10\\% & MAE & Acc@10\\% \\\\")
    rows.append("\\midrule")
    # ChatGPT 5.5 existing baseline (read via the dedicated baseline() helper)
    li = baseline("llm_itw"); lo = baseline("llm_obj")
    rows.append(f"ChatGPT~5.5 (queried via OpenAI API) & {fmt_mae(li)} & {fmt_acc(li)} & {fmt_mae(lo)} & {fmt_acc(lo)} \\\\")
    for slug, label in [
        ("gpt-5.5",            "gpt-5.5 (via Shaked RITS endpoint)"),
        ("gpt-oss-120b",       "gpt-oss-120b"),
        ("deepseek-v4-flash",  "deepseek-v4-flash"),
    ]:
        ti = metrics_from(RES / f"text_only_{slug}")
        to = metrics_from(RES / f"objectron_text_only_{slug}")
        rows.append(f"{label} & {fmt_mae(ti)} & {fmt_acc(ti)} & {fmt_mae(to)} & {fmt_acc(to)} \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\caption{\\textbf{Frontier-LLM text-only baselines.} "
                "Four LLMs queried with the dimension question and no image. "
                "ChatGPT~5.5 from two endpoints (our existing query and "
                "Shaked's RITS endpoint) replicate within $\\pm 0.4$ pp; "
                "gpt-oss-120b lands in the same neighborhood on Objectron. "
                "Objectron numbers are broadcast from the $27$ unique "
                "(category, axis) queries to all $10{,}482$ rows (valid "
                "because text-only inputs are image-independent).}")
    rows.append("\\label{tab:llm_baselines}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


def table_fse_itw():
    """Failure/Scale/Estimation decomposition on in-the-wild for all 12
    open VLMs, sorted by ascending MAE_E."""
    rows_data = []
    open_rows = []
    for slug, label in MODELS:
        r = _fse_row(RES / slug)
        if r is not None:
            open_rows.append((label, r))
    # sort open by MAE_E ascending (best estimators first)
    open_rows.sort(key=lambda x: (x[1]["MAE_E"] if x[1]["MAE_E"] is not None else 1e9))
    for label, r in open_rows:
        rows_data.append((label, r, False))

    def fmt_pct(v):  return f"{v:.1f}" if v is not None else "--"
    def fmt_mae(v):  return f"{v:.2f}" if v is not None else "--"
    def fmt_acc(v):  return f"{v:.1f}\\%" if v is not None else "--"

    rows = []
    rows.append("\\begin{table*}[t]")
    rows.append("\\centering\\small")
    rows.append("\\begin{tabular}{l|r|rrr|r|rr}")
    rows.append("\\toprule")
    rows.append("Model & $N$ & \\%F & \\%S & \\%E & "
                "med.\\,AE$_{S}$ & MAE$_{E}$ & Acc@10$_{E}$ \\\\")
    rows.append("\\midrule")
    in_ref = False
    for label, r, is_ref in rows_data:
        if in_ref and not is_ref:
            rows.append("\\midrule")
            in_ref = False
        rows.append(
            f"{label} & {r['N']} & {fmt_pct(r['pct_F'])} & "
            f"{fmt_pct(r['pct_S'])} & {fmt_pct(r['pct_E'])} & "
            f"{fmt_mae(r['med_AE_S'])} & {fmt_mae(r['MAE_E'])} & "
            f"{fmt_acc(r['Acc10_E'])} \\\\"
        )
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append(
        "\\caption{\\textbf{Failure / Scale / Estimation (F/S/E) decomposition "
        "on the in-the-wild split.} Each model's $N$ predictions partition into "
        "\\textbf{F} = parse failure (no numeric in the response, including "
        "refusals like ``cannot measure without a reference''); "
        "\\textbf{S} = scale failure ($|\\hat{y} - y| > y$; the model commits "
        "to a number off by more than the true value); and "
        "\\textbf{E} = estimation ($|\\hat{y} - y| \\leq y$). The last three "
        "columns describe the $E$ subset: median absolute error within $S$ "
        "(for context on how bad the wrong numbers are), MAE on $E$ (typical "
        "accuracy \\emph{conditional on the model committing to a reasonable "
        "answer}), and the corresponding Acc@10\\%. Open VLMs are sorted by "
        "ascending MAE$_E$. The decomposition separates ``the model doesn't "
        "engage'' from ``the model engages but is miscalibrated'': both contribute "
        "to a model's headline MAE differently and warrant different mitigations.}"
    )
    rows.append("\\label{tab:fse_itw}")
    rows.append("\\end{table*}")
    return "\n".join(rows) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Fraction probe — well-posed control on the zoom canvases
# ─────────────────────────────────────────────────────────────────────────────
FRACTION_MODELS = [
    # (well_posed dir prefix, paper slug for the cm zoom runs, display label)
    ("qwen3.5-397b",           "Qwen3.5-397B",                "Qwen3.5-397B"),
    ("qwen3-vl-235b",          "Qwen3-VL-235B-A22B-Instruct", "Qwen3-VL-235B"),
    ("Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B-Instruct",      "Qwen2.5-VL-7B"),
    ("LLaVA-OneVision-7B",     "LLaVA-OneVision-7B",          "LLaVA-OV"),
    ("LLaVA-v1.6-Mistral-7B",  "LLaVA-v1.6-Mistral-7B",       "LLaVA-v1.6"),
    ("SmolVLM-Instruct",       "SmolVLM-Instruct",            "SmolVLM"),
    ("Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-3B-Instruct",      "Qwen2.5-VL-3B"),
]

CAP_FRACTION = 0.98  # zoom-in gt fractions above this are clipped by the canvas


def _pair_key(df):
    return df["image"].map(lambda s: os.path.basename(str(s))) + "||" + df["query"].astype(str)


def _median_paired_ratio(df_in, df_out, col_in, col_out, capped_keys=None):
    a = df_in[["image", "query", col_in]].copy()
    a["key"] = _pair_key(a)
    a = a.rename(columns={col_in: "val_in"})
    b = df_out[["image", "query", col_out]].copy()
    b["key"] = _pair_key(b)
    b = b.rename(columns={col_out: "val_out"})
    m = a.merge(b[["key", "val_out"]], on="key")
    if capped_keys is not None:
        m = m[~m["key"].isin(capped_keys)]
    m = m[m["val_in"].notna() & m["val_out"].notna()]
    m = m[(m["val_in"] > 0) & (m["val_out"] > 0)]
    if len(m) < MIN_VALID:
        return None, len(m)
    return float((m["val_in"] / m["val_out"]).median()), len(m)


def table_fraction_probe():
    """Well-posed control: on the same zoom-in/zoom-out canvases, the
    fraction-of-image-width query has a known correct readout (ratio 2.33),
    while the centimeter query does not depend on apparent size."""
    wp = RES / "well_posed"
    body = []
    printable = []
    for wp_name, cm_slug, label in FRACTION_MODELS:
        f_in = wp / f"{wp_name}_zoom_in" / "predictions.csv"
        f_out = wp / f"{wp_name}_zoom_out" / "predictions.csv"
        frac_ratio, n_frac = (None, 0)
        capped = None
        if f_in.exists() and f_out.exists():
            di = pd.read_csv(f_in)
            do = pd.read_csv(f_out)
            if "gt_fraction" in di.columns:
                ci = di.copy(); ci["key"] = _pair_key(ci)
                capped = set(ci.loc[ci["gt_fraction"] * 1.4 >= CAP_FRACTION, "key"])
            frac_ratio, n_frac = _median_paired_ratio(
                di, do, "pred_fraction", "pred_fraction", capped_keys=capped)
        c_in = RES / f"zoom_in_{cm_slug}" / "predictions.csv"
        c_out = RES / f"zoom_out_{cm_slug}" / "predictions.csv"
        cm_ratio, n_cm = (None, 0)
        if c_in.exists() and c_out.exists():
            cm_ratio, n_cm = _median_paired_ratio(
                pd.read_csv(c_in), pd.read_csv(c_out),
                "predicted_value_cm", "predicted_value_cm", capped_keys=capped)
        fr = f"{frac_ratio:.2f}" if frac_ratio is not None else "--"
        cr = f"{cm_ratio:.2f}" if cm_ratio is not None else "--"
        body.append(f"{label} & {fr} & {cr} & {min(n_frac, n_cm) if n_frac and n_cm else max(n_frac, n_cm)} \\\\")
        printable.append((label, frac_ratio, n_frac, cm_ratio, n_cm))
    print("fraction probe (median paired in/out ratios, expected 2.33 for a "
          "direct readout):")
    for label, frq, nf, cmr, nc in printable:
        print(f"  {label:16s} fraction={frq if frq is not None else '--'} "
              f"(n={nf})  cm={cmr if cmr is not None else '--'} (n={nc})")
    rows = []
    rows.append("\\begin{table}[t]")
    rows.append("\\centering\\small")
    rows.append("\\begin{tabular}{l|cc|c}")
    rows.append("\\toprule")
    rows.append(" & \\multicolumn{2}{c|}{median paired ratio (in/out)} & \\\\")
    rows.append("Model & fraction query & cm query & $n$ \\\\")
    rows.append("\\midrule")
    rows.extend(body)
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append(
        "\\caption{\\textbf{Well-posed control (fraction probe).} On the same "
        "object-only zoom canvases ($1.4\\times$ vs $0.6\\times$; "
        "Section~\\ref{sec:int_wellposed}), each model answers a well-posed "
        "query (\\emph{what fraction of the image width does the object "
        "span?}) and the ill-posed metric query (\\emph{how many "
        "centimeters?}). A direct readout of apparent size gives a paired "
        "in/out ratio of $2.33$. Capable models track the manipulation under "
        "the fraction query while their centimeter answers on identical "
        "inputs stay near $1.0$: the apparent-size signal reaches the "
        "models' output, so metric invariance reflects evidence arbitration, "
        "not preprocessing or encoder canonicalization. Ratios are medians "
        "over paired rows with parsable answers in both conditions; zoom-in "
        "rows whose ground-truth fraction would exceed the canvas "
        "($\\geq 0.98$ after $1.4\\times$) are excluded. The probe covers "
        "the $187$ of $342$ context-suite queries with zoom canvases and "
        "the $7$ pool models with fraction runs.}"
    )
    rows.append("\\label{tab:fraction_probe}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


def main():
    tables = {
        "table_main_results.tex":      table_main_results(),
        "table_main_results_mae.tex":  table_main_results_mae(),
        "table_ranking.tex":           table_ranking(),
        "table_fse_itw.tex":           table_fse_itw(),
        "table_language_prior.tex":    table_language_prior(),
        "table_language_prior_compact.tex": table_language_prior_compact(),
        "table_context.tex":           table_context(),
        "table_zoom_in.tex":           table_zoom_in(),
        "table_zoom_out.tex":          table_zoom_out(),
        "table_apparent_size_bias.tex": table_apparent_size_bias(),
        "table_distortion.tex":        table_distortion(),
        "table_context_deltas_ci.tex": table_context_deltas_ci(),
        "table_fraction_probe.tex":    table_fraction_probe(),
    }
    # Remove the old combined zoom table if it exists
    legacy = OUT / "table_zoom.tex"
    if legacy.exists():
        legacy.unlink()
    for name, txt in tables.items():
        (OUT / name).write_text(txt)
        print(f"wrote {OUT / name}  ({len(txt.splitlines())} lines)")


if __name__ == "__main__":
    main()
