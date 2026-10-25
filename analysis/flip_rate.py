"""Row-level flip-rate ("churn") analysis on the in-the-wild split.

For each model, pair its text-only run (no image in the prompt) with its
full-image run on the canonical 331 ITW rows and classify every row by the
text-only outcome:

  prior-correct : text-only answer within 10% of GT
  prior-wrong   : everything else (unparsable counts as incorrect)

  flip rate = fraction of prior-correct rows that become WRONG with the image
  fix  rate = fraction of prior-wrong  rows that become CORRECT with the image

Models
  Open pool: the 11 open models (Molmo-7B-D excluded -- it refuses text-only,
  returning empty on every query).
  Closed reference: gpt-5.5-no-reasoning_bundle (primary), gpt-5.5_bundle and
  gemini-3.1-pro_bundle (default-API ablations). gemini-3.1-pro-no-reasoning
  is skipped (broken / distortions-only rerun pending).

Join conventions (documented because they matter)
  PRIMARY (this table / CSV):
    * Open models: each side is mapped to the canonical 331-row frame with the
      rebuttal_numbers.py fallback chain -- (image, normalized query) primary,
      (image, answer) fallback (recovers the 8 typo-fixed v3 queries in the
      newer text-only manifests), (normalized query, answer) fallback (needed
      for RITS text-only manifests whose image column is '<text-only>').
      Coverage 322-330 of 331 rows per model (>= 97%).
    * Closed bundles: vision x text_only joined within bundle on
      (normalized query, answer) -- the two manifests are the same canonical
      331 rows, so this covers all 331. Identical to the rebuttal
      "closed churn" convention; reproduces numbers.md exactly.
  LEGACY (numbers.md open-pool line, reproduced for the acceptance check):
    prior_split.py convention -- key = base image filename + case-SENSITIVE
    stripped query. Case differences between the text-only and vision
    manifests drop ~149 rows per local model (n=182 joined), and the three
    RITS models (InternVL3.5, Qwen3-VL-235B, Qwen3.5-397B) contribute ZERO
    rows because their text-only image column is '<text-only>'. The quoted
    "11 models" pool therefore effectively contains 8 models. The primary
    convention above fixes both defects; pooled rates move 49%->48% (flips)
    and 10.7%->12.1% (fixes), same reading.

Outputs
  results/flip_rate_summary.csv
  paper/emnlp_paper/tables/table_flip_rate.tex   (appendix, label tab:flip_rate)

Run:  conda run -n vlm-eval python paper/emnlp_scripts/flip_rate.py
"""

from __future__ import annotations

import re
import os
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from release_data import load_itw, TABLES, FIGS  # noqa: E402

REPO = Path(os.environ.get("METRICVQA_ROOT", "."))
RES = REPO / "results"
OUT_CSV = RES / "flip_rate_summary.csv"
OUT_TEX = TABLES / "table_flip_rate.tex"
TABLES.mkdir(parents=True, exist_ok=True)

# 11 open models (build_tables.py pool minus Molmo-7B-D, which refuses
# text-only), light to heavy.
OPEN_MODELS = [
    ("Qwen2.5-VL-3B-Instruct",      "Qwen2.5-VL-3B"),
    ("LLaVA-v1.6-Mistral-7B",       "LLaVA-v1.6"),
    ("SmolVLM-Instruct",            "SmolVLM"),
    ("Gemma3-4B-IT",                "Gemma3-4B"),
    ("granite-vision-3.3-2b",       "granite-3.3-2b"),
    ("Qwen2.5-VL-7B-Instruct",      "Qwen2.5-VL-7B"),
    ("LLaVA-OneVision-7B",          "LLaVA-OV"),
    ("InternVL2-8B",                "InternVL2"),
    ("InternVL3.5",                 "InternVL3.5-241B"),
    ("Qwen3-VL-235B-A22B-Instruct", "Qwen3-VL-235B"),
    ("Qwen3.5-397B",                "Qwen3.5-397B"),
]

CLOSED_BUNDLES = [
    ("gpt-5.5-no-reasoning_bundle", "GPT-5.5 (no reasoning)"),
    ("gpt-5.5_bundle",              "GPT-5.5 (default API)"),
    ("gemini-3.1-pro_bundle",       "Gemini 3.1 Pro (default API)"),
    # gemini-3.1-pro-no-reasoning_bundle intentionally skipped (broken run)
]


def norm_q(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.lower().str.strip()
            .str.replace(r"\s+", " ", regex=True).str.rstrip("?").str.strip())


def correct_series(df: pd.DataFrame, pred_col: str) -> pd.Series:
    """Acc@10% row correctness; unparsable prediction counts as incorrect."""
    p = pd.to_numeric(df[pred_col], errors="coerce")
    y = pd.to_numeric(df["answer"], errors="coerce")
    return ((p - y).abs() / y <= 0.10).fillna(False)


# ── canonical 331-row ITW frame (same construction as rebuttal_numbers.py) ──
def build_canon():
    # 331 released in-the-wild queries, in release order (same rows as the paper's frame).
    canon = load_itw()[["image", "query", "answer"]].copy()
    canon["nq"] = norm_q(canon["query"])
    canon["gt"] = pd.to_numeric(canon["answer"], errors="coerce")
    canon["rid"] = np.arange(len(canon))
    assert len(canon) == 331 and canon["gt"].notna().all(), "canonical frame broken"
    gtr = canon["gt"].round(2).astype(str)
    k_iq = pd.Series(canon.rid.values, index=canon["image"].str.strip() + "||" + canon["nq"])
    k_ia = pd.Series(canon.rid.values, index=canon["image"].str.strip() + "||" + gtr)
    k_qa = pd.Series(canon.rid.values, index=canon["nq"] + "||" + gtr)
    assert k_iq.index.is_unique and k_qa.index.is_unique
    # (image, answer) can be ambiguous within an image; keep first (rebuttal convention)
    k_ia = k_ia.groupby(level=0).first()
    return canon, k_iq, k_ia, k_qa


CANON, K_IQ, K_IA, K_QA = build_canon()


def map_rid(df: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical row id via (img,nq) -> (img,gt) -> (nq,gt) fallbacks."""
    df = df.copy()
    nq = norm_q(df["query"])
    gtr = pd.to_numeric(df["answer"], errors="coerce").round(2).astype(str)
    img = df["image"].astype(str).map(lambda x: Path(x).name)
    r1 = (img + "||" + nq).map(K_IQ)
    r2 = (img + "||" + gtr).map(K_IA)
    r3 = (nq + "||" + gtr).map(K_QA)
    df["rid"] = r1.fillna(r2).fillna(r3)
    return df


def churn_counts(j: pd.DataFrame) -> dict:
    """j has boolean columns c_t (text-only correct) and c_v (vision correct)."""
    pc = j["c_t"]
    n_pc, n_pw = int(pc.sum()), int((~pc).sum())
    n_flip = int((pc & ~j["c_v"]).sum())
    n_fix = int((~pc & j["c_v"]).sum())
    return {
        "n_joined": len(j),
        "n_prior_correct": n_pc, "n_flipped": n_flip,
        "flip_rate": n_flip / n_pc if n_pc else np.nan,
        "n_prior_wrong": n_pw, "n_fixed": n_fix,
        "fix_rate": n_fix / n_pw if n_pw else np.nan,
    }


# ── primary computation ─────────────────────────────────────────────────────
def open_churn(slug: str) -> dict | None:
    if not ((RES / f"text_only_{slug}" / "predictions.csv").exists()
            and (RES / slug / "predictions.csv").exists()):
        return None
    t = map_rid(pd.read_csv(RES / f"text_only_{slug}" / "predictions.csv"))
    v = map_rid(pd.read_csv(RES / slug / "predictions.csv"))
    t["c"] = correct_series(t, "predicted_value_cm")
    v["c"] = correct_series(v, "predicted_value_cm")
    t = t.dropna(subset=["rid"]).drop_duplicates("rid")
    v = v.dropna(subset=["rid"]).drop_duplicates("rid")
    j = (t[["rid", "c"]].rename(columns={"c": "c_t"})
         .merge(v[["rid", "c"]].rename(columns={"c": "c_v"}), on="rid"))
    return churn_counts(j)


def closed_churn(bundle: str) -> dict | None:
    if not ((RES / bundle / "in_the_wild_text_only" / "predictions.csv").exists()
            and (RES / bundle / "in_the_wild_vision" / "predictions.csv").exists()):
        return None
    t = pd.read_csv(RES / bundle / "in_the_wild_text_only" / "predictions.csv")
    v = pd.read_csv(RES / bundle / "in_the_wild_vision" / "predictions.csv")
    for df in (t, v):
        df["k"] = norm_q(df["query"]) + "||" + \
            pd.to_numeric(df["answer"], errors="coerce").round(2).astype(str)
        df["c"] = correct_series(df, "prediction")
    assert t["k"].is_unique and v["k"].is_unique, f"{bundle}: duplicate keys"
    j = (t[["k", "c"]].rename(columns={"c": "c_t"})
         .merge(v[["k", "c"]].rename(columns={"c": "c_v"}), on="k"))
    assert len(j) == 331, f"{bundle}: joined {len(j)} != 331"
    return churn_counts(j)


# ── legacy numbers.md open-pool convention (acceptance reproduction) ────────
def legacy_open_pool() -> dict:
    def base_key(img):
        b = Path(str(img)).name
        b = re.split(r"__", b)[0]
        return b if b.endswith(".png") else b + ".png"

    def load(path):
        df = pd.read_csv(Path(path) / "predictions.csv")
        out = pd.DataFrame({
            "key": df["image"].map(base_key) + "||" + df["query"].astype(str).str.strip(),
            "c": correct_series(df, "predicted_value_cm"),
        })
        return out.drop_duplicates("key")

    tot = {"n_joined": 0, "n_prior_correct": 0, "n_flipped": 0,
           "n_prior_wrong": 0, "n_fixed": 0}
    for slug, _ in OPEN_MODELS:
        if not ((RES / f"text_only_{slug}" / "predictions.csv").exists()
                and (RES / slug / "predictions.csv").exists()):
            continue
        j = (load(RES / f"text_only_{slug}").rename(columns={"c": "c_t"})
             .merge(load(RES / slug).rename(columns={"c": "c_v"}), on="key"))
        c = churn_counts(j)
        for k in tot:
            tot[k] += c[k]
    tot["flip_rate"] = tot["n_flipped"] / tot["n_prior_correct"] if tot["n_prior_correct"] else float("nan")
    tot["fix_rate"] = tot["n_fixed"] / tot["n_prior_wrong"] if tot["n_prior_wrong"] else float("nan")
    return tot


# ── output helpers ──────────────────────────────────────────────────────────
def fmt_pct(x):
    return "--" if x != x else f"{100 * x:.0f}\\%"


def build_tex(rows):
    """rows: list of (label, counts, kind) with kind in open/pooled/closed."""
    L = []
    L.append("\\begin{table*}[t]")
    L.append("\\centering\\small")
    L.append("\\begin{tabular}{l|c|ccc|ccc}")
    L.append("\\toprule")
    L.append(" & & \\multicolumn{3}{c|}{prior-correct rows} & "
             "\\multicolumn{3}{c}{prior-wrong rows} \\\\")
    L.append("Model & $n$ & $n_{\\text{corr}}$ & flipped & flip rate & "
             "$n_{\\text{wrong}}$ & fixed & fix rate \\\\")
    L.append("\\midrule")
    prev_kind = None
    for label, c, kind in rows:
        if prev_kind is not None and kind != prev_kind and kind == "closed":
            L.append("\\midrule")
        cells = (f"{c['n_joined']} & {c['n_prior_correct']} & {c['n_flipped']} & "
                 f"{fmt_pct(c['flip_rate'])} & {c['n_prior_wrong']} & "
                 f"{c['n_fixed']} & {fmt_pct(c['fix_rate'])}")
        if kind == "pooled":
            label = f"\\textbf{{{label}}}"
            cells = " & ".join(f"\\textbf{{{x.strip()}}}" for x in cells.split("&"))
        L.append(f"{label} & {cells} \\\\")
        prev_kind = kind
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append(
        "\\caption{\\textbf{Row-level flip-rate (churn) analysis on the "
        "in-the-wild split.} For each model we pair its text-only run (no "
        "image in the prompt) with its full-image run on the canonical $331$ "
        "rows ($n$ = joined rows) and classify every row by the text-only "
        "outcome: \\emph{prior-correct} (text-only answer within $10\\%$ of "
        "ground truth) or \\emph{prior-wrong} (anything else; unparsable "
        "answers count as incorrect). \\emph{Flip rate} is the fraction of "
        "prior-correct rows that become wrong once the image is added; "
        "\\emph{fix rate} is the fraction of prior-wrong rows the image "
        "repairs. If a VLM simply recited its language prior when handed an "
        "image, both rates would be near zero. Instead, adding the image "
        "overturns roughly half of the open pool's prior-correct answers "
        "while repairing only about one in eight prior-wrong rows: the "
        "models \\emph{arbitrate} between the prior and visual evidence "
        "rather than reciting the prior, and because their visual size "
        "evidence is unreliable, the arbitration is a net loss exactly on "
        "the rows the prior already had right. The frontier closed models "
        "show the reverse profile (flip rate $<$ fix rate): the same "
        "arbitration pays off once the visual evidence is strong enough. "
        "Molmo-7B-D is excluded (it refuses text-only prompts). Open-model "
        "rows are joined to the canonical rows by (image, normalized query) "
        "with (image, answer) and (query, answer) fallbacks (join "
        "$\\geq 97\\%$); closed-model rows are joined within bundle on "
        "(normalized query, answer), covering all $331$ rows.}")
    L.append("\\label{tab:flip_rate}")
    L.append("\\end{table*}")
    return "\n".join(L) + "\n"


# ── acceptance check vs numbers.md ──────────────────────────────────────────
def acceptance(primary_closed, legacy_pool):
    print("\n== ACCEPTANCE vs numbers.md ==")
    ok_all = True

    def check(name, got, want):
        nonlocal ok_all
        ok = got == want
        ok_all &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: computed {got}  expected {want}")

    lp = legacy_pool
    check("open pool (legacy conv) flips", f"{lp['n_flipped']}/{lp['n_prior_correct']} "
          f"({100 * lp['flip_rate']:.0f}%)", "54/111 (49%)")
    check("open pool (legacy conv) fixes", f"{lp['n_fixed']}/{lp['n_prior_wrong']} "
          f"({100 * lp['fix_rate']:.0f}%)", "144/1345 (11%)")
    g = primary_closed["gpt-5.5_bundle"]
    check("GPT-5.5 flips", f"{g['n_flipped']}/{g['n_prior_correct']} "
          f"({100 * g['flip_rate']:.0f}%)", "29/92 (32%)")
    check("GPT-5.5 fixes", f"{g['n_fixed']}/{g['n_prior_wrong']} "
          f"({100 * g['fix_rate']:.0f}%)", "107/239 (45%)")
    n = primary_closed["gpt-5.5-no-reasoning_bundle"]
    check("GPT-5.5-nr flips %", f"{100 * n['flip_rate']:.0f}%", "43%")
    check("GPT-5.5-nr fixes %", f"{100 * n['fix_rate']:.0f}%", "38%")
    m = primary_closed["gemini-3.1-pro_bundle"]
    check("Gemini 3.1 Pro flips %", f"{100 * m['flip_rate']:.0f}%", "30%")
    check("Gemini 3.1 Pro fixes %", f"{100 * m['fix_rate']:.0f}%", "44%")
    print(f"  acceptance: {'ALL PASS' if ok_all else 'SOME FAILED'}")
    return ok_all


def main():
    rows = []           # (label, counts, kind)
    pooled = {"n_joined": 0, "n_prior_correct": 0, "n_flipped": 0,
              "n_prior_wrong": 0, "n_fixed": 0}

    print("== flip-rate analysis (primary convention, canonical 331 join) ==")
    print(f"{'model':32s} {'n':>4s} {'prior-corr':>10s} {'flipped':>8s} "
          f"{'flip%':>6s} {'prior-wrong':>11s} {'fixed':>6s} {'fix%':>6s}")
    for slug, label in OPEN_MODELS:
        c = open_churn(slug)
        if c is None:
            print(f"{label:32s} skipped (no text_only/vision predictions in results/)")
            continue
        rows.append((label, c, "open"))
        for k in pooled:
            pooled[k] += c[k]
        print(f"{label:32s} {c['n_joined']:4d} {c['n_prior_correct']:10d} "
              f"{c['n_flipped']:8d} {100 * c['flip_rate']:5.0f}% "
              f"{c['n_prior_wrong']:11d} {c['n_fixed']:6d} {100 * c['fix_rate']:5.0f}%")

    if pooled["n_joined"] == 0:
        raise SystemExit("no model has both text_only_<model>/ and <model>/ predictions in "
                         f"{RES}; run vlm_eval.py and text_only_baseline.py first")
    pooled["flip_rate"] = pooled["n_flipped"] / pooled["n_prior_correct"]
    pooled["fix_rate"] = pooled["n_fixed"] / pooled["n_prior_wrong"]
    rows.append(("Open pool (11 models, pooled)", pooled, "pooled"))
    print(f"{'OPEN POOL (pooled)':32s} {pooled['n_joined']:4d} "
          f"{pooled['n_prior_correct']:10d} {pooled['n_flipped']:8d} "
          f"{100 * pooled['flip_rate']:5.0f}% {pooled['n_prior_wrong']:11d} "
          f"{pooled['n_fixed']:6d} {100 * pooled['fix_rate']:5.0f}%")

    primary_closed = {}
    for bundle, label in CLOSED_BUNDLES:
        c = closed_churn(bundle)
        if c is None:
            print(f"{label:32s} skipped (bundle not in results/)")
            continue
        primary_closed[bundle] = c
        rows.append((label, c, "closed"))
        print(f"{label:32s} {c['n_joined']:4d} {c['n_prior_correct']:10d} "
              f"{c['n_flipped']:8d} {100 * c['flip_rate']:5.0f}% "
              f"{c['n_prior_wrong']:11d} {c['n_fixed']:6d} {100 * c['fix_rate']:5.0f}%")

    # legacy reproduction + acceptance
    legacy = legacy_open_pool()
    print(f"\nlegacy numbers.md open-pool convention (case-sensitive join; "
          f"3 RITS models contribute 0 rows): flips "
          f"{legacy['n_flipped']}/{legacy['n_prior_correct']} "
          f"({100 * legacy['flip_rate']:.0f}%), fixes "
          f"{legacy['n_fixed']}/{legacy['n_prior_wrong']} "
          f"({100 * legacy['fix_rate']:.0f}%)")
    print("primary (corrected) open-pool convention: flips "
          f"{pooled['n_flipped']}/{pooled['n_prior_correct']} "
          f"({100 * pooled['flip_rate']:.0f}%), fixes "
          f"{pooled['n_fixed']}/{pooled['n_prior_wrong']} "
          f"({100 * pooled['fix_rate']:.0f}%)")
    # The acceptance numbers are the paper's; they only apply with every model present.
    if len(primary_closed) == len(CLOSED_BUNDLES) and \
            sum(k == "open" for _, _, k in rows) == len(OPEN_MODELS):
        acceptance(primary_closed, legacy)
    else:
        print("acceptance vs paper numbers skipped (not all models in results/)")

    # CSV
    csv_rows = []
    for label, c, kind in rows:
        csv_rows.append({"model": label, "pool": kind, **c})
    df = pd.DataFrame(csv_rows)[
        ["model", "pool", "n_joined", "n_prior_correct", "n_flipped",
         "flip_rate", "n_prior_wrong", "n_fixed", "fix_rate"]]
    df["flip_rate"] = df["flip_rate"].round(4)
    df["fix_rate"] = df["fix_rate"].round(4)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")

    # TeX
    OUT_TEX.write_text(build_tex(rows))
    print(f"wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
