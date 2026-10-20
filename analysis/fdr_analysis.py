"""Benjamini-Hochberg FDR correction across the 60 model x channel tests.

Reconstruction of the ARR-rebuttal analysis (Jul 2026) that produced
paper/rebuttal_arr_jul2026/scripts/fdr_results.csv. For each of the 12 open
models and 5 contrasts we compute the per-row paired difference of the
correct@10% indicator, test it with a two-sided sign-flip permutation test
(B=10,000, zero diffs dropped, shared numpy Generator seeded 1 -- consumed
in test order so the run is bit-reproducible), and apply Benjamini-Hochberg
at q=0.05 across all 60 tests.

Row-pairing convention (matches the rebuttal's prior_split.py loader):
  key   = base image name (part before '__', '.png' re-appended) + '||' +
          stripped query;
  correct = |pred - gt| / gt <= 0.10, unparsable counted incorrect;
  per-condition drop_duplicates on key (keep first), then an inner
  pivot-join across the two conditions of each contrast.

Contrast direction: delta_pp = Acc(cond_a) - Acc(cond_b), positive = the
channel works in the claimed direction (object identity helps, target
pixels help, ring helps, distortion hurts).

Outputs (only after the survivor set is verified against the original):
  paper/rebuttal_arr_jul2026/scripts/fdr_results.csv        (regenerated)
  paper/rebuttal_arr_jul2026/scripts/fdr_results_original.csv (backup, once)
  paper/emnlp_paper/tables/table_fdr.tex
"""
from __future__ import annotations

import re
import shutil
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("METRICVQA_ROOT", "."))
RES = REPO / "results"
SCRIPTS = REPO / "paper/rebuttal_arr_jul2026/scripts"
CSV_OUT = SCRIPTS / "fdr_results.csv"
CSV_BACKUP = SCRIPTS / "fdr_results_original.csv"
TEX_OUT = REPO / "paper/emnlp_paper/tables/table_fdr.tex"

B_PERM = 10_000
Q_FDR = 0.05

# (slug, paper label) in the canonical lighter-to-heavier order of
# emnlp_scripts/build_tables.py; sorted(slug) gives the rng consumption order.
MODELS = [
    ("Qwen2.5-VL-3B-Instruct",      "Qwen2.5-VL-3B"),
    ("LLaVA-v1.6-Mistral-7B",       "LLaVA-v1.6"),
    ("SmolVLM-Instruct",            "SmolVLM"),
    ("Gemma3-4B-IT",                "Gemma3-4B"),
    ("granite-vision-3.3-2b",       "granite-3.3-2b"),
    ("Qwen2.5-VL-7B-Instruct",      "Qwen2.5-VL-7B"),
    ("LLaVA-OneVision-7B",          "LLaVA-OV"),
    ("Molmo-7B-D",                  "Molmo-7B-D"),
    ("InternVL2-8B",                "InternVL2"),
    ("InternVL3.5",                 "InternVL3.5-241B"),
    ("Qwen3-VL-235B-A22B-Instruct", "Qwen3-VL-235B"),
    ("Qwen3.5-397B",                "Qwen3.5-397B"),
]
LABEL = dict(MODELS)

# context result dirs use HF-style short names for four models
CTX_MAP = {
    "Gemma3-4B-IT": "gemma-3-4b-it",
    "LLaVA-v1.6-Mistral-7B": "llava-v1.6-mistral-7b-hf",
    "LLaVA-OneVision-7B": "llava-onevision-qwen2-7b-ov-hf",
    "Molmo-7B-D": "Molmo-7B-D-0924",
}

CONDITIONS = {
    "full":         lambda s: RES / s,
    "object_only":  lambda s: RES / "context_object_only" / CTX_MAP.get(s, s),
    "object_ring":  lambda s: RES / "context_object_ring" / CTX_MAP.get(s, s),
    "wrong_object": lambda s: RES / "context_wrong_object" / CTX_MAP.get(s, s),
    "random_mask":  lambda s: RES / "context_random_mask" / CTX_MAP.get(s, s),
    "dist_k015":    lambda s: RES / f"distorted_{s}",
    "dist_pin015":  lambda s: RES / f"distorted_pin015_{s}",
}

# (name, cond_a, cond_b); effect = acc_a - acc_b
CONTRASTS = [
    ("identity (obj>wrong)",    "object_only", "wrong_object"),
    ("pixels (obj>random)",     "object_only", "random_mask"),
    ("ring (ring>obj)",         "object_ring", "object_only"),
    ("dist barrel (full>k015)", "full",        "dist_k015"),
    ("dist pin (full>pin015)",  "full",        "dist_pin015"),
]


def base_key(img) -> str:
    b = Path(str(img)).name
    b = re.split(r"__", b)[0]
    return b if b.endswith(".png") else b + ".png"


def load_preds(path: Path) -> pd.DataFrame | None:
    p = path / "predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    pred_col = "predicted_value_cm"
    if pred_col not in df.columns:
        pred_col = "prediction" if "prediction" in df.columns else pred_col
    out = pd.DataFrame({
        "key": df["image"].map(base_key) + "||" + df["query"].astype(str).str.strip(),
        "gt": pd.to_numeric(df["answer"], errors="coerce"),
        "pred": pd.to_numeric(df[pred_col], errors="coerce"),
    })
    out["correct"] = ((out["pred"] - out["gt"]).abs() / out["gt"] <= 0.10).fillna(False)
    return out.drop_duplicates("key")


def build_long() -> pd.DataFrame:
    rows = []
    for m, _ in MODELS:
        for cond, pf in CONDITIONS.items():
            df = load_preds(pf(m))
            if df is None:
                print(f"  MISSING: {m} / {cond}")
                continue
            df = df.copy()
            df["model"], df["cond"] = m, cond
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def run_tests(long: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(1)

    def signflip_p(d: np.ndarray) -> float:
        d = d[d != 0]
        if len(d) == 0:
            return 1.0
        obs = d.mean()
        flips = rng.choice([-1, 1], size=(B_PERM, len(d)))
        null = (flips * np.abs(d)).mean(axis=1)
        return float((np.abs(null) >= abs(obs)).mean())

    tests = []
    for m in sorted(long.model.unique()):
        sub = long[long.model == m]
        for name, a, b in CONTRASTS:
            piv = sub[sub.cond.isin([a, b])].pivot_table(
                index="key", columns="cond", values="correct")
            piv = piv.dropna()
            if len(piv) < 50 or a not in piv or b not in piv:
                continue
            d = (piv[a].astype(float) - piv[b].astype(float)).values
            tests.append((m, name, d.mean() * 100, signflip_p(d)))

    df = pd.DataFrame(tests, columns=["model", "contrast", "delta_pp", "p"])
    df = df.sort_values("p").reset_index(drop=True)
    m_tests = len(df)
    df["bh_crit"] = (df.index + 1) / m_tests * Q_FDR
    k = df[df.p <= df.bh_crit].index.max()
    df["sig_fdr"] = df.index <= k if pd.notna(k) else False
    df["sig_raw"] = df.p < 0.05
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX appendix table
# ─────────────────────────────────────────────────────────────────────────────
CONTRAST_HEADS = [
    ("identity (obj>wrong)",    "identity",  "obj $>$ wrong"),
    ("pixels (obj>random)",     "pixels",    "obj $>$ random"),
    ("ring (ring>obj)",         "ring",      "ring $>$ obj"),
    ("dist barrel (full>k015)", "barrel",    "full $>$ $k{=}.15$"),
    ("dist pin (full>pin015)",  "pincushion", "full $>$ $k{=}{-}.15$"),
]


def fmt_p(p: float) -> str:
    return "$<$.0001" if p < 1e-4 else f"{p:.4f}".lstrip("0").replace("0.", ".")


def table_fdr(df: pd.DataFrame) -> str:
    idx = df.set_index(["model", "contrast"])
    rows = []
    rows.append("\\begin{table*}[t]")
    rows.append("\\centering\\small")
    rows.append("\\begin{tabular}{l|" + "rc|" * 4 + "rc}")
    rows.append("\\toprule")
    rows.append(" & " + " & ".join(
        f"\\multicolumn{{2}}{{c{'|' if i < 4 else ''}}}{{{h}}}"
        for i, (_, h, _) in enumerate(CONTRAST_HEADS)) + " \\\\")
    rows.append("Model & " + " & ".join(
        f"\\multicolumn{{2}}{{c{'|' if i < 4 else ''}}}{{\\scriptsize {d}}}"
        for i, (_, _, d) in enumerate(CONTRAST_HEADS)) + " \\\\")
    rows.append(" & " + " & ".join(
        ["$\\Delta$pp & {\\scriptsize $p$}"] * 5) + " \\\\")
    rows.append("\\midrule")
    for slug, label in MODELS:
        cells = []
        for cname, _, _ in CONTRAST_HEADS:
            r = idx.loc[(slug, cname)]
            d_s = f"${r.delta_pp:+.1f}$"
            p_s = "{\\scriptsize " + fmt_p(r.p) + "}"
            if r.sig_fdr:
                d_s = f"$\\mathbf{{{r.delta_pp:+.1f}}}$"
                p_s = "{\\scriptsize \\textbf{" + fmt_p(r.p) + "}}"
            cells += [d_s, p_s]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    n_raw = int(df.sig_raw.sum())
    n_fdr = int(df.sig_fdr.sum())
    rows.append(
        "\\caption{\\textbf{Multiple-comparison-corrected channel tests.} "
        "Per-model paired sign-flip permutation tests on the per-row "
        "correct@10\\% indicator difference ($B{=}10^4$ permutations, "
        "two-sided) for the five intervention channels, with "
        "Benjamini--Hochberg correction at $q{=}0.05$ applied across the "
        "full grid of $60$ model$\\times$channel tests. $\\Delta$pp is the "
        "paired accuracy difference in the claimed direction (identity: "
        "object\\_only $-$ wrong\\_object; pixels: object\\_only $-$ "
        "random\\_mask; ring: object\\_ring $-$ object\\_only; distortion: "
        "full $-$ distorted, so positive means the distortion hurts). "
        "\\textbf{Bold} marks tests that survive FDR correction: the "
        f"${n_raw}$ raw $p<.05$ discoveries reduce to ${n_fdr}$ "
        "(identity $8/12$, target pixels $4/12$ --- exactly the top four "
        "models by ITW accuracy, local context $2/12$, lens distortion $0/12$). "
        "The identity channel is the most robust image-side effect; the "
        "geometry null is clean after correction.}")
    rows.append("\\label{tab:fdr}")
    rows.append("\\end{table*}")
    return "\n".join(rows) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("building long table...")
    long = build_long()
    n_cells = long.groupby(["model", "cond"]).size()
    assert n_cells.shape[0] == len(MODELS) * len(CONDITIONS), \
        f"expected {len(MODELS) * len(CONDITIONS)} (model, cond) cells, got {n_cells.shape[0]}"

    print("running 60 sign-flip tests + BH...")
    df = run_tests(long)
    assert len(df) == 60, f"expected 60 tests, got {len(df)}"

    # ── verify against the original rebuttal CSV before touching anything ──
    ref_path = CSV_BACKUP if CSV_BACKUP.exists() else CSV_OUT
    ref = pd.read_csv(ref_path)
    surv_new = set(map(tuple, df[df.sig_fdr][["model", "contrast"]].values))
    surv_ref = set(map(tuple, ref[ref.sig_fdr][["model", "contrast"]].values))
    j = df.merge(ref, on=["model", "contrast"], suffixes=("_new", "_ref"))
    assert len(j) == 60, "model/contrast sets differ from the original CSV"
    max_dd = (j.delta_pp_new - j.delta_pp_ref).abs().max()
    max_dp = (j.p_new - j.p_ref).abs().max()

    print(f"\nverification vs {ref_path.name}:")
    print(f"  max |delta_pp| discrepancy: {max_dd:.6f} pp")
    print(f"  max |p| discrepancy:        {max_dp:.6f}")
    print(f"  survivors new={len(surv_new)} ref={len(surv_ref)} "
          f"identical={surv_new == surv_ref}")

    if surv_new != surv_ref or max_dd > 0.1:
        print("\nMISMATCH — refusing to overwrite. Differences:")
        for t in sorted(surv_ref - surv_new):
            print(f"  lost survivor:  {t}")
        for t in sorted(surv_new - surv_ref):
            print(f"  extra survivor: {t}")
        bad = j[(j.delta_pp_new - j.delta_pp_ref).abs() > 0.1]
        for _, r in bad.iterrows():
            print(f"  delta mismatch: {r.model} | {r.contrast}: "
                  f"new {r.delta_pp_new:+.3f} vs ref {r.delta_pp_ref:+.3f}")
        sys.exit(1)

    # ── outputs ────────────────────────────────────────────────────────────
    if not CSV_BACKUP.exists():
        shutil.copy2(CSV_OUT, CSV_BACKUP)
        print(f"\nbacked up original -> {CSV_BACKUP}")
    df.to_csv(CSV_OUT, index=False)
    print(f"wrote {CSV_OUT}")
    TEX_OUT.write_text(table_fdr(df))
    print(f"wrote {TEX_OUT}")

    # ── summary ────────────────────────────────────────────────────────────
    print(f"\ntotal tests: {len(df)} | raw p<.05: {int(df.sig_raw.sum())} | "
          f"survive BH-FDR q={Q_FDR}: {int(df.sig_fdr.sum())}")
    print("per-contrast survivors (claimed direction, delta_pp > 0):")
    for name, _, _ in CONTRASTS:
        s = df[df.contrast == name]
        raw = int(((s.sig_raw) & (s.delta_pp > 0)).sum())
        fdr = int(((s.sig_fdr) & (s.delta_pp > 0)).sum())
        who = sorted(s[(s.sig_fdr) & (s.delta_pp > 0)].model)
        print(f"  {name:28s} raw {raw:2d}/12 -> FDR {fdr:2d}/12  {who}")


if __name__ == "__main__":
    main()
