"""Paths and loaders for the public Metric VQA release (bmeivar/MetricVQA).

Everything resolves under METRICVQA_ROOT, the directory the HF dataset was downloaded to:

    $METRICVQA_ROOT/itw/questions.csv     331 in-the-wild queries (question, answer_cm, ...)
    $METRICVQA_ROOT/itw/images/            in-the-wild photos
    $METRICVQA_ROOT/objectron/questions.csv
    $METRICVQA_ROOT/sam_phrases/           written by interventions/masks_from_release.py
    $METRICVQA_ROOT/results/               written by the eval scripts, read by analysis/
    $METRICVQA_ROOT/outputs/{tables,figs}  written by analysis/ and figures/

Scripts work in the legacy (image, query, answer) schema; load_itw() converts to it.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("METRICVQA_ROOT", "."))
ITW_CSV = ROOT / "itw" / "questions.csv"
ITW_IMAGES = ROOT / "itw" / "images"
PHRASES_CSV = ROOT / "sam_phrases" / "sam_phrases_v1.csv"
MASKS_DIR = ROOT / "sam_phrases" / "masks"
RESULTS = ROOT / "results"
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figs"


def to_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """Release schema (question / answer_cm, image = images/<file>) -> image / query / answer."""
    if "answer_cm" in df.columns:
        df = df.rename(columns={"question": "query", "answer_cm": "answer"})
        df["image"] = df["image"].astype(str).str.replace("images/", "", regex=False)
    return df


def load_itw(csv: str | Path | None = None) -> pd.DataFrame:
    """In-the-wild questions as (image, query, answer, ...), image = bare file name."""
    return to_legacy(pd.read_csv(csv or ITW_CSV))


def require(path: Path, what: str) -> Path:
    """Exit with a clear message when an input that is not part of the release is missing."""
    if not Path(path).exists():
        raise SystemExit(f"{what} not found: {path}\n"
                         "It is not part of the public release; produce it first (see README).")
    return Path(path)
