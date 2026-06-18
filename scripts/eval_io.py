"""Shared helper for persisting evaluation results.

Saves two files per run, side by side under `output_dir`:
    <ts>-<kind>.csv         the per-question DataFrame
    <ts>-<kind>.meta.json   the knobs the run used (model, k, embedding,
                            git rev, ...) plus aggregate scores

`ts` = YYYY-MM-DD-HHMMSS in local time. `kind` is "retrieval" or "answers".
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _git_rev() -> str | None:
    try:
        repo_root = Path(__file__).resolve().parent.parent
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _aggregate_numeric(df: pd.DataFrame) -> dict[str, float]:
    """Mean of every numeric column. Skips NaNs."""
    aggs: dict[str, float] = {}
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        if len(series):
            aggs[col] = float(series.mean())
    return aggs


def save_eval_results(
    df: pd.DataFrame,
    kind: str,
    run_meta: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    csv_path = output_dir / f"{ts}-{kind}.csv"
    meta_path = output_dir / f"{ts}-{kind}.meta.json"

    df.to_csv(csv_path, index=False)

    meta = {
        "timestamp": ts,
        "kind": kind,
        "git_rev": _git_rev(),
        "n_questions": len(df),
        "aggregates": _aggregate_numeric(df),
        **run_meta,
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return csv_path, meta_path
