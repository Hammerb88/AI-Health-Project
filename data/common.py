"""Shared helpers for the v3 pipeline: config loading and per-stage logs."""

from __future__ import annotations

import functools
import numbers
from datetime import datetime

import pandas as pd
import yaml

import loaders as L

CONFIG = L.DATA_DIR / "config"
LOGS = L.DERIVED / "logs"
REPORTS = L.DATA_DIR / "reports"

COHORTS = ["breast", "nsclc", "prostate", "crc"]


@functools.lru_cache(maxsize=None)
def config(name: str) -> dict:
    """Load config/<name>.yaml."""
    with open(CONFIG / f"{name}.yaml") as fh:
        return yaml.safe_load(fh)


def pipeline_cfg() -> dict:
    return config("pipeline")


def md_table(df: pd.DataFrame, index: bool = True, floatfmt: str = "{:.3g}") -> str:
    """Render a DataFrame as a GitHub markdown table (no tabulate dependency)."""
    d = df.reset_index() if index else df
    def fmt(v):
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, numbers.Integral):
            return f"{v:,}"
        if isinstance(v, numbers.Real):
            return "" if pd.isna(v) else floatfmt.format(v)
        return "" if v is None or (not isinstance(v, (list, tuple)) and pd.isna(v)) else str(v)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(d[c]) else "---" for c in d.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def write_log(stage: str, summary: str, tables: dict[str, pd.DataFrame] | None = None) -> None:
    """Write data/derived/logs/<stage>.md: one paragraph plus optional tables."""
    LOGS.mkdir(parents=True, exist_ok=True)
    lines = [f"# {stage}", "", f"_Written {datetime.now():%Y-%m-%d %H:%M}_", "", summary.strip(), ""]
    for title, df in (tables or {}).items():
        lines += [f"## {title}", "", md_table(df), ""]
    (LOGS / f"{stage}.md").write_text("\n".join(lines), encoding="utf-8")


def check(cond: bool, msg: str) -> None:
    """Checklist assert (Section 16): fail the stage rather than write bad data."""
    if not cond:
        raise AssertionError(f"CHECKLIST FAILED: {msg}")
    print(f"  [ok] {msg}")
