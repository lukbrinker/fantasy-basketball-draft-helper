from __future__ import annotations

import numpy as np
import pandas as pd

from ..analytics.espn_points import add_espn_points_columns
from .normalize import normalize_dataframe

STAT_COLS: list[str] = [
    "FGM",
    "FGA",
    "FTM",
    "FTA",
    "3PM",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TO",
    "PTS",
]


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in STAT_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = (
            df[col]
            .astype(float)
            .fillna(df[col].median(skipna=True) if df[col].notna().any() else 0.0)
        )
    if "GP" not in df.columns:
        df["GP"] = 70
    return df


def blend_projections(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Blend projection sources already merged in columns with suffixes.

    If no suffixed columns exist, just impute and return.
    """
    df = normalize_dataframe(df)

    has_suffix = any(col.endswith("_LAST") for col in df.columns)
    if not has_suffix:
        return _impute_missing(df)

    base_cols = {c for c in df.columns if not c.endswith("_LAST")}
    out = df[list(base_cols)].copy()

    src_weights = weights or {"CURR": 0.7, "LAST": 0.3}

    for col in STAT_COLS + ["GP"]:
        curr = df[col] if col in df.columns else pd.Series(0.0, index=df.index)
        last = df[f"{col}_LAST"] if f"{col}_LAST" in df.columns else pd.Series(0.0, index=df.index)
        out[col] = src_weights.get("CURR", 0.7) * curr.astype(float).fillna(0.0) + src_weights.get(
            "LAST", 0.3
        ) * last.astype(float).fillna(0.0)

    out = _impute_missing(out)
    out = add_espn_points_columns(out)
    return out


def add_per_game_and_per36(df: pd.DataFrame, minutes_col: str = "MIN") -> pd.DataFrame:
    df = df.copy()
    for col in STAT_COLS:
        df[f"{col}_PG"] = df[col].astype(float)
    if minutes_col in df.columns:
        min_col = df[minutes_col].astype(float).replace(0, np.nan)
        for col in STAT_COLS:
            df[f"{col}_P36"] = df[col].astype(float) / min_col * 36.0
    return df


__all__ = ["blend_projections", "add_per_game_and_per36", "STAT_COLS"]
