from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.core import percentile_bounds
from .espn_points import SCORING, compute_fantasy_points_row


def _scale_stats_for_minutes(row: pd.Series, new_min: float) -> pd.Series:
    cur_min = float(row.get("MIN", 0.0))
    if cur_min <= 0 or new_min <= 0:
        return row
    scale = float(new_min) / cur_min
    stats = row.copy()
    for col in ["FGM", "FGA", "FTM", "FTA", "3PM", "REB", "AST", "STL", "BLK", "TO", "PTS"]:
        if col in stats:
            stats[col] = float(stats[col]) * scale
    return stats


def estimate_std_from_proxies(df: pd.DataFrame) -> pd.Series:
    to = df.get("TO", pd.Series(1.5, index=df.index)).astype(float).fillna(1.5)
    stl = df.get("STL", pd.Series(1.0, index=df.index)).astype(float).fillna(1.0)
    blk = df.get("BLK", pd.Series(1.0, index=df.index)).astype(float).fillna(1.0)
    gp = df.get("GP", pd.Series(60, index=df.index)).astype(float).replace(0, np.nan).fillna(60)
    base_std = 0.15 * to + 0.1 * (stl + blk)
    std = base_std / np.sqrt(np.clip(gp, 1.0, None))
    return std.clip(lower=0.5, upper=6.0)


def add_risk_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"MIN_P30", "MIN_P50", "MIN_P70"}.issubset(set(df.columns)):
        p30, p50, p70 = [], [], []
        for _, row in df.iterrows():
            s30 = _scale_stats_for_minutes(row, float(row.get("MIN_P30", row.get("MIN", 0.0))))
            s50 = _scale_stats_for_minutes(row, float(row.get("MIN_P50", row.get("MIN", 0.0))))
            s70 = _scale_stats_for_minutes(row, float(row.get("MIN_P70", row.get("MIN", 0.0))))
            p30.append(compute_fantasy_points_row(s30, SCORING))
            p50.append(compute_fantasy_points_row(s50, SCORING))
            p70.append(compute_fantasy_points_row(s70, SCORING))
        df["P30"] = p30
        df["P50"] = p50
        df["P70"] = p70
        df["UPSIDE"] = df["P70"] - df["P50"]
        return df

    mean = df["FPPG"].astype(float).fillna(0.0)
    std = estimate_std_from_proxies(df)

    p30, p50, p70 = [], [], []
    for m, s in zip(mean, std, strict=False):
        lo, mid, hi = percentile_bounds(float(m), float(s), 0.3, 0.5, 0.7)
        p30.append(lo)
        p50.append(mid)
        p70.append(hi)

    df["P30"] = p30
    df["P50"] = p50
    df["P70"] = p70
    df["UPSIDE"] = df["P70"] - df["P50"]
    return df


__all__ = ["add_risk_percentiles", "estimate_std_from_proxies"]
