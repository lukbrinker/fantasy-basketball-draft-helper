from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ScoringWeights

SCORING = ScoringWeights().to_mapping()


def compute_fantasy_points_row(stats: pd.Series, scoring: dict[str, float] | None = None) -> float:
    """Compute ESPN fantasy points for a single player row.

    Required columns include: FGM,FGA,FTM,FTA,3PM,REB,AST,STL,BLK,TO,PTS.

    Missing values are treated as 0.

    >>> import pandas as pd
    >>> s = pd.Series({
    ...     'FGM': 10, 'FGA': 20, 'FTM': 5, 'FTA': 6, '3PM': 3,
    ...     'REB': 8, 'AST': 7, 'STL': 2, 'BLK': 1, 'TO': 4, 'PTS': 28
    ... })
    >>> compute_fantasy_points_row(s)
    56.0
    >>> compute_fantasy_points_row(pd.Series({}))
    0.0
    """
    weights = SCORING if scoring is None else scoring
    total = 0.0
    for stat, weight in weights.items():
        val = float(stats.get(stat, 0) or 0)
        total += val * weight
    return float(total)


essential_cols = ["FGM", "FGA", "FTM", "FTA", "3PM", "REB", "AST", "STL", "BLK", "TO", "PTS"]


def add_espn_points_columns(df: pd.DataFrame, gp_col: str = "GP") -> pd.DataFrame:
    """Add FPPG and FP_TOTAL columns to the dataframe.

    - FPPG is computed per row using the scoring weights.
    - FP_TOTAL = FPPG * GP (if GP present) else NaN.
    """
    df = df.copy()
    df["FPPG"] = df.apply(compute_fantasy_points_row, axis=1)
    if gp_col in df.columns:
        df["FP_TOTAL"] = df["FPPG"] * df[gp_col].astype(float).fillna(0)
    else:
        df["FP_TOTAL"] = np.nan
    return df


__all__ = ["compute_fantasy_points_row", "add_espn_points_columns", "SCORING"]
