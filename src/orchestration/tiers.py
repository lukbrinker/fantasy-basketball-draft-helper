from __future__ import annotations

import numpy as np
import pandas as pd


def compute_tiers(df: pd.DataFrame, n_tiers: int = 7) -> pd.DataFrame:
    df = df.copy()
    n = max(2, min(int(n_tiers), 10))
    score = df[["FPPG", "VORP"]].astype(float).fillna(0.0).sum(axis=1)
    ranks = score.rank(ascending=False, method="first")
    quantiles = np.linspace(0, 1, n + 1)
    bins = [0] + [float(df.shape[0]) * q for q in quantiles[1:]]
    bins = np.unique(np.array(bins))
    tiers = np.digitize(ranks - 1, bins=bins, right=True)
    min_tier = tiers.min() if isinstance(tiers, np.ndarray) else int(tiers)
    df["TIER"] = (tiers - min_tier + 1).astype(int)
    return df


__all__ = ["compute_tiers"]
