from __future__ import annotations

import math

import pandas as pd

from ..config import get_config


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def add_adp_return_odds(df: pd.DataFrame) -> pd.DataFrame:
    cfg = get_config()
    df = df.copy()
    next_pick = int(cfg.my_pick_slot + cfg.num_teams)
    sigma = 12.0

    def _odds(adp_val: float | None) -> float:
        try:
            mu = float(adp_val)
        except Exception:
            return 0.0
        return float(1.0 - normal_cdf((mu - next_pick) / sigma))

    if "ADP" in df.columns:
        df["RETURN_ODDS"] = df["ADP"].apply(_odds)
    else:
        df["RETURN_ODDS"] = 0.0
    return df


__all__ = ["add_adp_return_odds"]
