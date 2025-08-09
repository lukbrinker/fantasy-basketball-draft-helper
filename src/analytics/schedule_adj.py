from __future__ import annotations

import pandas as pd

from ..config import LeagueConfig


def apply_playoff_schedule_adjustment(df: pd.DataFrame, config: LeagueConfig) -> pd.DataFrame:
    return df.copy()


__all__ = ["apply_playoff_schedule_adjustment"]
