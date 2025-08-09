from __future__ import annotations

import math

import pandas as pd

from ..config import LeagueConfig, get_config


def replacement_rank(num_teams: int, starters_at_pos: int) -> int:
    """Compute replacement rank per position.

    Formula: ceil(NUM_TEAMS × starters_at_pos × 1.05)
    """
    return int(math.ceil(num_teams * starters_at_pos * 1.05))


def compute_vorp(df: pd.DataFrame, config: LeagueConfig | None = None) -> pd.DataFrame:
    cfg = config or get_config()
    df = df.copy()

    if "PosList" not in df.columns and "Pos" in df.columns:
        df["PosList"] = df["Pos"].astype(str).str.upper().str.split("/")

    starters_map: dict[str, int] = {
        k: int(v) for k, v in cfg.roster_slots.items() if k in {"PG", "SG", "SF", "PF", "C"}
    }
    pos_to_rank: dict[str, int] = {
        pos: replacement_rank(cfg.num_teams, starters_at_pos)
        for pos, starters_at_pos in starters_map.items()
    }

    df_sorted = df.sort_values("FPPG", ascending=False)

    rep_fppg: dict[str, float] = {}
    for pos, rnk in pos_to_rank.items():
        candidates = df_sorted[
            df_sorted["PosList"].apply(lambda xs, p=pos: p in xs if isinstance(xs, list) else False)
        ]
        if candidates.empty:
            rep_fppg[pos] = 0.0
        else:
            rnk_clamped = min(rnk, len(candidates))
            rep_fppg[pos] = float(candidates.iloc[rnk_clamped - 1]["FPPG"])

    def _player_replacement(row: pd.Series) -> float:
        positions: list[str] = row.get("PosList", []) or []
        if not positions:
            return 0.0
        return float(min(rep_fppg.get(p, 0.0) for p in positions))

    df["REPLACEMENT_FPPG"] = df.apply(_player_replacement, axis=1)
    df["VORP"] = df["FPPG"].astype(float) - df["REPLACEMENT_FPPG"].astype(float)
    return df


__all__ = ["compute_vorp", "replacement_rank"]
