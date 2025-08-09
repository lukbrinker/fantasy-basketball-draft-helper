from __future__ import annotations

from collections import Counter

import pandas as pd

from .config import LeagueConfig, get_config


def compute_roster_needs(
    draft_state: dict[str, list] | None, config: LeagueConfig | None = None
) -> dict[str, int]:
    cfg = config or get_config()
    if not draft_state:
        return {
            p: int(n) for p, n in cfg.roster_slots.items() if p in {"PG", "SG", "SF", "PF", "C"}
        }
    filled: Counter[str] = Counter(draft_state.get("my_roster_positions", []))
    needs = {}
    for pos in ["PG", "SG", "SF", "PF", "C"]:
        needs[pos] = max(int(cfg.roster_slots.get(pos, 0)) - int(filled.get(pos, 0)), 0)
    return needs


def add_roster_fit_boost(df: pd.DataFrame, roster_needs: dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    total_needed = max(1, sum(int(v) for v in roster_needs.values()))

    def _fit(row: pd.Series) -> float:
        positions = row.get("PosList", []) or []
        score = sum(float(roster_needs.get(p, 0)) for p in positions)
        return float(score) / float(total_needed)

    df["ROSTER_FIT"] = df.apply(_fit, axis=1)
    return df


__all__ = ["compute_roster_needs", "add_roster_fit_boost"]
