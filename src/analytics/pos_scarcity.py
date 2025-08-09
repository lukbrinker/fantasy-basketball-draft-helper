from __future__ import annotations

import pandas as pd


def add_positional_scarcity(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    df = df.copy()

    def pos_slope_for_row(row: pd.Series) -> float:
        positions = row.get("PosList", []) or []
        best = 0.0
        for p in positions:
            pool = df[
                df["PosList"].apply(lambda xs, pos=p: pos in xs if isinstance(xs, list) else False)
            ].copy()
            if pool.empty:
                continue
            pool = pool.sort_values("FPPG", ascending=False).reset_index(drop=True)
            idx = pool.index[pool["Player"] == row["Player"]]
            if len(idx) == 0:
                continue
            i = int(idx[0])
            next_k = pool.iloc[i + 1 : i + 1 + max(1, k)]
            if next_k.empty:
                continue
            drop = float(row["FPPG"]) - float(next_k["FPPG"].mean())
            if drop > best:
                best = drop
        return best

    df["POS_SCARCITY"] = df.apply(pos_slope_for_row, axis=1)
    return df


__all__ = ["add_positional_scarcity"]
