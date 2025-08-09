from __future__ import annotations

import re
import unicodedata

import pandas as pd


def normalize_player_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", (name or "")).strip()
    return cleaned


def make_name_key(name: str) -> str:
    if not name:
        return ""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(ch for ch in n if not unicodedata.combining(ch))
    n = n.casefold().strip()
    n = re.sub(r"[^a-z0-9]+", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def split_positions(pos: str) -> list[str]:
    if not pos:
        return []
    tokens = re.split(r"[,/\s]+", pos)
    return [t.upper() for t in tokens if t]


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Player" in df.columns:
        df["Player"] = df["Player"].astype(str).map(normalize_player_name)
        df["NameKey"] = df["Player"].map(make_name_key)
    if "Pos" in df.columns:
        df["PosList"] = df["Pos"].astype(str).map(split_positions)
    return df


__all__ = ["normalize_player_name", "make_name_key", "split_positions", "normalize_dataframe"]
