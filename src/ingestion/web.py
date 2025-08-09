from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ..transform.normalize import normalize_dataframe, normalize_player_name

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.0 Safari/605.1.15"
    )
}


def _collapse_team_splits(df: pd.DataFrame) -> pd.DataFrame:
    if "Tm" not in df.columns:
        return df
    df = df.copy()
    df["_is_tot"] = (df["Tm"].astype(str) == "TOT").astype(int)
    df = df.sort_values(["Player", "_is_tot"], ascending=[True, False])
    df = df.drop_duplicates(subset=["Player"], keep="first")
    df = df.drop(columns=["_is_tot"])
    return df


def fetch_bref_per_game(season: int, sleep_s: float = 0.8) -> pd.DataFrame | None:
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_per_game.html"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception:
        return None
    soup = BeautifulSoup(resp.content, "lxml")
    table = soup.find("table", {"id": "per_game_stats"})
    if table is None:
        return None
    html = str(table)
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        return None
    raw = tables[0]
    raw = raw[raw["Player"] != "Player"].copy()
    raw = raw.reset_index(drop=True)
    raw = _collapse_team_splits(raw)
    col_map = {
        "FG": "FGM",
        "FGA": "FGA",
        "FT": "FTM",
        "FTA": "FTA",
        "3P": "3PM",
        "TRB": "REB",
        "AST": "AST",
        "STL": "STL",
        "BLK": "BLK",
        "TOV": "TO",
        "PTS": "PTS",
        "MP": "MIN",
        "G": "GP",
        "Tm": "Team",
        "Pos": "Pos",
    }
    use_cols = [c for c in col_map.keys() if c in raw.columns]
    df = raw[["Player", *use_cols]].rename(columns=col_map)
    if "Team" not in df.columns:
        df["Team"] = ""
    if "Pos" not in df.columns:
        df["Pos"] = ""
    df = normalize_dataframe(df)
    stat_cols = [
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
        "GP",
        "MIN",
    ]
    for c in stat_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    time.sleep(sleep_s)
    return df


def fetch_fantasypros_adp() -> pd.DataFrame | None:
    url = "https://www.fantasypros.com/nba/adp/overall.php"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        html = resp.content.decode("utf-8", errors="ignore")
        tables = pd.read_html(io.StringIO(html))
        for t in tables:
            if {"Player", "ADP"}.issubset(set(t.columns)):
                out = t[["Player", "ADP"]].copy()
                out["Player"] = out["Player"].astype(str).map(normalize_player_name)
                out["ADP"] = pd.to_numeric(out["ADP"], errors="coerce")
                return out
    except Exception:
        return None
    return None


def build_projections_from_history(
    history: dict[int, pd.DataFrame], weights: dict[int, float] | None = None
) -> pd.DataFrame:
    if not history:
        raise ValueError("history is empty")

    seasons = sorted(history.keys(), reverse=True)
    if weights is None:
        base = [0.6, 0.3, 0.1, 0.0, 0.0]
        w = base[: len(seasons)]
        s = sum(w)
        weights = {yr: w[i] / s for i, yr in enumerate(seasons)}

    latest = history[seasons[0]][
        [c for c in ["Player", "Team", "Pos", "PosList"] if c in history[seasons[0]].columns]
    ].copy()
    if "Team" not in latest.columns:
        latest["Team"] = ""
    if "Pos" not in latest.columns:
        latest["Pos"] = ""
    if "PosList" not in latest.columns:
        latest["PosList"] = latest["Pos"].astype(str).str.upper().str.split("/")

    stat_cols = [
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
        "GP",
        "MIN",
    ]
    all_players = latest[["Player"]].drop_duplicates()

    accum = None
    for yr in seasons:
        df = history[yr][["Player", *[c for c in stat_cols if c in history[yr].columns]]].copy()
        for c in stat_cols:
            if c not in df.columns:
                df[c] = 0.0
        df = all_players.merge(df, on="Player", how="left").fillna(0.0)
        w = float(weights.get(yr, 0.0))
        contrib = df[stat_cols].astype(float) * w
        accum = contrib if accum is None else (accum + contrib)

    proj = pd.concat([all_players, accum], axis=1)
    proj = latest.merge(proj, on="Player", how="left")
    return proj


def cache_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


__all__ = [
    "fetch_bref_per_game",
    "fetch_fantasypros_adp",
    "build_projections_from_history",
    "cache_to_csv",
]
