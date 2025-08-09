from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path so `import src` works when running pytest
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.analytics.espn_points import add_espn_points_columns, compute_fantasy_points_row
from src.analytics.replacement import compute_vorp, replacement_rank


def test_replacement_rank():
    assert replacement_rank(12, 2) == 26
    assert replacement_rank(10, 1) == 11


def test_compute_fppg_row():
    s = pd.Series(
        {
            "FGM": 10,
            "FGA": 20,
            "FTM": 5,
            "FTA": 6,
            "3PM": 3,
            "REB": 8,
            "AST": 7,
            "STL": 2,
            "BLK": 1,
            "TO": 4,
            "PTS": 28,
        }
    )
    fp = compute_fantasy_points_row(s)
    assert abs(fp - 56.0) < 1e-6


def test_add_espn_points_and_vorp():
    df = pd.DataFrame(
        [
            {
                "Player": "A",
                "Pos": "PG",
                "PosList": ["PG"],
                "GP": 70,
                "FGM": 8,
                "FGA": 16,
                "FTM": 3,
                "FTA": 4,
                "3PM": 3,
                "REB": 4,
                "AST": 8,
                "STL": 1.5,
                "BLK": 0.3,
                "TO": 3,
                "PTS": 22,
            },
            {
                "Player": "B",
                "Pos": "PG",
                "PosList": ["PG"],
                "GP": 70,
                "FGM": 6,
                "FGA": 13,
                "FTM": 2,
                "FTA": 2.5,
                "3PM": 2,
                "REB": 3,
                "AST": 6,
                "STL": 1.0,
                "BLK": 0.1,
                "TO": 2,
                "PTS": 16,
            },
            {
                "Player": "C",
                "Pos": "C",
                "PosList": ["C"],
                "GP": 70,
                "FGM": 9,
                "FGA": 15,
                "FTM": 4,
                "FTA": 5,
                "3PM": 0.5,
                "REB": 11,
                "AST": 3,
                "STL": 0.8,
                "BLK": 1.5,
                "TO": 2.5,
                "PTS": 21,
            },
        ]
    )
    df = add_espn_points_columns(df)
    assert "FPPG" in df.columns
    df = compute_vorp(df)
    assert "VORP" in df.columns
