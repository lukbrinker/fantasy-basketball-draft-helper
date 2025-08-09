from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from ..analytics.adp_utils import add_adp_return_odds
from ..analytics.espn_points import add_espn_points_columns
from ..analytics.pos_scarcity import add_positional_scarcity
from ..analytics.replacement import compute_vorp
from ..analytics.risk import add_risk_percentiles
from ..config import get_config
from ..data_ingest import assemble_player_table, load_all_sources
from ..transform.projections import add_per_game_and_per36, blend_projections
from ..utils.core import ensure_dir, safe_zscore
from ..utils.logging import setup_logging
from .tiers import compute_tiers

logger = logging.getLogger(__name__)


def compute_dvs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Z_FPPG"] = safe_zscore(df["FPPG"])
    df["Z_VORP"] = safe_zscore(df["VORP"])
    df["Z_SCARCITY"] = safe_zscore(df.get("POS_SCARCITY", pd.Series(0.0, index=df.index)))
    df["Z_UPSIDE"] = safe_zscore(df.get("UPSIDE", pd.Series(0.0, index=df.index)))
    df["Z_RETURN_ODDS"] = safe_zscore(df.get("RETURN_ODDS", pd.Series(0.0, index=df.index)))
    df["DVS"] = (
        0.5 * df["Z_FPPG"]
        + 0.3 * df["Z_VORP"]
        + 0.1 * df["Z_SCARCITY"]
        + 0.1 * df["Z_UPSIDE"]
        - 0.05 * df["Z_RETURN_ODDS"]
    )
    logger.info("Computed DVS with RETURN_ODDS term")
    return df


def build_big_board() -> pd.DataFrame:
    setup_logging()
    cfg = get_config()
    logger.info("Loading sources and assembling player table")

    sources = load_all_sources(cfg)
    merged = assemble_player_table(sources)

    logger.info("Blending projections and computing ESPN points")
    blended = blend_projections(merged)
    blended = add_per_game_and_per36(blended)
    blended = add_espn_points_columns(blended, gp_col="GP")

    logger.info("Computing VORP and positional scarcity")
    blended = compute_vorp(blended, cfg)
    blended = add_positional_scarcity(blended)

    logger.info("Adding ADP return odds and risk percentiles")
    blended = add_adp_return_odds(blended)
    blended = add_risk_percentiles(blended)

    blended = compute_dvs(blended)
    blended = compute_tiers(blended, n_tiers=7)

    cols = [
        "Player",
        "Team",
        "Pos",
        "PosList",
        "NameKey",
        "GP",
        "FPPG",
        "FP_TOTAL",
        "VORP",
        "POS_SCARCITY",
        "RETURN_ODDS",
        "P30",
        "P50",
        "P70",
        "UPSIDE",
        "DVS",
        "TIER",
        "ADP",
    ]
    out_cols = [c for c in cols if c in blended.columns]
    board = (
        blended[out_cols]
        .sort_values(["DVS", "FPPG"], ascending=[False, False])
        .reset_index(drop=True)
    )
    board.index = board.index + 1
    board["RANK"] = board.index
    board = board[["RANK", *[c for c in board.columns if c != "RANK"]]]
    logger.info("Big Board built with %d players", len(board))
    return board


def export_board(df: pd.DataFrame) -> None:
    base = Path("data/processed")
    ensure_dir(str(base))
    df.to_csv(base / "big_board.csv", index=False)
    (base / "big_board.html").write_text(df.to_html(index=False))
    logger.info("Exported Big Board to CSV and HTML")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    board = build_big_board()
    if args.export:
        export_board(board)
    else:
        print(board.head(20).to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
