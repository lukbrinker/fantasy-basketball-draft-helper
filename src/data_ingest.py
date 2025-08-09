from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .config import LeagueConfig, get_config
from .ingestion.web import (
    build_projections_from_history,
    cache_to_csv,
    fetch_bref_per_game,
    fetch_fantasypros_adp,
)
from .models import (
    build_temporal_training_gbm,
    build_temporal_training_ridge,
    fit_gbm_models,
    fit_ridge_models,
    predict_next_season_gbm,
    predict_next_season_ridge,
)
from .transform.normalize import normalize_dataframe
from .utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            logger.info("Reading CSV: %s", path)
            return pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - robustness
            logger.warning("Failed to read %s: %s", path, exc)
            return None
    logger.debug("CSV not found: %s", path)
    return None


def load_all_sources(config: LeagueConfig | None = None) -> dict[str, pd.DataFrame | None]:
    cfg = config or get_config()
    base = Path(cfg.data_dir)
    fallback = base / "fallback"

    projections = _safe_read_csv(base / "projections.csv")
    if projections is None:
        logger.info("Using fallback projections")
        projections = _safe_read_csv(fallback / "projections.csv")

    adp = _safe_read_csv(base / "adp.csv")
    if adp is None:
        logger.info("Using fallback ADP")
        adp = _safe_read_csv(fallback / "adp.csv")

    last_season = _safe_read_csv(base / "last_season.csv")
    if last_season is None:
        logger.info("Using fallback last season stats")
        last_season = _safe_read_csv(fallback / "last_season.csv")

    if projections is not None:
        projections = normalize_dataframe(projections)
    if adp is not None:
        adp = normalize_dataframe(adp)
    if last_season is not None:
        last_season = normalize_dataframe(last_season)

    return {"projections": projections, "adp": adp, "last_season": last_season}


def assemble_player_table(sources: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    proj = sources.get("projections")
    last = sources.get("last_season")
    adp = sources.get("adp")

    if proj is None and last is None:  # pragma: no cover - ensured by fallbacks
        logger.error("No projection or last season data available")
        raise RuntimeError("No projection or last season data available.")

    base = proj.copy() if proj is not None else last.copy()
    logger.info("Assembling table: base=%s", "projections" if proj is not None else "last_season")

    if last is not None and proj is not None:
        logger.info("Merging last season stats onto projections via NameKey")
        last_cols = [c for c in last.columns if c not in {"Player", "Team", "Pos", "NameKey"}]
        base = base.merge(
            last[["NameKey", *last_cols]], on="NameKey", how="left", suffixes=("", "_LAST")
        )

    if adp is not None and "ADP" in adp.columns:
        logger.info("Merging ADP via NameKey")
        if "ADP" in base.columns:
            mapping = dict(zip(adp["NameKey"], adp["ADP"], strict=False))
            base["ADP"] = base.apply(
                lambda r: r["ADP"] if pd.notna(r.get("ADP")) else mapping.get(r.get("NameKey")),
                axis=1,
            )
        else:
            base = base.merge(adp[["NameKey", "ADP"]], on="NameKey", how="left")

    if "GP" not in base.columns:
        logger.info("GP missing; defaulting to 70")
        base["GP"] = 70

    cols = [
        "Player",
        "Team",
        "Pos",
        "PosList",
        "NameKey",
    ]
    front = [c for c in cols if c in base.columns]
    others = [c for c in base.columns if c not in front]
    base = base[front + others]

    if "NameKey" in base.columns:
        before = len(base)
        base = base.drop_duplicates(subset=["NameKey"], keep="first")
        after = len(base)
        if after < before:
            logger.info("Dropped %d duplicate rows by NameKey", before - after)

    return base


def _catboost_available() -> bool:
    try:
        import catboost  # noqa: F401

        return True
    except Exception:
        return False


def fetch_and_cache_history(years: int = 3) -> None:
    setup_logging()
    cfg = get_config()
    base = Path(cfg.data_dir)
    fallback_dir = base / "fallback"
    base.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching %d years of BRef per-game data", years)
    current_year = pd.Timestamp.today().year
    seasons = list(range(current_year, current_year - years, -1))

    history = {}
    for yr in seasons:
        df = fetch_bref_per_game(yr)
        if df is not None:
            cache_to_csv(df, base / f"bref_per_game_{yr}.csv")
            logger.info("Fetched BRef per-game for season %d: %d players", yr, len(df))
            history[yr] = df
        else:
            logger.warning("Failed to fetch BRef per-game for season %d", yr)
    if not history:
        raise RuntimeError("Failed to fetch any Basketball-Reference data.")

    logger.info("Building naive decay-weighted projections as baseline")
    naive_proj = build_projections_from_history(history)

    if _catboost_available():
        logger.info("CatBoost detected; training GBM quantile models")
        try:
            X, y = build_temporal_training_gbm(history)
            models = fit_gbm_models(X, y)
            gbm_proj = predict_next_season_gbm(history, models)
            cache_to_csv(normalize_dataframe(gbm_proj), base / "projections.csv")
            logger.info("Wrote projections.csv from GBM")
        except Exception as exc:
            logger.warning("GBM training failed: %s; falling back to ridge", exc)
            try:
                train, meta = build_temporal_training_ridge(history)
                r_models = fit_ridge_models(train, meta["feature_cols"].tolist(), alpha=2.0)
                ridge_proj = predict_next_season_ridge(history, r_models)
                cache_to_csv(normalize_dataframe(ridge_proj), base / "projections.csv")
                logger.info("Wrote projections.csv from ridge fallback")
            except Exception as exc2:
                logger.warning("Ridge training failed: %s; using naive projections", exc2)
                cache_to_csv(normalize_dataframe(naive_proj), base / "projections.csv")
                logger.info("Wrote projections.csv from naive blend")
    else:
        logger.info("CatBoost not available; using ridge models")
        try:
            train, meta = build_temporal_training_ridge(history)
            r_models = fit_ridge_models(train, meta["feature_cols"].tolist(), alpha=2.0)
            ridge_proj = predict_next_season_ridge(history, r_models)
            cache_to_csv(normalize_dataframe(ridge_proj), base / "projections.csv")
            logger.info("Wrote projections.csv from ridge")
        except Exception as exc:
            logger.warning("Ridge training failed: %s; using naive projections", exc)
            cache_to_csv(normalize_dataframe(naive_proj), base / "projections.csv")
            logger.info("Wrote projections.csv from naive blend")

    latest_season = max(history.keys())
    cache_to_csv(normalize_dataframe(history[latest_season]), base / "last_season.csv")
    logger.info("Wrote last_season.csv for season %d", latest_season)

    adp = fetch_fantasypros_adp()
    if adp is not None:
        cache_to_csv(normalize_dataframe(adp), base / "adp.csv")
        logger.info("Cached ADP from FantasyPros")
    else:
        fb = _safe_read_csv(fallback_dir / "adp.csv")
        if fb is not None:
            cache_to_csv(normalize_dataframe(fb), base / "adp.csv")
            logger.info("Cached ADP from fallback")
        else:
            logger.warning("No ADP available from web or fallback")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch-history")
    f.add_argument("--years", type=int, default=3)

    args = parser.parse_args()

    if args.cmd == "fetch-history":
        fetch_and_cache_history(args.years)


if __name__ == "__main__":  # pragma: no cover
    main()
