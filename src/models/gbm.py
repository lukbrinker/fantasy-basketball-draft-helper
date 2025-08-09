from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency
    CatBoostRegressor = None  # type: ignore[misc, assignment]

from ..transform.projections import STAT_COLS


def _per_minute_rates(df: pd.DataFrame, min_col: str = "MIN") -> pd.DataFrame:
    out = df.copy()
    min_vals = out.get(min_col, pd.Series(0.0, index=out.index)).astype(float).replace(0, np.nan)
    for col in STAT_COLS:
        if col in out.columns:
            out[f"{col}_PMIN"] = out[col].astype(float) / min_vals
        else:
            out[f"{col}_PMIN"] = 0.0
    out = out.fillna(0.0)
    return out


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    rates = _per_minute_rates(df)
    feats = rates[[f"{c}_PMIN" for c in STAT_COLS]].copy()
    feats["MIN"] = rates.get("MIN", pd.Series(0.0, index=rates.index)).astype(float)
    feats["GP"] = rates.get("GP", pd.Series(0.0, index=rates.index)).astype(float)

    def _bucket(pos_list: list[str] | float) -> str:
        if not isinstance(pos_list, list) or not pos_list:
            return "U"
        first = pos_list[0]
        if first in ("C",):
            return "C"
        if first in ("PF", "SF"):
            return "F"
        if first in ("PG", "SG"):
            return "G"
        return "U"

    feats["POS_BUCKET"] = df.get("PosList", pd.Series([], index=df.index)).map(_bucket)
    return feats


def build_temporal_training(history: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = sorted(history.keys())
    rows: list[pd.DataFrame] = []
    for i in range(len(seasons) - 1):
        yr_t = seasons[i]
        yr_tp1 = seasons[i + 1]
        df_t = history[yr_t]
        df_tp1 = history[yr_tp1]
        left = df_t[["Player", "PosList", "MIN", "GP", *STAT_COLS]].copy()
        right = df_tp1[["Player", "MIN", "GP", *STAT_COLS]].copy()
        merged = left.merge(right, on="Player", how="inner", suffixes=("_t", "_tp1"))
        if merged.empty:
            continue
        feats = _prepare_features(
            merged[["PosList", "MIN_t", "GP_t", *[f"{c}_t" for c in STAT_COLS]]].rename(
                columns={"MIN_t": "MIN", "GP_t": "GP", **{f"{c}_t": c for c in STAT_COLS}}
            )
        )
        targets = pd.DataFrame(
            {
                "MIN": merged["MIN_tp1"].astype(float),
                "GP": merged["GP_tp1"].astype(float),
                **{
                    f"{c}_PMIN": (
                        merged[f"{c}_tp1"].astype(float)
                        / merged["MIN_tp1"].astype(float).replace(0, np.nan)
                    ).fillna(0.0)
                    for c in STAT_COLS
                },
            }
        )
        rows.append(
            pd.concat(
                [
                    merged[["Player"]].reset_index(drop=True),
                    feats.reset_index(drop=True),
                    targets.reset_index(drop=True),
                ],
                axis=1,
            )
        )
    if not rows:
        raise ValueError("Insufficient overlapping seasons for training")
    train = pd.concat(rows, axis=0, ignore_index=True)
    X = train.drop(columns=["Player", *("MIN", "GP", *[f"{c}_PMIN" for c in STAT_COLS])])
    y = train[["MIN", "GP", *[f"{c}_PMIN" for c in STAT_COLS]]]
    return X, y


def _fit_catboost(
    X: pd.DataFrame, y: pd.Series, loss: str, cat_features: list[int] | None = None
) -> CatBoostRegressor:
    assert CatBoostRegressor is not None, "CatBoost not installed"
    model = CatBoostRegressor(
        loss_function=loss,
        depth=6,
        learning_rate=0.08,
        n_estimators=600,
        random_seed=7,
        verbose=False,
    )
    model.fit(X, y, cat_features=cat_features)
    return model


def fit_gbm_models(X: pd.DataFrame, y: pd.DataFrame) -> dict[str, CatBoostRegressor]:
    assert CatBoostRegressor is not None, "CatBoost not installed"
    cat_cols = [i for i, c in enumerate(X.columns) if c == "POS_BUCKET"]
    models: dict[str, CatBoostRegressor] = {}
    for target, alphas in ("MIN", [0.3, 0.5, 0.7]), ("GP", [0.3, 0.5, 0.7]):
        for a in alphas:
            key = f"{target}_P{int(a*100)}"
            models[key] = _fit_catboost(
                X, y[target], loss=f"Quantile:alpha={a}", cat_features=cat_cols
            )
    for c in STAT_COLS:
        key = f"{c}_PMIN"
        models[key] = _fit_catboost(X, y[key], loss="RMSE", cat_features=cat_cols)
    return models


def _pos_bucket(pos_list: list[str] | float) -> str:
    if not isinstance(pos_list, list) or not pos_list:
        return "U"
    first = pos_list[0]
    if first in ("C",):
        return "C"
    if first in ("PF", "SF"):
        return "F"
    if first in ("PG", "SG"):
        return "G"
    return "U"


def _compute_archetype_priors(history: dict[int, pd.DataFrame]) -> dict[str, dict[str, float]]:
    latest = history[max(history.keys())].copy()
    latest["POS_BUCKET"] = latest.get("PosList").map(_pos_bucket)
    rates = _per_minute_rates(latest)
    priors: dict[str, dict[str, float]] = {}
    for bucket, grp in rates.groupby("POS_BUCKET"):
        priors[bucket] = {
            f"{c}_PMIN": float((grp[c] / grp["MIN"]).replace(0, np.nan).fillna(0.0).mean())
            if c in grp.columns
            else 0.0
            for c in STAT_COLS
        }
    return priors


def _shrink_rates(row: pd.Series, priors: dict[str, dict[str, float]]) -> dict[str, float]:
    bucket = _pos_bucket(row.get("PosList", []))
    prior = priors.get(bucket, {f"{c}_PMIN": 0.0 for c in STAT_COLS})
    gp = float(row.get("GP", 50.0))
    lam = max(0.3, min(gp / 82.0, 0.9))
    out = {}
    for c in STAT_COLS:
        key = f"{c}_PMIN"
        pred = float(row.get(key, 0.0))
        out[key] = lam * pred + (1 - lam) * float(prior.get(key, 0.0))
    return out


def predict_next_season_gbm(
    history: dict[int, pd.DataFrame], models: dict[str, CatBoostRegressor]
) -> pd.DataFrame:
    latest_year = max(history.keys())
    latest = history[latest_year]
    X = _prepare_features(latest[["PosList", "MIN", "GP", *STAT_COLS]].copy())
    min_p50 = models["MIN_P50"].predict(X)
    min_p30 = models["MIN_P30"].predict(X)
    min_p70 = models["MIN_P70"].predict(X)
    gp_p50 = models["GP_P50"].predict(X)
    gp_p30 = models["GP_P30"].predict(X)
    gp_p70 = models["GP_P70"].predict(X)

    pred = pd.DataFrame(
        {
            "MIN": np.clip(min_p50, 0.0, 40.0),
            "MIN_P30": np.clip(min_p30, 0.0, 40.0),
            "MIN_P50": np.clip(min_p50, 0.0, 40.0),
            "MIN_P70": np.clip(min_p70, 0.0, 40.0),
            "GP": np.clip(gp_p50, 0.0, 82.0),
            "GP_P30": np.clip(gp_p30, 0.0, 82.0),
            "GP_P50": np.clip(gp_p50, 0.0, 82.0),
            "GP_P70": np.clip(gp_p70, 0.0, 82.0),
        }
    )
    for c in STAT_COLS:
        pred[f"{c}_PMIN"] = models[f"{c}_PMIN"].predict(X)
    priors = _compute_archetype_priors(history)
    shrink = pred.join(latest[["PosList", "GP"]], how="left")
    shrunk = shrink.apply(lambda r: _shrink_rates(r, priors), axis=1, result_type="expand")
    for c in STAT_COLS:
        pred[c] = np.clip(shrunk[f"{c}_PMIN"], 0.0, None) * pred["MIN"]

    out = pd.concat(
        [
            latest[["Player", "Team", "Pos", "PosList"]].copy().reset_index(drop=True),
            pred.reset_index(drop=True),
        ],
        axis=1,
    )
    return out


__all__ = [
    "build_temporal_training",
    "fit_gbm_models",
    "predict_next_season_gbm",
]
