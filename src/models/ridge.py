from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..transform.projections import STAT_COLS


@dataclass(frozen=True)
class RidgeModel:
    feature_names: list[str]
    weights: np.ndarray  # shape (n_features,)
    intercept: float

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_mat = X[self.feature_names].to_numpy(dtype=float)
        return (X_mat @ self.weights) + self.intercept


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, float]:
    X_mean = X.mean(axis=0, keepdims=True)
    y_mean = y.mean()
    Xc = X - X_mean
    yc = y - y_mean
    n_features = X.shape[1]
    A = Xc.T @ Xc + alpha * np.eye(n_features)
    w = np.linalg.solve(A, Xc.T @ yc)
    b = float(y_mean - (X_mean @ w).item())
    return w, b


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
    return feats


def build_temporal_training(
    history: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    seasons = sorted(history.keys())
    rows: list[pd.Series] = []
    for i in range(len(seasons) - 1):
        yr_t = seasons[i]
        yr_tp1 = seasons[i + 1]
        df_t = history[yr_t]
        df_tp1 = history[yr_tp1]
        left = df_t[["Player", "Team", "Pos", "PosList", "MIN", "GP", *STAT_COLS]].copy()
        right = df_tp1[["Player", "MIN", "GP", *STAT_COLS]].copy()
        merged = left.merge(right, on="Player", how="inner", suffixes=("_t", "_tp1"))
        if merged.empty:
            continue
        feats = _prepare_features(
            merged[["MIN_t", "GP_t", *[f"{c}_t" for c in STAT_COLS]]].rename(
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
        design = pd.concat(
            [
                merged[["Player"]].reset_index(drop=True),
                feats.reset_index(drop=True),
                targets.reset_index(drop=True),
            ],
            axis=1,
        )
        rows.append(design)
    if not rows:
        raise ValueError("Insufficient overlapping seasons for training")
    train = pd.concat(rows, axis=0, ignore_index=True)
    feature_cols = [*([f"{c}_PMIN" for c in STAT_COLS]), "MIN", "GP"]
    return train, {"feature_cols": pd.Series(feature_cols)}


def fit_ridge_models(
    train: pd.DataFrame, feature_cols: list[str], alpha: float = 1.0
) -> dict[str, RidgeModel]:
    X = train[feature_cols].to_numpy(dtype=float)
    models: dict[str, RidgeModel] = {}
    w, b = _ridge_fit(X, train["MIN"].to_numpy(dtype=float), alpha=alpha)
    models["MIN"] = RidgeModel(feature_cols, w, b)
    w, b = _ridge_fit(X, train["GP"].to_numpy(dtype=float), alpha=alpha)
    models["GP"] = RidgeModel(feature_cols, w, b)
    for col in STAT_COLS:
        target = f"{col}_PMIN"
        w, b = _ridge_fit(X, train[target].to_numpy(dtype=float), alpha=alpha)
        models[target] = RidgeModel(feature_cols, w, b)
    return models


def predict_next_season(
    history: dict[int, pd.DataFrame], models: dict[str, RidgeModel]
) -> pd.DataFrame:
    latest_year = max(history.keys())
    latest = history[latest_year]
    feats = _prepare_features(latest[["MIN", "GP", *STAT_COLS]].copy())
    pred_min = models["MIN"].predict(feats)
    pred_gp = models["GP"].predict(feats)
    pred_min = np.clip(pred_min, 0.0, 40.0)
    pred_gp = np.clip(pred_gp, 0.0, 82.0)

    pred = pd.DataFrame({"MIN": pred_min, "GP": pred_gp})
    for col in STAT_COLS:
        rate = models[f"{col}_PMIN"].predict(feats)
        rate = np.clip(rate, 0.0, None)
        pred[col] = rate * pred_min
    out = pd.concat(
        [
            latest[["Player", "Team", "Pos", "PosList"]].copy().reset_index(drop=True),
            pred.reset_index(drop=True),
        ],
        axis=1,
    )
    for c in ["MIN", "GP", *STAT_COLS]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).clip(lower=0.0)
    return out


__all__ = [
    "RidgeModel",
    "build_temporal_training",
    "fit_ridge_models",
    "predict_next_season",
]
