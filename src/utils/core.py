from __future__ import annotations

import numpy as np
import pandas as pd


def safe_zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - mean) / std


def ensure_dir(path: str) -> None:
    import os

    os.makedirs(path, exist_ok=True)


def top_n_mean(values: pd.Series, n: int) -> float:
    n_clamped = max(1, min(int(n), int(values.shape[0])))
    return float(values.nlargest(n_clamped).mean())


def percentile_bounds(
    mean: float, std: float, p_low: float, p_mid: float, p_high: float
) -> tuple[float, float, float]:
    z_low = float(np.clip(np.sqrt(2) * erfinv(2 * p_low - 1), -3.0, 3.0))
    z_mid = float(np.clip(np.sqrt(2) * erfinv(2 * p_mid - 1), -3.0, 3.0))
    z_high = float(np.clip(np.sqrt(2) * erfinv(2 * p_high - 1), -3.0, 3.0))
    return mean + z_low * std, mean + z_mid * std, mean + z_high * std


def erfinv(x: float) -> float:
    # Approximation by Winitzki for inverse error function
    a = 0.147
    sign = 1.0 if x >= 0 else -1.0
    ln = np.log(1 - x * x)
    t1 = 2 / (np.pi * a) + ln / 2
    inner = t1 * t1 - ln / a
    return sign * np.sqrt(np.sqrt(inner) - t1)


__all__ = ["safe_zscore", "ensure_dir", "top_n_mean", "percentile_bounds", "erfinv"]
