from __future__ import annotations

from .ridge import (
    RidgeModel,
    fit_ridge_models,
)
from .ridge import (
    build_temporal_training as build_temporal_training_ridge,
)
from .ridge import (
    predict_next_season as predict_next_season_ridge,
)

try:
    from .gbm import (
        build_temporal_training as build_temporal_training_gbm,
    )
    from .gbm import (
        fit_gbm_models,
        predict_next_season_gbm,
    )
except Exception:  # pragma: no cover - optional
    build_temporal_training_gbm = None  # type: ignore[assignment]
    fit_gbm_models = None  # type: ignore[assignment]
    predict_next_season_gbm = None  # type: ignore[assignment]

__all__ = [
    "RidgeModel",
    "build_temporal_training_ridge",
    "fit_ridge_models",
    "predict_next_season_ridge",
    "build_temporal_training_gbm",
    "fit_gbm_models",
    "predict_next_season_gbm",
]
