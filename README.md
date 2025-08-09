# Fantasy NBA Draft Assistant (ESPN H2H Points)

A production-ready, analytics-first assistant to generate a Big Board and provide live draft guidance for ESPN H2H Points leagues with custom scoring.

## Quickstart

1) Install Python 3.12 (recommended):

```
brew install python@3.12
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

2) Env (optional):

```
cp env.example .env
```

3) Fetch multi-year data (Basketball-Reference per-game + ADP):

```
python -m src.data_ingest fetch-history --years 4
```

4) Build the Big Board:

```
python -m src.big_board --export
```

Outputs:
- `data/processed/big_board.csv`
- `data/processed/big_board.html`

## What’s inside
- Projections: ridge models predict next-season MIN, GP, and per-minute rates from season t → t+1; compose per-game stats.
- Scoring: ESPN FPPG and totals via your weights.
- VORP: replacement by position.
- Scarcity: local marginal drop to next k at the player’s position (dynamic).
- Risk: uses minutes quantiles (if present) to compute P30/P50/P70 of FPPG, else a variance proxy.
- Tiers: quantile tiers on FPPG + VORP.
- DVS: 0.5*Z(FPPG) + 0.3*Z(VORP) + 0.1*Z(Scarcity) + 0.1*Z(Upside).

## Optional: GBM models (advanced)
Install CatBoost to enable GBM (quantile) models for MIN/GP and per-minute rates with archetype shrinkage.

```
pip install catboost
```

GBM is auto-detected during `fetch-history` and falls back to ridge if training fails.

## Live Draft
- CLI:
  - `python -m src.live_draft log-pick "Player Name"`
  - `python -m src.live_draft who-do-i-pick "A, B, C"`
- UI:
  - `streamlit run src/ui_cli/ui.py`

## Project layout (src/)
- ingestion: `ingestion/web.py`
- transform: `transform/normalize.py`, `transform/projections.py`
- analytics: `analytics/espn_points.py`, `analytics/replacement.py`, `analytics/pos_scarcity.py`, `analytics/risk.py`, `analytics/adp_utils.py`
- models: `models/ridge.py`, `models/gbm.py`, `models/__init__.py`
- orchestration: `orchestration/tiers.py`, `orchestration/big_board.py`
- utils: `utils/core.py`, `utils/logging.py`
- ui/cli: `ui_cli/ui.py`, `live_draft.py`
- entries: `big_board.py` (shim to orchestration), `data_ingest.py`
