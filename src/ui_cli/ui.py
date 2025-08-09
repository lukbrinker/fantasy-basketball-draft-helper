from __future__ import annotations

import json
from pathlib import Path
import sys
from pathlib import Path as _Path

import pandas as pd
import streamlit as st

# Ensure project root is on sys.path so absolute imports work under Streamlit
_ROOT = str(_Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.orchestration.big_board import build_big_board

STATE_PATH = Path("draft_state.json")


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            return {"picks": []}
    return {"picks": []}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def main() -> None:
    st.set_page_config(page_title="Fantasy NBA Draft Assistant", layout="wide")
    st.title("Fantasy NBA Draft Assistant — ESPN H2H Points")

    state = load_state()

    with st.sidebar:
        st.header("Draft State")
        new_pick = st.text_input("Log pick")
        if st.button("Add pick") and new_pick.strip():
            state.setdefault("picks", []).append(new_pick.strip())
            save_state(state)
            st.experimental_rerun()
        if st.button("Reset picks"):
            state["picks"] = []
            save_state(state)
            st.experimental_rerun()

    st.subheader("Big Board")
    board = build_big_board()
    taken = set(state.get("picks", []))
    available = board[~board["Player"].isin(taken)]

    with st.expander("Filters"):
        max_tier_val = int(available["TIER"].max()) if "TIER" in available.columns else 7
        min_tier, max_tier = st.slider("Tier range", 1, max_tier_val, (1, max_tier_val))
        pos = st.multiselect("Positions", ["PG", "SG", "SF", "PF", "C"], [])
        adp_default = 250.0
        if "ADP" in available.columns:
            adp_series = pd.to_numeric(available["ADP"], errors="coerce")
            if adp_series.notna().any():
                adp_default = float(adp_series.max())
        adp_max = st.number_input("Max ADP", value=adp_default)
        show_cols = st.multiselect(
            "Columns",
            [
                "RANK",
                "Player",
                "Pos",
                "FPPG",
                "VORP",
                "POS_SCARCITY",
                "P50",
                "P70",
                "UPSIDE",
                "DVS",
                "TIER",
                "ADP",
            ],
            default=["RANK", "Player", "Pos", "FPPG", "VORP", "DVS", "TIER"],
        )

    filt = available.copy()
    filt = filt[(filt["TIER"] >= min_tier) & (filt["TIER"] <= max_tier)]
    if pos:
        filt = filt[filt["Pos"].str.contains("|".join(pos))]
    if "ADP" in filt.columns:
        adp_vals = pd.to_numeric(filt["ADP"], errors="coerce")
        # Keep players with missing ADP, and those within the threshold
        filt = filt[(adp_vals.isna()) | (adp_vals <= float(adp_max))]

    st.dataframe(filt[show_cols], use_container_width=True)

    st.subheader("Who do I pick?")
    candidates = st.text_input("Enter candidates (comma-separated)")
    if st.button("Rank candidates") and candidates.strip():
        names = [s.strip() for s in candidates.split(",") if s.strip()]
        focus = available[available["Player"].isin(names)].copy()
        if focus.empty:
            st.info("No candidates matched.")
        else:
            st.dataframe(
                focus.sort_values(["DVS", "FPPG"], ascending=[False, False])[show_cols],
                use_container_width=True,
            )


if __name__ == "__main__":  # pragma: no cover
    main()
