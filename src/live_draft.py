from __future__ import annotations

import argparse
import json
from pathlib import Path

from .orchestration.big_board import build_big_board
from .roster_fit import add_roster_fit_boost, compute_roster_needs

STATE_PATH = Path("draft_state.json")


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            return {"picks": [], "my_roster_positions": []}
    return {"picks": [], "my_roster_positions": []}


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def log_pick(player: str) -> None:
    state = _load_state()
    state.setdefault("picks", []).append(player)
    _save_state(state)
    print(f"Logged pick: {player}")


def who_do_i_pick(candidates_csv: str, my_positions_csv: str | None = None, top_k: int = 6) -> None:
    candidates = [c.strip() for c in candidates_csv.split(",") if c.strip()]
    my_positions = [
        p.strip().upper() for p in (my_positions_csv.split(",") if my_positions_csv else [])
    ]

    board = build_big_board()
    # Remove taken players
    state = _load_state()
    taken = set(state.get("picks", []))
    board_available = board[~board["Player"].isin(taken)].copy()

    # Compute roster fit if positions provided
    draft_state = {"my_roster_positions": my_positions}
    needs = compute_roster_needs(draft_state)
    board_available = add_roster_fit_boost(board_available, needs)

    focus = board_available[board_available["Player"].isin(candidates)].copy()
    if focus.empty:
        print("No candidates matched the board.")
        return

    # Composite: DVS + 0.2 * ROSTER_FIT
    focus["LIVE_SCORE"] = focus["DVS"].astype(float) + 0.2 * focus.get("ROSTER_FIT", 0.0)
    focus = focus.sort_values(["LIVE_SCORE", "DVS", "FPPG"], ascending=[False, False, False])

    cols = [
        c
        for c in [
            "Player",
            "Pos",
            "FPPG",
            "VORP",
            "UPSIDE",
            "POS_SCARCITY",
            "DVS",
            "ROSTER_FIT",
            "TIER",
        ]
        if c in focus.columns
    ]
    print(focus[cols].head(top_k).to_string(index=False))

    # Simple rationale
    top = focus.iloc[0]
    print("\nRationale:")
    tier_str = int(top["TIER"]) if "TIER" in top else "-"
    print(
        f"{top['Player']} leads with DVS {top['DVS']:.2f}, "
        f"FPPG {top['FPPG']:.1f}, VORP {top['VORP']:.2f}, "
        f"tier {tier_str}; roster fit {top.get('ROSTER_FIT', 0.0):.2f}."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("log-pick")
    p1.add_argument("player")

    p2 = sub.add_parser("who-do-i-pick")
    p2.add_argument("candidates")
    p2.add_argument("--my-positions", default=None)
    p2.add_argument("--top-k", type=int, default=6)

    args = parser.parse_args()

    if args.cmd == "log-pick":
        log_pick(args.player)
    elif args.cmd == "who-do-i-pick":
        who_do_i_pick(args.candidates, args.my_positions, args.top_k)


if __name__ == "__main__":  # pragma: no cover
    main()
