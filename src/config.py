from __future__ import annotations

import json
import os
from collections.abc import Mapping
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError


class ScoringWeights(BaseModel):
    field_goals_made: float = Field(default=2.0, description="FGM weight")
    field_goals_attempted: float = Field(default=-1.0, description="FGA weight")
    free_throws_made: float = Field(default=1.0, description="FTM weight")
    free_throws_attempted: float = Field(default=-1.0, description="FTA weight")
    threes_made: float = Field(default=1.0, description="3PM weight")
    rebounds: float = Field(default=1.0, description="REB weight")
    assists: float = Field(default=2.0, description="AST weight")
    steals: float = Field(default=4.0, description="STL weight")
    blocks: float = Field(default=4.0, description="BLK weight")
    turnovers: float = Field(default=-2.0, description="TO weight")
    points: float = Field(default=1.0, description="PTS weight")

    def to_mapping(self) -> Mapping[str, float]:
        return {
            "FGM": self.field_goals_made,
            "FGA": self.field_goals_attempted,
            "FTM": self.free_throws_made,
            "FTA": self.free_throws_attempted,
            "3PM": self.threes_made,
            "REB": self.rebounds,
            "AST": self.assists,
            "STL": self.steals,
            "BLK": self.blocks,
            "TO": self.turnovers,
            "PTS": self.points,
        }


class LeagueConfig(BaseModel):
    num_teams: int = Field(default=12)
    my_pick_slot: int = Field(default=5)
    roster_slots: dict[str, int] = Field(
        default_factory=lambda: {
            "PG": 2,
            "SG": 2,
            "SF": 2,
            "PF": 2,
            "C": 2,
            "G": 1,
            "F": 1,
            "UTIL": 2,
            "BENCH": 4,
        }
    )
    position_eligibility: dict[str, list[str]] = Field(default_factory=dict)
    playoff_weeks: list[int] = Field(default_factory=lambda: [20, 21, 22])
    data_dir: str = Field(default="data")
    scoring: ScoringWeights = Field(default_factory=ScoringWeights)


def _parse_json_env(value: str | None) -> dict | list | None:
    if value is None or value.strip() == "":
        return None
    try:
        parsed = json.loads(value)
        return parsed
    except json.JSONDecodeError as exc:  # pragma: no cover - guarded
        raise ValueError(f"Invalid JSON in environment variable: {value}") from exc


@lru_cache(maxsize=1)
def get_config() -> LeagueConfig:
    load_dotenv()
    num_teams = int(os.getenv("NUM_TEAMS", "12"))
    my_pick_slot = int(os.getenv("MY_PICK_SLOT", "5"))

    roster_slots_env = os.getenv("ROSTER_SLOTS")
    position_eligibility_env = os.getenv("POSITION_ELIGIBILITY")
    playoff_weeks_env = os.getenv("PLAYOFF_WEEKS")
    data_dir = os.getenv("DATA_DIR", "data")

    roster_slots = _parse_json_env(roster_slots_env) or LeagueConfig().roster_slots
    position_eligibility = _parse_json_env(position_eligibility_env) or {}
    playoff_weeks = _parse_json_env(playoff_weeks_env) or [20, 21, 22]

    try:
        return LeagueConfig(
            num_teams=num_teams,
            my_pick_slot=my_pick_slot,
            roster_slots=roster_slots,
            position_eligibility=position_eligibility,
            playoff_weeks=playoff_weeks,
            data_dir=data_dir,
        )
    except ValidationError as exc:  # pragma: no cover - validated in tests
        raise RuntimeError("Invalid configuration") from exc


__all__ = ["LeagueConfig", "ScoringWeights", "get_config"]
