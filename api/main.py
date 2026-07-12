"""
xG Model API
============
Serves a logistic-regression Expected Goals model trained on 8,049 real
non-penalty shots from 334 international matches (StatsBomb open data:
World Cup 2018/2022, Euro 2020, Women's World Cup 2019, AFCON 2023).
Held out entirely from training: UEFA Euro 2024, used as the test set.

Run:
    uvicorn main:app --reload

Docs:
    http://localhost:8000/docs
"""
import json
import os
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from geometry import dist, shot_angle, freeze_frame_features, GOAL_X, GOAL_Y_CENTER

app = FastAPI(
    title="Expected Goals (xG) API",
    description="Predicts the probability a shot results in a goal, given shot location, technique, and defensive context.",
    version="1.0.0",
)

HERE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(HERE, "xg_model.joblib"))
with open(os.path.join(HERE, "..", "models", "metrics.json")) as f:
    MODEL_INFO = json.load(f)


class Opponent(BaseModel):
    x: float = Field(..., ge=0, le=120)
    y: float = Field(..., ge=0, le=80)
    position_name: Optional[str] = Field(None, description="e.g. 'Goalkeeper' - used to identify the keeper")


class ShotRequest(BaseModel):
    x: float = Field(..., ge=0, le=120, description="Shot location, StatsBomb pitch coords (0-120 long axis, goal at x=120)")
    y: float = Field(..., ge=0, le=80, description="Shot location, StatsBomb pitch coords (0-80 short axis, goal center y=40)")
    body_part: str = Field(..., description="'Right Foot' | 'Left Foot' | 'Head' | 'Other'")
    technique: str = Field("Normal", description="'Normal' | 'Volley' | 'Half Volley' | 'Lob' | 'Overhead Kick' | 'Backheel' | 'Diving Header'")
    shot_type: str = Field("Open Play", description="'Open Play' | 'Free Kick' | 'Corner'  (penalties are out of scope for this model)")
    play_pattern: str = Field("Regular Play")
    under_pressure: bool = False
    first_time: bool = False
    open_goal: bool = False
    deflected: bool = False
    aerial_won: bool = False
    follows_dribble: bool = False
    minute: int = Field(45, ge=0, le=130)
    score_diff_before_shot: int = Field(0, description="Shooting team's goal difference at the moment of the shot")
    opponents: Optional[list[Opponent]] = Field(
        None, description="Defender/GK positions at the moment of the shot (StatsBomb 360 freeze frame). Omit if unavailable - the model falls back to geometry-only features."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "x": 108.5, "y": 42.0,
                "body_part": "Right Foot", "technique": "Normal", "shot_type": "Open Play",
                "play_pattern": "Regular Play", "under_pressure": True, "first_time": False,
                "minute": 67, "score_diff_before_shot": 0,
                "opponents": [
                    {"x": 116.0, "y": 40.0, "position_name": "Goalkeeper"},
                    {"x": 112.0, "y": 38.5, "position_name": "Center Back"}
                ]
            }
        }


class ShotResponse(BaseModel):
    xg: float
    distance_to_goal: float
    shot_angle_degrees: float
    model_version: str = "logreg_v1"


@app.get("/")
def root():
    return {
        "service": "xG Model API",
        "docs": "/docs",
        "model_performance": MODEL_INFO["results"]["Logistic Regression (full features)"],
        "benchmark_statsbomb_xg": MODEL_INFO["results"]["StatsBomb proprietary xG (benchmark)"],
        "test_set": MODEL_INFO["test_tournament"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ShotResponse)
def predict(shot: ShotRequest):
    if shot.shot_type == "Penalty":
        raise HTTPException(400, "Penalties are out of scope for this model (near-constant ~76% conversion; model separately in practice).")

    distance = dist(shot.x, shot.y, GOAL_X, GOAL_Y_CENTER)
    angle = shot_angle(shot.x, shot.y)

    opponents = [{"x": o.x, "y": o.y, "position_name": o.position_name} for o in (shot.opponents or [])]
    ff_feats = freeze_frame_features(shot.x, shot.y, opponents)

    row = {
        "distance_to_goal": distance,
        "shot_angle_rad": angle,
        "minute": shot.minute,
        "score_diff_before_shot": shot.score_diff_before_shot,
        "n_opponents_in_cone": ff_feats["n_opponents_in_cone"],
        "n_opponents_total": ff_feats["n_opponents_total"],
        "nearest_opponent_dist": ff_feats["nearest_opponent_dist"],
        "gk_dist_to_goal_center": ff_feats["gk_dist_to_goal_center"],
        "gk_dist_from_shot_line": ff_feats["gk_dist_from_shot_line"],
        "body_part": shot.body_part,
        "technique": shot.technique,
        "shot_type": shot.shot_type,
        "play_pattern": shot.play_pattern,
        "under_pressure": shot.under_pressure,
        "first_time": shot.first_time,
        "open_goal": shot.open_goal,
        "deflected": shot.deflected,
        "aerial_won": shot.aerial_won,
        "follows_dribble": shot.follows_dribble,
    }

    X = pd.DataFrame([row])
    xg = float(model.predict_proba(X)[0, 1])

    return ShotResponse(
        xg=round(xg, 4),
        distance_to_goal=round(distance, 2),
        shot_angle_degrees=round(angle * 57.29578, 1),
    )
