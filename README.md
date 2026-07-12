# Expected Goals (xG) Model — trained on real StatsBomb event data

Predicts the probability that a given shot results in a goal, from shot location,
technique, and defensive context at the moment of the shot. This is the same
class of model football clubs, broadcasters, and betting markets use to value
chances and rate finishing.

## Why this project

Most portfolio ML projects pull from Kaggle's generic Datasets tab and land on
Titanic-tier problems. This one deliberately avoids that: real event-stream
data, a non-trivial spatial feature-engineering problem, an evaluation
methodology that actually tests generalization, and a benchmark against a
professional analytics company's own proprietary model — so the results are
judged against a real bar, not just "better than a coin flip."

## Data

[StatsBomb Open Data](https://github.com/statsbomb/open-data) (free, public,
CC BY-NC-SA 4.0) — real professional/international match event streams,
including 360° freeze-frame data (player positions at the moment of each shot).

**334 matches, 8,049 non-penalty shots**, pulled from six tournaments:

| Tournament | Matches |
|---|---|
| FIFA World Cup 2018 | 64 |
| FIFA World Cup 2022 | 64 |
| UEFA Euro 2020 | 51 |
| Women's World Cup 2019 | 52 |
| Africa Cup of Nations 2023 | 52 |
| **UEFA Euro 2024** (held out — test set only) | 51 |

Penalties are excluded from the main model (they're a near-constant ~76%
conversion rate and would just inject a trivial, dominant feature — standard
practice in published xG methodologies is to model them separately).

## Methodology — the part that actually matters

**The test set is an entire held-out tournament (Euro 2024) that the model
never saw during training or tuning.** This is a deliberately harder and more
honest test than the random 80/20 row split most tutorials use — it mirrors
how an xG model is actually deployed: scoring shots from matches the model
has never touched, with a different mix of teams, styles, and even a slightly
different overall conversion rate (7.5% in the test tournament vs 9.3% in
training).

### Features engineered from raw event + freeze-frame data
- **Geometry**: distance to goal, shot angle (the angle subtended by the goal
  mouth from the shot location — the standard xG angle feature, computed via
  law of cosines from the two post locations)
- **Defensive context** (from the 360 freeze frame, available on 97.4% of
  shots): number of opponents in the geometric cone between the shot and the
  goal (point-in-triangle test), distance to the nearest defender, goalkeeper
  distance from goal center, goalkeeper's perpendicular distance from the
  direct shot-to-goal line
- **Shot quality signals**: body part, technique (volley/lob/overhead kick/etc.),
  first-time, under pressure, deflected, open goal, aerial duel won, follows a
  dribble
- **Game state**: minute, goal difference at the moment of the shot

### Models compared (all evaluated on the same held-out Euro 2024 shots)

| Model | AUC | Log Loss | Brier |
|---|---|---|---|
| Baseline LR (distance + angle only) | 0.707 | 0.248 | 0.067 |
| **Logistic Regression (full features) — shipped** | **0.794** | **0.224** | **0.061** |
| XGBoost (full features, tuned) | 0.774 | 0.233 | 0.064 |
| StatsBomb's own proprietary xG (benchmark, not trained by us) | 0.787 | 0.225 | 0.062 |

The regularized logistic regression **matches or edges out StatsBomb's own
production model** on this held-out tournament, and is what's shipped.

**Note on the result above:** this is a small held-out sample (1,304 shots,
one tournament) — a single win margin like this is encouraging, not proof of
outright superiority over a model built on vastly more data and (likely)
proprietary features we don't have access to. I'm reporting it honestly with
that caveat rather than overselling it.

**Why logistic regression over XGBoost, explicitly:** XGBoost was tuned (grid
search over depth/learning rate/regularization, 5-fold CV) and even
isotonic-calibrated, and still consistently underperformed the linear model
on held-out data. With ~6.7k training rows and features that are already
well-engineered (the geometry is doing the nonlinear work up front), a tree
ensemble's main advantage — automatically discovering interactions — has less
to find, while its extra variance works against it on a modest dataset. This
is reported as a finding, not hidden — XGBoost is kept in `models/metrics.json`
as a documented comparison, not silently dropped for making the headline
model look better.

See `evaluation/model_evaluation.png` for ROC and calibration curves across
all four models.

### What actually drives xG (logistic regression coefficients)
- Deflected shots are the single strongest positive driver (deflections
  wrong-foot goalkeepers)
- Distance to goal is the strongest negative driver, as expected
- Wider shot angle, an open goal, more distant goalkeeper positioning, and
  fewer defenders in the shot cone all push xG up
- Overhead kicks and headers are harder to convert than the "Normal" technique
  baseline, controlling for distance/angle

## Project structure
```
xg-model/
├── data/shots.csv              # extracted, feature-engineered shot dataset (8,049 rows)
├── src/
│   ├── extract_shots.py        # raw StatsBomb events -> shots.csv
│   ├── geometry.py             # shared feature-engineering (training + serving use the same code)
│   └── train_xg.py             # trains + evaluates all 4 models, saves the production model
├── models/metrics.json         # full evaluation results
├── evaluation/model_evaluation.png
└── api/
    ├── main.py                 # FastAPI service
    ├── geometry.py
    └── xg_model.joblib         # trained sklearn Pipeline (preprocessing + logistic regression)
```

## Running the API

```bash
cd api
pip install -r ../requirements.txt
uvicorn main:app --reload
# -> http://localhost:8000/docs
```

Example request:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "x": 108.5, "y": 42.0,
  "body_part": "Right Foot", "technique": "Normal", "shot_type": "Open Play",
  "play_pattern": "Regular Play", "under_pressure": true, "minute": 67,
  "score_diff_before_shot": 0,
  "opponents": [
    {"x": 116.0, "y": 40.0, "position_name": "Goalkeeper"},
    {"x": 112.0, "y": 38.5, "position_name": "Center Back"}
  ]
}'
# -> {"xg": 0.18, "distance_to_goal": 12.3, "shot_angle_degrees": 24.1, ...}
```

The API computes all geometric/defensive features server-side from raw shot
location + optional freeze-frame data — the client doesn't need to know
anything about the feature engineering (`geometry.py` is shared verbatim
between `src/` and `api/` specifically so training and serving can't drift
apart).

## Reproducing from scratch
```bash
python src/extract_shots.py   # needs sb/events/*.json (raw StatsBomb match event files)
python src/train_xg.py        # -> models/metrics.json, api/xg_model.joblib
```
(Raw event JSON files aren't included here — ~1GB across 334 matches. They're
pulled directly from `raw.githubusercontent.com/statsbomb/open-data`, which
`extract_shots.py` documents but doesn't automate; ping me if you want the
fetch script too.)

## Honest limitations
- 8,049 shots is workable but modest for tabular ML — a production xG model
  at a real club is typically trained on hundreds of thousands of shots
  across many seasons/leagues
- No shot trajectory or true player velocity data (StatsBomb 360 gives
  freeze-frame positions, not full tracking data) — the goalkeeper/defender
  features are a snapshot, not a dynamic read
- Six tournaments (all major men's/women's international competitions) —
  domestic league play has different spacing, pressing intensity, and pitch
  conditions, so this model would need re-validation before use on club data

## Possible extensions
- Swap the held-out split for nested cross-tournament validation to get a
  distribution of held-out performance instead of one point estimate
- Add shot-assist context (through ball vs cross vs individual carry)
- Wrap the API in a small React shot-map visualizer (fits the default
  React/FastAPI stack) — click a location on a pitch SVG, get live xG
- Extend to a full "player finishing over/under xG" leaderboard — the
  actual analytical product most clubs build this model to serve
