# ML Class Project: Expected Goals (xG) Model Training & Evaluation
# Name: Saurabh
# Instructor: Dr. Smith (Machine Learning 101)

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
import xgboost as xgb
import joblib

# Load our extracted shot dataset
df = pd.read_csv('sb_out/shots.csv')
# print(df.shape) # debug check size

# split: we hold out the Euro 2024 tournament entirely for testing!
# (Stronger test of generalization than a simple random row split)
TEST_TAG = '55_282'  
train_df = df[df['tournament'] != TEST_TAG].reset_index(drop=True)
test_df = df[df['tournament'] == TEST_TAG].reset_index(drop=True)

print(f"Train shots: {len(train_df)}  ({train_df['is_goal'].mean()*100:.2f}% goals)")
print(f"Test shots:  {len(test_df)}  ({test_df['is_goal'].mean()*100:.2f}% goals)  [Euro 2024, held out tournament]")

# Columns we will feed to the model
NUMERIC = ['distance_to_goal', 'shot_angle_rad', 'minute', 'score_diff_before_shot',
           'n_opponents_in_cone', 'n_opponents_total', 'nearest_opponent_dist',
           'gk_dist_to_goal_center', 'gk_dist_from_shot_line']
CATEGORICAL = ['body_part', 'technique', 'shot_type', 'play_pattern']
BOOLEAN = ['under_pressure', 'first_time', 'open_goal', 'deflected', 'aerial_won', 'follows_dribble']

FEATURES = NUMERIC + CATEGORICAL + BOOLEAN
TARGET = 'is_goal'

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

# ---- Preprocessing Pipelines ----
# Standard scaling for numbers + imputing missing values (like missing GK stats)
numeric_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
])
# One-hot encoding for categorical variables (body part, etc.)
categorical_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])
# Combine everything using ColumnTransformer
preprocess = ColumnTransformer([
    ('num', numeric_pipe, NUMERIC),
    ('cat', categorical_pipe, CATEGORICAL),
    ('bool', 'passthrough', BOOLEAN),
])

results = {}

def evaluate(name, y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    ll = log_loss(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    results[name] = {'auc': auc, 'log_loss': ll, 'brier': brier}
    print(f"{name:35s}  AUC={auc:.4f}  LogLoss={ll:.4f}  Brier={brier:.4f}")

# ---- Model 1: Baseline LR (distance + angle only) ----
# Just using the distance and angle - the classic textbook features
baseline_pipe = Pipeline([
    ('prep', ColumnTransformer([('num', StandardScaler(), ['distance_to_goal', 'shot_angle_rad'])])),
    ('clf', LogisticRegression(max_iter=1000)),
])
baseline_pipe.fit(train_df[['distance_to_goal', 'shot_angle_rad']], y_train)
pred_baseline = baseline_pipe.predict_proba(test_df[['distance_to_goal', 'shot_angle_rad']])[:, 1]
evaluate('Baseline LR (distance+angle only)', y_test, pred_baseline)

# ---- Model 2: Logistic Regression (all features) ----
# Our main pipeline model
logreg_pipe = Pipeline([
    ('prep', preprocess),
    ('clf', LogisticRegression(max_iter=2000)),
])
logreg_pipe.fit(X_train, y_train)
pred_logreg = logreg_pipe.predict_proba(X_test)[:, 1]
evaluate('Logistic Regression (full features)', y_test, pred_logreg)

# ---- Model 3: XGBoost ----
# Let's try tree boosting!
xgb_pipe = Pipeline([
    ('prep', preprocess),
    ('clf', xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_lambda=2.0,
        eval_metric='logloss', random_state=42,
    )),
])
xgb_pipe.fit(X_train, y_train)
pred_xgb = xgb_pipe.predict_proba(X_test)[:, 1]
evaluate('XGBoost (full features)', y_test, pred_xgb)

# ---- Benchmark: StatsBomb's proprietary model ----
# Using their official values from the dataset as a control/benchmark
pred_sb = test_df['statsbomb_xg'].values
evaluate('StatsBomb proprietary xG (benchmark)', y_test, pred_sb)

print()
print("=== Summary (lower LogLoss/Brier = better, higher AUC = better) ===")
for name, m in results.items():
    print(f"  {name:38s} AUC={m['auc']:.4f}  LogLoss={m['log_loss']:.4f}  Brier={m['brier']:.4f}")

# ---- Save production model ----
# Surprisingly, regularized logistic regression slightly beats XGBoost on the test set!
# This is probably because our engineered geometry features (like distance and angle)
# already capture the non-linear physics of a shot, so a simpler linear model
# generalizes better than tree boosting on ~6.7k rows.
# So we will dump the logistic regression model as the production model!
joblib.dump(logreg_pipe, 'sb_out/xg_model.joblib')

# Save metrics so our Streamlit dashboard/API can read and display them
with open('sb_out/metrics.json', 'w') as f:
    json.dump({
        'features': {'numeric': NUMERIC, 'categorical': CATEGORICAL, 'boolean': BOOLEAN},
        'results': results,
        'train_shots': len(train_df),
        'test_shots': len(test_df),
        'test_tournament': 'UEFA Euro 2024 (held out entirely from training)',
    }, f, indent=2)

# Save curve data for plotting ROC and calibration charts
np.savez('sb_out/curve_data.npz',
    y_test=y_test.values,
    pred_baseline=pred_baseline, pred_logreg=pred_logreg,
    pred_xgb=pred_xgb, pred_sb=pred_sb)

print("\nSaved: sb_out/xg_model.joblib, sb_out/metrics.json, sb_out/curve_data.npz")
