# train_fantasy_model.py

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from math import sqrt

DATA_CSV = "player_season_stats.csv"
MODEL_PATH = "models/fantasy_model.pkl"


FEATURE_COLS = [
    "season",
    "team",
    "runs",
    "balls_faced",
    "outs",
    "fours",
    "sixes",
    "strike_rate",
    "balls_bowled",
    "runs_conceded",
    "wickets",
    "overs",
    "economy",
    "is_batsman",
    "is_bowler",
    "is_allrounder",
    "is_pure_batsman",
    "is_pure_bowler",
]


def main():
    print("📥 Loading player-season dataset...")
    df = pd.read_csv(DATA_CSV)

    # Ensure boolean flags are numeric (0/1)
    bool_cols = ["is_batsman", "is_bowler", "is_allrounder", "is_pure_batsman", "is_pure_bowler"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    X = df[FEATURE_COLS].copy()
    y = df["fantasy_points"].astype(float)

    # Split train/test
    print("✂️ Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocess: team is categorical, rest numeric
    categorical_features = ["team", "season"]
    numeric_features = [c for c in FEATURE_COLS if c not in categorical_features]


    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )

    print("🧠 Training RandomForestRegressor...")
    model.fit(X_train, y_train)

    print("📊 Evaluating on test set...")
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R^2  : {r2:.3f}")

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
