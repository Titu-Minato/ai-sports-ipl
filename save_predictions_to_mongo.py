# save_predictions_to_mongo.py

import pandas as pd
import joblib
from pymongo import MongoClient
from config import Config

DATA_CSV = "player_season_stats.csv"
MODEL_PATH = "models/fantasy_model.pkl"
COLLECTION_NAME = "player_season_predictions"

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

# 👇 Only fix weird labels, leave 2009, 2020, etc. as they are
SEASON_MAP = {
    "2007/08": "2008",
    "2009/10": "2010",
    "2020/21": "2021",
}


def main():
    print("📥 Loading player-season dataset...")
    df = pd.read_csv(DATA_CSV)

    # --- Normalize season labels ---
    df["season"] = df["season"].astype(str).str.strip()
    df["season"] = df["season"].replace(SEASON_MAP)

    # Clean team names
    df["team"] = df["team"].astype(str).str.strip()

    # Ensure booleans are booleans
    bool_cols = ["is_batsman", "is_bowler", "is_allrounder", "is_pure_batsman", "is_pure_bowler"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    print("🧠 Loading trained model...")
    model = joblib.load(MODEL_PATH)

    print("🔮 Predicting fantasy points with model...")
    X = df[FEATURE_COLS]
    preds = model.predict(X)
    df["predicted_points"] = preds

    print("🔌 Connecting to MongoDB...")
    client = MongoClient(Config.MONGO_URI)
    db = client[Config.MONGO_DB_NAME]
    coll = db[COLLECTION_NAME]

    print(f"🗑 Dropping old collection '{COLLECTION_NAME}' (if exists)...")
    coll.drop()

    print(f"💾 Inserting {len(df)} documents into '{COLLECTION_NAME}'...")
    records = df.to_dict(orient="records")
    coll.insert_many(records)

    print("📚 Creating indexes for fast queries (team + season + player_name)...")
    coll.create_index([("team", 1), ("season", 1)])
    coll.create_index([("player_name", 1), ("season", 1)])

    print("✅ Done. Normalized seasons stored in MongoDB.")


if __name__ == "__main__":
    main()
