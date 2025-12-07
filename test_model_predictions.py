import pandas as pd
import joblib

MODEL_PATH = "models/fantasy_model.pkl"
DATA_CSV = "player_season_stats.csv"

print("📥 Loading model...")
model = joblib.load(MODEL_PATH)

print("📥 Loading dataset...")
df = pd.read_csv(DATA_CSV)

# Pick 5 random players for testing
sample = df.sample(5, random_state=42)
X = sample.drop(columns=["fantasy_points", "player_name"])
y_true = sample["fantasy_points"]

print("\n🔍 SAMPLE TEST:")
print(sample[["player_name", "team", "fantasy_points"]], "\n")

y_pred = model.predict(X)

sample["predicted"] = y_pred

print("📊 MODEL vs TRUE VALUES")
print(sample[["player_name", "fantasy_points", "predicted"]])
