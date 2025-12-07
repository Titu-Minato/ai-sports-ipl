import pandas as pd

DATA_CSV = "player_season_stats.csv"

def main():
    print("📥 Loading dataset...")
    df = pd.read_csv(DATA_CSV)

    print("\n🔎 Columns:")
    print(df.columns.tolist())

    print("\n📏 Shape (rows, columns):")
    print(df.shape)

    print("\n🧼 Missing values per column:")
    print(df.isna().sum())

    print("\n📊 Basic stats for main numeric columns:")
    print(df[[
        "runs",
        "balls_faced",
        "fours",
        "sixes",
        "wickets",
        "overs",
        "economy",
        "fantasy_points"
    ]].describe())

    print("\n🏏 Top 10 players by fantasy_points:")
    print(
        df.sort_values("fantasy_points", ascending=False)[
            ["player_name", "team", "season", "fantasy_points", "runs", "wickets"]
        ].head(10).to_string(index=False)
    )

    print("\n📈 Correlation with fantasy_points:")
    corr = df[[
        "runs",
        "balls_faced",
        "fours",
        "sixes",
        "wickets",
        "overs",
        "economy",
        "fantasy_points"
    ]].corr()
    print(corr["fantasy_points"])

if __name__ == "__main__":
    main()
