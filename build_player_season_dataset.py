# build_player_season_dataset.py

import os
import pandas as pd
import numpy as np

MATCHES_CSV = "data/matches.csv"
DELIVERIES_CSV = "data/deliveries.csv"
OUTPUT_CSV = "player_season_stats.csv"



def choose_col(df, options, required=True, what=""):
    """
    Helper: pick the first existing column name from a list of options.
    Raises a clear error if not found and required=True.
    """
    for col in options:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Could not find any of {options} for {what}. "
                         f"Available columns: {list(df.columns)}")
    return None


def compute_batting_stats(deliveries_merged):
    """
    Aggregate batting stats per (player_name, team, season) for THIS dataset:
    uses columns: batter, batting_team, batsman_runs, extras_type, player_dismissed, season
    """
    df = deliveries_merged.copy()

    # Player and team names
    df["player_name"] = df["batter"]
    df["team"] = df["batting_team"]

    # Handle extras for valid balls (no wides/noballs as balls faced)
    df["extras_type_lower"] = df["extras_type"].astype(str).str.lower().fillna("")

    df["is_wide"] = df["extras_type_lower"].isin(["wides", "wide"])
    df["is_noball"] = df["extras_type_lower"].isin(["noballs", "noball"])

    # A ball faced is valid if it's NOT a wide or no-ball
    df["valid_ball_faced"] = ~(df["is_wide"] | df["is_noball"])

    # Fours & sixes
    df["is_four"] = df["batsman_runs"] == 4
    df["is_six"] = df["batsman_runs"] == 6

    grp_cols = ["player_name", "team", "season"]

    batting_agg = df.groupby(grp_cols).agg(
        runs=("batsman_runs", "sum"),
        balls_faced=("valid_ball_faced", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
    ).reset_index()

    # Dismissals: when this batter is the player_dismissed
    dis_df = df[df["player_dismissed"].notna()].copy()
    dis_df = dis_df[dis_df["player_dismissed"] == dis_df["player_name"]]

    dismiss_agg = dis_df.groupby(grp_cols).agg(
        outs=("player_dismissed", "count")
    ).reset_index()

    batting = pd.merge(batting_agg, dismiss_agg, on=grp_cols, how="left")
    batting["outs"] = batting["outs"].fillna(0).astype(int)

    # Strike rate
    batting["strike_rate"] = batting.apply(
        lambda row: (row["runs"] * 100.0 / row["balls_faced"]) if row["balls_faced"] > 0 else 0.0,
        axis=1
    )

    return batting



def compute_bowling_stats(deliveries_merged):
    """
    Aggregate bowling stats per (player_name, team, season) for THIS dataset:
    uses columns: bowler, bowling_team, total_runs, extra_runs, extras_type,
                  is_wicket, dismissal_kind, season
    """
    df = deliveries_merged.copy()

    df["player_name"] = df["bowler"]
    df["team"] = df["bowling_team"]

    df["extras_type_lower"] = df["extras_type"].astype(str).str.lower().fillna("")

    # Legal balls: not wides, not no-balls
    df["is_wide"] = df["extras_type_lower"].isin(["wides", "wide"])
    df["is_noball"] = df["extras_type_lower"].isin(["noballs", "noball"])
    df["valid_ball_bowled"] = ~(df["is_wide"] | df["is_noball"])

    # Byes & leg-byes do NOT count as bowler's runs conceded
    df["bye_runs"] = np.where(df["extras_type_lower"].isin(["byes", "bye"]), df["extra_runs"], 0)
    df["legbye_runs"] = np.where(df["extras_type_lower"].isin(["legbyes", "legbye"]), df["extra_runs"], 0)

    df["runs_conceded"] = df["total_runs"] - df["bye_runs"] - df["legbye_runs"]

    # Wickets credited to bowler (exclude run-outs, etc.)
    wicket_kinds = {
        "bowled",
        "caught",
        "lbw",
        "stumped",
        "caught and bowled",
        "caught & bowled",
        "hit wicket",
        "hit-wicket",
    }
    wicket_kinds_lower = {w.lower() for w in wicket_kinds}

    df["dismissal_kind_lower"] = df["dismissal_kind"].astype(str).str.lower()
    df["is_bowler_wicket"] = (
        (df["is_wicket"] == 1)
        & df["dismissal_kind_lower"].isin(wicket_kinds_lower)
    )

    grp_cols = ["player_name", "team", "season"]

    bowling = df.groupby(grp_cols).agg(
        balls_bowled=("valid_ball_bowled", "sum"),
        runs_conceded=("runs_conceded", "sum"),
        wickets=("is_bowler_wicket", "sum"),
    ).reset_index()

    # Overs & economy
    bowling["overs"] = bowling["balls_bowled"] / 6.0
    bowling["economy"] = bowling.apply(
        lambda row: (row["runs_conceded"] * 6.0 / row["balls_bowled"]) if row["balls_bowled"] > 0 else 0.0,
        axis=1
    )

    return bowling


def compute_fantasy_points(row):
    """
    Fantasy scoring Formula A (Dream11-style simplified):

    +1 per run
    +1 per four
    +2 per six
    +25 per wicket
    +10 bonus if balls_faced >= 10 and strike_rate >= 140
    +10 bonus if overs >= 2 and economy <= 6
    """
    points = 0.0

    # Basic
    points += row.get("runs", 0) * 1.0
    points += row.get("fours", 0) * 1.0
    points += row.get("sixes", 0) * 2.0
    points += row.get("wickets", 0) * 25.0

    # Strike rate bonus
    if row.get("balls_faced", 0) >= 10 and row.get("strike_rate", 0) >= 140:
        points += 10.0

    # Economy bonus
    if row.get("overs", 0) >= 2 and row.get("economy", 99) <= 6:
        points += 10.0

    return points


def main():
    print("📥 Loading CSVs...")
    matches = pd.read_csv(MATCHES_CSV)
    deliveries = pd.read_csv(DELIVERIES_CSV)

    # Choose join keys
    match_id_in_deliveries = choose_col(deliveries, ["match_id", "Match_Id", "matchID"], what="match id in deliveries")
    match_id_in_matches = choose_col(matches, ["id", "match_id", "ID"], what="match id in matches")
    season_col = choose_col(matches, ["season", "Season"], what="season in matches")

    # Attach season to each delivery
    print("🔗 Merging deliveries with matches to get season...")
    matches_small = matches[[match_id_in_matches, season_col]].rename(
        columns={match_id_in_matches: "match_id", season_col: "season"}
    )
    deliveries_merged = deliveries.rename(columns={match_id_in_deliveries: "match_id"})
    deliveries_merged = deliveries_merged.merge(matches_small, on="match_id", how="left")

    # Compute batting & bowling stats
    print("📊 Computing batting stats...")
    batting = compute_batting_stats(deliveries_merged)

    print("📊 Computing bowling stats...")
    bowling = compute_bowling_stats(deliveries_merged)

    # Merge batting and bowling into single player-season rows
    print("🧩 Combining batting & bowling...")
    merge_cols = ["player_name", "team", "season"]
    all_players = pd.merge(
        batting,
        bowling,
        on=merge_cols,
        how="outer",
        suffixes=("_bat", "_bowl")
    )

    # Fill NaNs with 0 for numeric stats
    for col in ["runs", "balls_faced", "fours", "sixes", "outs", "strike_rate",
                "balls_bowled", "runs_conceded", "wickets", "overs", "economy"]:
        if col in all_players.columns:
            all_players[col] = all_players[col].fillna(0)

    # Player type flags
    # Simple heuristic:
    # Batsman if runs >= 200 or balls_faced >= 100
    # Bowler if overs >= 10
    all_players["is_batsman"] = (all_players["runs"] >= 200) | (all_players["balls_faced"] >= 100)
    all_players["is_bowler"] = (all_players["overs"] >= 10)
    all_players["is_allrounder"] = all_players["is_batsman"] & all_players["is_bowler"]
    all_players["is_pure_batsman"] = all_players["is_batsman"] & (~all_players["is_bowler"])
    all_players["is_pure_bowler"] = all_players["is_bowler"] & (~all_players["is_batsman"])

    # Fantasy points
    print("🧮 Computing fantasy points...")
    all_players["fantasy_points"] = all_players.apply(compute_fantasy_points, axis=1)

    # Keep relevant columns & sort
    cols = [
        "player_name",
        "team",
        "season",
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
        "fantasy_points",
    ]
    all_players = all_players[cols].sort_values(["season", "team", "fantasy_points"], ascending=[True, True, False])

    print(f"💾 Saving player-season dataset to {OUTPUT_CSV} ...")
    all_players.to_csv(OUTPUT_CSV, index=False)
    print("✅ Done.")


if __name__ == "__main__":
    main()
