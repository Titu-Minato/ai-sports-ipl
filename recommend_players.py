import pandas as pd
from recommendation_engine import (
    compute_rec_scores,
    TeamConstraints,
    recommend_team,
    assign_captain_vice,
)


def infer_role(row):
    """
    Infer simplified role from boolean flags in your dataset.
    Uses: is_batsman, is_bowler, is_allrounder
    """
    if row.get("is_allrounder", False):
        return "AR"
    if row.get("is_batsman", False) and not row.get("is_bowler", False):
        return "BAT"
    if row.get("is_bowler", False) and not row.get("is_batsman", False):
        return "BOWL"
    # Fallbacks:
    if row.get("is_batsman", False) and row.get("is_bowler", False):
        return "AR"
    return "BAT"


def season_sort_key(label):
    """
    Robust sorting for season labels like:
    - 2008
    - "2009"
    - "2007/08"
    - "2020/21"
    """
    if isinstance(label, (int, float)):
        return int(label)
    text = str(label)
    # Extract digit chunks: "2007/08" -> ["2007", "08"]
    parts = "".join(ch if ch.isdigit() else " " for ch in text).split()
    if parts:
        four_digit = [p for p in parts if len(p) == 4]
        if four_digit:
            return int(four_digit[-1])
        return int(parts[-1])
    return 0


def generate_best_xi(match_teams=None, risk_mode="balanced"):
    """
    Core function:
    - Loads player_season_stats.csv
    - Aggregates fantasy_points by player/team/season
    - Builds season-weighted stats
    - Optionally filters by match_teams (2-team match)
    - Computes RecScore
    - Selects best XI with constraints
    - Assigns Captain (C) and Vice-Captain (VC)
    """

    print("📥 Loading raw player-season stats...")
    raw = pd.read_csv("player_season_stats.csv")

    # 1) Infer role_detailed from is_batsman / is_bowler / is_allrounder
    required_flags = ["is_batsman", "is_bowler", "is_allrounder"]
    missing_flags = [c for c in required_flags if c not in raw.columns]
    if missing_flags:
        raise ValueError(f"Missing role flag columns in CSV: {missing_flags}")

    raw["role_detailed"] = raw.apply(infer_role, axis=1)

    # 2) Aggregate fantasy_points per player per season
    if "fantasy_points" not in raw.columns:
        raise ValueError("Column 'fantasy_points' not found in player_season_stats.csv")

    agg = (
        raw.groupby(["player_name", "team", "role_detailed", "season"], as_index=False)[
            "fantasy_points"
        ]
        .sum()
    )

    # 3) Pivot: one row per player, seasons become columns
    pivot = agg.pivot_table(
        index=["player_name", "team", "role_detailed"],
        columns="season",
        values="fantasy_points",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    # 4) Detect last 3 seasons dynamically
    season_cols = [c for c in pivot.columns if c not in ["player_name", "team", "role_detailed"]]
    if not season_cols:
        raise ValueError("No season columns found after pivot. Check 'season' in CSV.")

    print("Detected season columns in pivot:", season_cols)

    season_cols_sorted = sorted(season_cols, key=season_sort_key)

    if len(season_cols_sorted) >= 3:
        y1, y2, y3 = season_cols_sorted[-3:]   # use last 3 seasons
    elif len(season_cols_sorted) == 2:
        y1, y2 = season_cols_sorted[-2:]
        y3 = y2
    else:
        y1 = y2 = y3 = season_cols_sorted[0]

    print(f"Using seasons {y1}, {y2}, {y3} as 2022, 2023, 2024 equivalents.")

    pivot["points_2022"] = pivot[y1]
    pivot["points_2023"] = pivot[y2]
    pivot["points_2024"] = pivot[y3]

    # 5) Credits based on most recent season (y3)
    max_recent = pivot[y3].max()
    if max_recent > 0:
        pivot["credits"] = 7.0 + 3.0 * (pivot[y3] / max_recent)  # approx 7–10
    else:
        pivot["credits"] = 8.5

    players_df = pivot[
        [
            "player_name",
            "team",
            "role_detailed",
            "credits",
            "points_2022",
            "points_2023",
            "points_2024",
        ]
    ].copy()

    print(f"✅ Prepared players_df (before match filter): {len(players_df)}")

    # 6) Optional match filter (for 2-team matches)
    if match_teams is not None and len(match_teams) > 0:
        match_teams_clean = [t.strip().lower() for t in match_teams]
        mask = players_df["team"].str.lower().isin(match_teams_clean)
        players_df = players_df[mask].copy()

        print(f"🎯 Match mode ON: filtering to teams: {match_teams}")
        print("Players after match filter:", len(players_df))

        if len(players_df) == 0:
            print("⚠️ No players found for those team names (case-insensitive).")
            print("   Example valid team names from dataset:")
            print(raw["team"].drop_duplicates().head(20).to_list())
            # Return empty team_df so caller can handle
            return players_df.assign(RecScore=[], C=[], VC=[])

        # For 2-team match, allow up to 7 players from the same team
        max_team_limit = 7 if len(match_teams) == 2 else 4
    else:
        print("🌍 Global mode: using all teams")
        max_team_limit = 4

    # 7) Compute recommendation scores
    scored_df = compute_rec_scores(players_df, risk_mode=risk_mode)

    # 8) Team constraints (no explicit WK role in this dataset)
    constraints = TeamConstraints(
        target_roles={"BAT": 4, "BOWL": 3, "AR": 4},  # 4 + 3 + 4 = 11
        max_players_per_real_team=max_team_limit,
        max_credits=100.0,
    )

    team_df = recommend_team(scored_df, constraints)
    team_df = assign_captain_vice(team_df)

    return team_df


if __name__ == "__main__":
    print("Choose mode:")
    print("1) Global best XI (all IPL teams)")
    print("2) Match best XI (only from 2 teams)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        team_global = generate_best_xi(match_teams=None, risk_mode="balanced")
        print("\n🏏 Global Best XI:")
        print(team_global[["player_name", "role_detailed", "team", "credits", "RecScore", "C", "VC"]])

    elif choice == "2":
        team1 = input("Enter Team 1 name (e.g. 'Chennai Super Kings'): ").strip()
        team2 = input("Enter Team 2 name (e.g. 'Mumbai Indians'): ").strip()
        match_teams = [team1, team2]

        team_match = generate_best_xi(match_teams=match_teams, risk_mode="balanced")
        print(f"\n🏏 Match Best XI ({team1} vs {team2}):")
        if not team_match.empty:
            print(team_match[["player_name", "role_detailed", "team", "credits", "RecScore", "C", "VC"]])
        else:
            print("No XI could be formed (check team names).")
    else:
        print("Invalid choice.")
