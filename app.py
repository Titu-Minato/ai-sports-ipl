from flask import Flask, render_template, request, url_for, redirect
import pandas as pd
import pickle

app = Flask(__name__)

# ---------- TEAM CONFIG ----------

teams = [
    {"code": "CSK",  "name": "Chennai Super Kings",            "logo": "csk.png"},
    {"code": "DC",   "name": "Delhi Capitals",                 "logo": "dc.png"},
    {"code": "GT",   "name": "Gujarat Titans",                 "logo": "gt.png"},
    {"code": "KKR",  "name": "Kolkata Knight Riders",          "logo": "kkr.png"},
    {"code": "LSG",  "name": "Lucknow Super Giants",           "logo": "lsg.png"},
    {"code": "MI",   "name": "Mumbai Indians",                 "logo": "mi.png"},
    {"code": "PBKS", "name": "Punjab Kings",                   "logo": "pbks.png"},
    {"code": "RCB",  "name": "Royal Challengers Bangalore",    "logo": "rcb.png"},
    {"code": "RR",   "name": "Rajasthan Royals",               "logo": "rr.png"},
    {"code": "SH",   "name": "Sunrisers Hyderabad",            "logo": "sh.png"},
]

# Some teams have old names in the CSV (Kings XI Punjab, Delhi Daredevils, etc.)
TEAM_NAME_ALIASES = {
    "CSK": ["Chennai Super Kings"],
    "MI": ["Mumbai Indians"],
    "RCB": ["Royal Challengers Bangalore"],
    "RR": ["Rajasthan Royals"],
    "KKR": ["Kolkata Knight Riders"],
    "SH": ["Sunrisers Hyderabad"],
    "DC": ["Delhi Daredevils", "Delhi Capitals"],
    "PBKS": ["Kings XI Punjab", "Punjab Kings"],
    # older / related franchises mapped to modern codes
    "GT": ["Gujarat Titans", "Gujarat Lions"],
    "LSG": ["Lucknow Super Giants"],  # new team, single name
}

# ---------- LOAD PLAYER STATS ONCE ----------

try:
    player_stats = pd.read_csv("player_season_stats.csv")
    print("Loaded player_season_stats.csv with columns:", list(player_stats.columns))
except Exception as e:
    print("Error loading player_season_stats.csv:", e)
    player_stats = pd.DataFrame()

# ---------- LOAD TRAINED MODEL ----------

try:
    with open("models/fantasy_model.pkl", "rb") as f:
        fantasy_model = pickle.load(f)
    print("Loaded fantasy model from models/fantasy_model.pkl, type:", type(fantasy_model))
except Exception as e:
    print("Could not load fantasy model:", e)
    fantasy_model = None



# ---------- INSIGHT HELPERS ----------

def get_team_insights(team: dict):
    """
    Last 2 seasons insights for a team:
    - Top batting average (runs/outs)
    - Highest run scorer
    - Top wicket taker
    - Best economy bowler (runs_conceded / overs, only real bowlers)
    """
    code = team["code"]
    possible_names = TEAM_NAME_ALIASES.get(code, [team["name"]])

    insights = {
        "top_avg_batsman": {"player_name": "-", "average": "-"},
        "top_run_scorer": {"player_name": "-", "runs": "-"},
        "top_wicket_taker": {"player_name": "-", "wickets": "-"},
        "best_economy": {"player_name": "-", "economy": "-"},
    }

    if player_stats.empty:
        return insights

    # Filter rows for this team (considering aliases)
    df = player_stats[player_stats["team"].isin(possible_names)].copy()
    if df.empty:
        return insights

    # Keep last 2 seasons
    if "season" in df.columns:
        seasons = sorted(df["season"].dropna().unique())
        if len(seasons) > 2:
            df = df[df["season"].isin(seasons[-2:])]

    if df.empty:
        return insights

    # ---- Top batting avg = runs / outs ----
    if "runs" in df.columns and "outs" in df.columns:
        df_bat = df.copy()
        df_bat["batting_avg"] = df_bat.apply(
            lambda row: row["runs"] / row["outs"] if row["outs"] > 0 else 0,
            axis=1,
        )
        top_avg_row = (
            df_bat.sort_values("batting_avg", ascending=False)
                 .head(1)
                 .iloc[0]
        )
        insights["top_avg_batsman"] = {
            "player_name": top_avg_row["player_name"],
            "average": round(float(top_avg_row["batting_avg"]), 2),
        }

    # ---- Highest run scorer ----
    if "runs" in df.columns:
        top_runs_row = df.sort_values("runs", ascending=False).head(1).iloc[0]
        insights["top_run_scorer"] = {
            "player_name": top_runs_row["player_name"],
            "runs": int(top_runs_row["runs"]),
        }

    # ---- Top wicket taker ----
    if "wickets" in df.columns:
        top_wkts_row = df.sort_values("wickets", ascending=False).head(1).iloc[0]
        insights["top_wicket_taker"] = {
            "player_name": top_wkts_row["player_name"],
            "wickets": int(top_wkts_row["wickets"]),
        }

    # ---- Best economy bowler (real bowlers, recomputed economy) ----
    needed_cols = {"runs_conceded", "overs", "balls_bowled"}
    if needed_cols.issubset(df.columns):
        df_bowl = df.copy()
        # must have actually bowled
        df_bowl = df_bowl[df_bowl["balls_bowled"] > 0]
        df_bowl = df_bowl[df_bowl["overs"] > 0]

        # require at least 5 overs to be considered
        df_bowl = df_bowl[df_bowl["overs"] >= 5]

        if not df_bowl.empty:
            df_bowl["econ_calc"] = df_bowl["runs_conceded"] / df_bowl["overs"]
            best_eco_row = (
                df_bowl.sort_values("econ_calc", ascending=True)
                       .head(1)
                       .iloc[0]
            )
            insights["best_economy"] = {
                "player_name": best_eco_row["player_name"],
                "economy": round(float(best_eco_row["econ_calc"]), 2),
            }

    return insights


def build_role_label(row):
    """Turn boolean flags into a simple role label for display."""
    if row.get("is_allrounder", 0) == 1:
        return "All-rounder"
    if row.get("is_pure_batsman", 0) == 1 or row.get("is_batsman", 0) == 1:
        return "Batter"
    if row.get("is_pure_bowler", 0) == 1 or row.get("is_bowler", 0) == 1:
        return "Bowler"
    return "Player"


def get_team_model_xi(team, n_players=11):
    """
    Return a balanced Best XI for the given team using the fantasy model.
    Captain / VC are chosen using ONLY the latest season (e.g. 2024).
    """
    team_name = team["name"]

    # Handle aliases (Kings XI Punjab -> PBKS, etc.)
    team_names = [team_name]
    for k, v in TEAM_NAME_ALIASES.items():
        if v == team_name:
            team_names.append(k)

    team_df = player_stats[player_stats["team"].isin(team_names)].copy()
    if team_df.empty:
        return []

    # --- choose seasons for XI (last 2 seasons as before) ---
    all_seasons = None
    if "season" in team_df.columns:
        all_seasons = sorted(team_df["season"].unique())
        recent_seasons = all_seasons[-2:]  # XI uses last 2 seasons
        team_df = team_df[team_df["season"].isin(recent_seasons)]

    if team_df.empty:
        return []

    # ---------- prediction source ----------
    if fantasy_model is not None:
        feature_cols = [
            c for c in team_df.columns
            if c not in ["player_name", "team", "season", "role"]
        ]
        X = team_df[feature_cols].fillna(0)

        if hasattr(fantasy_model, "n_features_in_"):
            n = fantasy_model.n_features_in_
            X = X.iloc[:, :n]

        preds = fantasy_model.predict(X)
        team_df["pred_score"] = preds
    else:
        if "fantasy_points" in team_df.columns:
            team_df["pred_score"] = team_df["fantasy_points"]
        else:
            team_df["pred_score"] = (
                team_df.get("runs", 0)
                + team_df.get("wickets", 0) * 20
                - team_df.get("economy", 0) * 2
            )

    # Role labels
    team_df["role_label"] = team_df.apply(build_role_label, axis=1)

    # Sort by predicted score for base ranking
    team_df = team_df.sort_values("pred_score", ascending=False)

    xi = []

    def add_players(sub_df, limit):
        added = 0
        for _, row in sub_df.iterrows():
            name = row["player_name"]
            if any(p["player_name"] == name for p in xi):
                continue
            xi.append({
                "player_name": name,
                "role": row["role_label"],
                "pred_score": float(row["pred_score"]),
                # flags will be filled later
                "is_captain": False,
                "is_vice_captain": False,
                "captain_score": 0.0,
            })
            added += 1
            if added >= limit:
                break

    # Balanced XI
    add_players(team_df[team_df["role_label"] == "Batter"], 4)
    add_players(team_df[team_df["role_label"] == "All-rounder"], 3)
    add_players(team_df[team_df["role_label"] == "Bowler"], 4)

    if len(xi) < n_players:
        add_players(team_df, n_players - len(xi))

    if not xi:
        return xi

    # ---------- C / VC based ONLY on latest season ----------

    # 1) pick latest season (e.g. 2024)
    latest_season_df = None
    if all_seasons is not None:
        latest_season = all_seasons[-1]
        latest_season_df = player_stats[
            (player_stats["team"].isin(team_names))
            & (player_stats["season"] == latest_season)
        ].copy()

    # If no season column / latest season data, fall back to team_df
    if latest_season_df is None or latest_season_df.empty:
        latest_season_df = team_df.copy()

    # Decide which metric to use for “last season performance”
    if "fantasy_points" in latest_season_df.columns:
        perf_col = "fantasy_points"
    elif "pred_score" in latest_season_df.columns:
        perf_col = "pred_score"
    else:
        latest_season_df["perf_metric"] = (
            latest_season_df.get("runs", 0)
            + latest_season_df.get("wickets", 0) * 25
        )
        perf_col = "perf_metric"

    # Aggregate per player for latest season
    season_perf = (
        latest_season_df.groupby("player_name")[perf_col]
        .mean()
        .to_dict()
    )

    # Role bonus – captaincy favours AR / top batters / strike bowlers
    ROLE_BONUS = {
        "All-rounder": 80,
        "Batter": 40,
        "Bowler": 30,
    }

    for p in xi:
        last_season_perf = season_perf.get(p["player_name"], 0.0)
        bonus = ROLE_BONUS.get(p["role"], 0)
        p["captain_score"] = float(last_season_perf) + bonus
        p["is_captain"] = False
        p["is_vice_captain"] = False

    # Choose C & VC by captain_score (latest season only)
    indices = sorted(
        range(len(xi)),
        key=lambda i: xi[i]["captain_score"],
        reverse=True,
    )
    if indices:
        xi[indices[0]]["is_captain"] = True
    if len(indices) > 1:
        xi[indices[1]]["is_vice_captain"] = True

    return xi





# ---------- ROUTES ----------

@app.route("/")
def home():
    """Homepage with IPL team cards grid."""
    return render_template("index.html", teams=teams)


@app.route("/team/<team_code>")
def team_detail(team_code):
    """Team insight page: last-2-season leaders + model XI."""
    team = next((t for t in teams if t["code"].lower() == team_code.lower()), None)
    if team is None:
        return "Team not found", 404

    insights = get_team_insights(team)
    model_xi = get_team_model_xi(team)

    print("INSIGHTS FOR", team["code"], ":", insights)
    return render_template(
        "team_detail.html",
        team=team,
        insights=insights,
        model_xi=model_xi,
    )


@app.route("/ask")
def ask_ai():
    """Handles query bar (currently just echoes back)."""
    query = request.args.get("q", "").strip()
    if not query:
        return redirect(url_for("home"))
    return f"You asked: {query} (AI answer logic will go here)"


# ---------- MAIN ----------

if __name__ == "__main__":
    app.run(debug=True)
