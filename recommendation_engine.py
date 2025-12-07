"""
Recommendation engine for fantasy XI selection.

This module exposes:

- compute_rec_scores(df, risk_mode, popularity_mode, beta_ml)
- TeamConstraints dataclass
- recommend_team(scored_df, constraints)
- assign_captain_vice(team_df)

Expected INPUT columns in players_df:

Required:
    player_name       : str
    role_detailed     : str   ("BAT", "BOWL", "AR", "WK-BAT", "WK-only")
    team              : str   (MI, CSK, etc.)
    credits           : float
    points_2022       : float (total or avg fantasy points in 2022)
    points_2023       : float
    points_2024       : float

Optional:
    fantasy_pred      : float (model prediction for next match)
    selection_pct     : float (ownership %, 0–100)

compute_rec_scores() will:
- weight seasons: 2024 * 0.50 + 2023 * 0.35 + 2022 * 0.15
- derive long_avg, recent_avg, std, venue/opp dummy stats
- normalize and compute RecScore

recommend_team() will:
- enforce role counts
- enforce max players per real team
- enforce credit limit
- prefer WK-BAT over WK-only for WK slot (small boost)

assign_captain_vice() will:
- choose best RecScore as Captain (C = True)
- choose balanced high RecScore + consistency as Vice-Captain (VC = True)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import heapq
import numpy as np
import pandas as pd

RiskMode = Literal["safe", "balanced", "aggressive"]
PopularityMode = Literal["ignore", "follow_crowd", "differential"]


# ---------- Helpers ---------------------------------------------------------


def _zscore(series: pd.Series) -> pd.Series:
    """Return z-score normalized series; handle constant/NaN safely."""
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def _map_base_role(role_detailed: str) -> str:
    """
    Map detailed role to base role:
    - "WK-BAT" and "WK-only" -> "WK"
    - others unchanged ("BAT", "BOWL", "AR")
    """
    if role_detailed in ("WK-BAT", "WK-only"):
        return "WK"
    return role_detailed


# ---------- Main scoring function ------------------------------------------


def compute_rec_scores(
    players_df: pd.DataFrame,
    risk_mode: RiskMode = "balanced",
    popularity_mode: PopularityMode = "ignore",
    beta_ml: float = 0.7,
) -> pd.DataFrame:
    """
    Compute RecScore for each player and return a new DataFrame.

    players_df must contain at least:
        player_name, role_detailed, team, credits,
        points_2022, points_2023, points_2024

    Optional:
        fantasy_pred   : model prediction
        selection_pct  : ownership %
    """
    df = players_df.copy()

    # ---------- 0) Force important columns to numeric & clean ---------------
    numeric_cols = ["points_2022", "points_2023", "points_2024", "credits"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Basic role handling
    df["role_base"] = df["role_detailed"].apply(_map_base_role)
    df["is_wk_bat"] = df["role_detailed"].eq("WK-BAT")

    # ---------- 1) Season-weighted performance (2024 > 2023 > 2022) ---------
    df["fp_year_weighted"] = (
        0.15 * df["points_2022"]
        + 0.35 * df["points_2023"]
        + 0.50 * df["points_2024"]
    )

    df["fp_long_avg"] = df[["points_2022", "points_2023", "points_2024"]].mean(axis=1)
    df["fp_recent_avg"] = df["points_2024"]
    df["fp_std"] = df[["points_2022", "points_2023", "points_2024"]].std(axis=1)

    # Dummy venue/opposition effects (can be replaced later)
    df["fp_venue_avg"] = df["points_2024"] * 0.9
    df["fp_opp_avg"] = df["points_2023"] * 0.8

    # ---------- 2) Prediction: ML + Stats -----------------------------------
    if "fantasy_pred" not in df.columns:
        df["fantasy_pred"] = df["fp_year_weighted"]

    # Ensure fantasy_pred is numeric
    df["fantasy_pred"] = pd.to_numeric(df["fantasy_pred"], errors="coerce").fillna(
        df["fp_year_weighted"]
    )

    w1, w2, w3, w4 = 0.5, 0.2, 0.2, 0.1
    df["P_stat"] = (
        w1 * df["fp_year_weighted"]
        + w2 * df["fp_recent_avg"]
        + w3 * df["fp_venue_avg"]
        + w4 * df["fp_opp_avg"]
    )

    df["P_ml_filled"] = df["fantasy_pred"].fillna(df["P_stat"])
    df["P_final"] = beta_ml * df["P_ml_filled"] + (1 - beta_ml) * df["P_stat"]

    # ---------- 3) Extra features -------------------------------------------
    df["ValueEff"] = df["P_final"] / df["credits"].replace(0, np.nan)
    df["ValueEff"] = df["ValueEff"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["Cons_raw"] = df["fp_long_avg"] / (df["fp_std"].replace(0, np.nan) + 1e-6)

    if "selection_pct" not in df.columns:
        df["selection_pct"] = 50.0
    df["selection_pct"] = pd.to_numeric(df["selection_pct"], errors="coerce").fillna(50.0)

    # ---------- 4) Normalize (Z-score) --------------------------------------
    df["P_norm"] = _zscore(df["P_final"])
    df["Form_norm"] = _zscore(df["fp_recent_avg"])
    df["Cons_norm"] = _zscore(df["Cons_raw"])
    df["Venue_norm"] = _zscore(df["fp_venue_avg"])
    df["Opp_norm"] = _zscore(df["fp_opp_avg"])
    df["ValueEff_norm"] = _zscore(df["ValueEff"])
    df["Own_norm"] = _zscore(df["selection_pct"])
    df["Risk_norm"] = _zscore(df["fp_std"])

    # ---------- 5) Risk & Popularity adjustments ----------------------------
    if risk_mode == "safe":
        df["RiskAdj"] = -df["Risk_norm"]
    elif risk_mode == "aggressive":
        df["RiskAdj"] = df["Risk_norm"]
    else:
        df["RiskAdj"] = 0.3 * df["Risk_norm"]

    if popularity_mode == "ignore":
        df["PopAdj"] = 0.0
    elif popularity_mode == "follow_crowd":
        df["PopAdj"] = df["Own_norm"]
    else:
        df["PopAdj"] = -df["Own_norm"]

    # ---------- 6) Final RecScore weights -----------------------------------
    a1, a2, a3, a4, a5, a6, a7, a8 = 0.35, 0.20, 0.15, 0.10, 0.05, 0.10, 0.03, 0.02

    df["RecScore"] = (
        a1 * df["P_norm"]
        + a2 * df["Form_norm"]
        + a3 * df["Cons_norm"]
        + a4 * df["Venue_norm"]
        + a5 * df["Opp_norm"]
        + a6 * df["ValueEff_norm"]
        + a7 * df["RiskAdj"]
        + a8 * df["PopAdj"]
    )

    # Make sure there are no NaNs in RecScore
    df["RecScore"] = df["RecScore"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


# ---------- Team selection (DSA: heaps + dicts) ----------------------------


@dataclass
class TeamConstraints:
    """
    Constraints for team selection.

    Example:
        TeamConstraints(
            target_roles={"BAT": 4, "BOWL": 3, "AR": 3, "WK": 1},
            max_players_per_real_team=4,
            max_credits=100.0,
        )
    """

    target_roles: Dict[str, int]  # base roles: BAT, BOWL, AR, WK
    max_players_per_real_team: int = 4
    max_credits: float = 100.0


def recommend_team(scored_df: pd.DataFrame, constraints: TeamConstraints) -> pd.DataFrame:
    """
    Build a recommended XI using heaps (priority queues) & greedy selection.

    scored_df must include:
        RecScore, role_base, is_wk_bat, team, credits
    """
    df = scored_df.copy()

    # Build heaps per base role
    role_heaps: Dict[str, list] = {role: [] for role in constraints.target_roles.keys()}

    for idx, row in df.iterrows():
        base_role = row["role_base"]
        if base_role not in role_heaps:
            continue

        heap_score = row["RecScore"]

        # Prefer WK-BAT slightly within WK role
        if base_role == "WK" and bool(row.get("is_wk_bat", False)):
            heap_score += 0.1

        # Python heapq is min-heap; push negative score for max-heap behavior
        heapq.heappush(role_heaps[base_role], (-heap_score, idx))

    selected_indices = []
    team_counts: Dict[str, int] = {}
    role_counts: Dict[str, int] = {role: 0 for role in constraints.target_roles.keys()}
    total_credits = 0.0

    # For each role, fill required slots
    for role, required_count in constraints.target_roles.items():
        heap = role_heaps[role]

        while role_counts[role] < required_count and heap:
            neg_score, idx = heapq.heappop(heap)
            player = df.loc[idx]

            real_team = player["team"]
            credits = float(player["credits"])

            # Team limit: max N players from same real team
            if team_counts.get(real_team, 0) >= constraints.max_players_per_real_team:
                continue

            # Credit limit
            if total_credits + credits > constraints.max_credits:
                continue

            # Select player
            selected_indices.append(idx)
            role_counts[role] += 1
            team_counts[real_team] = team_counts.get(real_team, 0) + 1
            total_credits += credits

    team_df = df.loc[selected_indices].copy()
    team_df = team_df.sort_values("RecScore", ascending=False).reset_index(drop=True)

    # (Optional) you can print debug info from here if needed
    # print("Selected players:", len(team_df))
    # print("Total credits used:", total_credits)
    # print("Role counts:", role_counts)
    # print("Team counts:", team_counts)

    return team_df


# ---------- Captain / Vice-Captain assignment ------------------------------


def assign_captain_vice(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign Captain (C) and Vice-Captain (VC) to players in team_df.

    - Captain: highest RecScore
    - VC: next best, considering RecScore + consistency + role bonus
    """
    df = team_df.copy()
    if df.empty:
        df["C"] = False
        df["VC"] = False
        return df

    # Captain = highest RecScore
    cap_idx = df["RecScore"].idxmax()
    df["C"] = False
    df["VC"] = False
    df.loc[cap_idx, "C"] = True

    # Role bonus for VC selection
    def role_bonus(row) -> float:
        if row["role_base"] == "AR":
            return 0.3
        if row["role_base"] == "WK" and bool(row.get("is_wk_bat", False)):
            return 0.25
        if row["role_base"] == "BAT":
            return 0.15
        return 0.10  # bowlers

    candidates = df.index[df.index != cap_idx]
    vc_scores = []
    for idx in candidates:
        row = df.loc[idx]
        score = row["RecScore"] + 0.2 * row.get("Cons_norm", 0.0) + role_bonus(row)
        vc_scores.append((score, idx))

    if vc_scores:
        _, vc_idx = max(vc_scores, key=lambda x: x[0])
        df.loc[vc_idx, "VC"] = True

    return df


# ---------- Optional: small self-test when run directly --------------------


if __name__ == "__main__":
    # Tiny demo with fake data (you can ignore this in production)
    dummy_players = pd.DataFrame(
        [
            # BATs
            {"player_name": "Player BAT1", "role_detailed": "BAT", "team": "MI",
             "credits": 9.0, "points_2022": 450, "points_2023": 500, "points_2024": 620},
            {"player_name": "Player BAT2", "role_detailed": "BAT", "team": "CSK",
             "credits": 8.5, "points_2022": 400, "points_2023": 420, "points_2024": 510},
            {"player_name": "Player BAT3", "role_detailed": "BAT", "team": "KKR",
             "credits": 8.0, "points_2022": 380, "points_2023": 390, "points_2024": 450},
            {"player_name": "Player BAT4", "role_detailed": "BAT", "team": "RCB",
             "credits": 9.5, "points_2022": 520, "points_2023": 480, "points_2024": 560},
            # BOWLers
            {"player_name": "Player BOWL1", "role_detailed": "BOWL", "team": "MI",
             "credits": 8.5, "points_2022": 350, "points_2023": 430, "points_2024": 510},
            {"player_name": "Player BOWL2", "role_detailed": "BOWL", "team": "CSK",
             "credits": 8.0, "points_2022": 300, "points_2023": 380, "points_2024": 490},
            {"player_name": "Player BOWL3", "role_detailed": "BOWL", "team": "KKR",
             "credits": 7.5, "points_2022": 280, "points_2023": 360, "points_2024": 470},
            # All-rounders
            {"player_name": "Player AR1", "role_detailed": "AR", "team": "MI",
             "credits": 9.0, "points_2022": 420, "points_2023": 510, "points_2024": 640},
            {"player_name": "Player AR2", "role_detailed": "AR", "team": "CSK",
             "credits": 9.2, "points_2022": 410, "points_2023": 490, "points_2024": 630},
            {"player_name": "Player AR3", "role_detailed": "AR", "team": "KKR",
             "credits": 8.8, "points_2022": 390, "points_2023": 470, "points_2024": 600},
            # WKs
            {"player_name": "Player WK1", "role_detailed": "WK-BAT", "team": "RCB",
             "credits": 8.7, "points_2022": 380, "points_2023": 450, "points_2024": 580},
            {"player_name": "Player WK2", "role_detailed": "WK-only", "team": "MI",
             "credits": 7.0, "points_2022": 200, "points_2023": 210, "points_2024": 230},
        ]
    )

    scored = compute_rec_scores(dummy_players)
    cons = TeamConstraints(
        target_roles={"BAT": 4, "BOWL": 3, "AR": 3, "WK": 1},
        max_players_per_real_team=4,
        max_credits=100.0,
    )
    team = recommend_team(scored, cons)
    team = assign_captain_vice(team)

    print(team[["player_name", "role_detailed", "role_base", "team", "credits", "RecScore", "C", "VC"]])
