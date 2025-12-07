# AI IPL Fantasy Team Recommender (ML + Flask)

An AI-powered web app that recommends the **best playing XI** and **Fantasy Captain/Vice-Captain** for each IPL team using historical performance data and a machine learning model.

The project is built with **Flask**, **Pandas**, and **scikit-learn**, with a clean UI for exploring team-wise insights.

---

## 🔍 Overview

This project analyzes IPL player performance data (batting, bowling, fantasy points, etc.) and:

- Builds a **data-driven Best XI** for each franchise
- Selects a **Captain (C)** and **Vice-Captain (VC)** using:
  - Latest season performance (e.g. 2024)
  - Player role (All-rounder / Batter / Bowler)
  - A custom **captain score** formula

The app is designed as a **fantasy cricket assistant** for IPL-style leagues.

---

## 🚀 Live Demo

> 🔗 **Live URL:** _coming soon_  
> (Will be updated once deployed to Render / Railway.)

---

## ✨ Features

### 🏏 Team Insights (Last 2 Seasons)

For each IPL team, the app shows:

- **Top batting average**
- **Highest run scorer**
- **Top wicket-taker**
- **Best economy bowler**

All calculated using the most recent available seasons in the dataset.

### 🧠 AI-Recommended Best XI

For a selected team, the app:

- Uses a **machine learning model** (`fantasy_model.pkl`)  
  or a fallback formula using runs, wickets, & economy
- Ranks players by predicted **fantasy score**
- Builds a **balanced XI**:
  - 4 Batters
  - 3 All-rounders
  - 4 Bowlers
  - Fills remaining spots by overall predicted score

### 🎯 Smart Captain & Vice-Captain Logic

Instead of just picking the top 2 scorers, the app:

- Looks at **latest season only** (e.g. 2024) to compute a performance metric
- Adds a **role-based bonus**:
  - All-rounder > Batter > Bowler
- Calculates a **captain_score = latest_season_performance + role_bonus**
- Picks:
  - **Captain (C)** = highest captain_score
  - **Vice-Captain (VC)** = second-highest captain_score

This better reflects real fantasy strategies, where all-rounders and in-form top-order batters are preferred.

---

## 🧱 Tech Stack

- **Backend:** Python, Flask
- **ML / Data:** Pandas, NumPy, scikit-learn
- **Frontend:** HTML, CSS (custom responsive layout)
- **Model:** Pickle file (`models/fantasy_model.pkl`)
- **Data:** CSV file (`player_season_stats.csv`)

---

## 📁 Project Structure

```text
.
├── app.py                     # Main Flask app
├── player_season_stats.csv    # IPL player season-level stats
├── models/
│   └── fantasy_model.pkl      # Trained ML model for fantasy score
├── templates/
│   ├── index.html             # Home page with team cards
│   └── team_detail.html       # Team insights + Best XI + C/VC
└── static/
    ├── css/
    │   └── style.css          # App styling
    └── logos/                 # IPL team logos
