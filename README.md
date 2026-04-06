# NFL Contract Value Analyzer

Predicts what NFL players should be paid based on their stats, then finds who is getting overpaid or underpaid relative to their market value.

[Live App](https://nfl-contract-analyzer.streamlit.app) · [Video](https://youtu.be/OokofsgDGGM) · [GitHub](https://github.com/ZainRizvi9/nfl-contract-analyzer)

## Overview

The NFL has no public tool that tells you whether a contract is good value relative to what a player actually produces. This project builds one. Separate XGBoost models are trained per position using four years of stats, and the predicted APY is compared against the actual contract to surface the biggest mismatches.

Notable findings: Deshaun Watson at $46M APY with 7 games played per season surfaces as the most overpaid player in the dataset. Patrick Mahomes and CJ Stroud surface as the most underpaid relative to their production.

## Model Performance

| Position | R2 | MAE |
|----------|-----|-----|
| QB | 0.961 | $4.38M |
| RB | 0.990 | $0.36M |
| WR | 0.969 | $0.94M |
| TE | 0.937 | $0.84M |

## How It Works

**Data sources**

Stats come from nflverse via nfl_data_py, covering 2021 to 2024 regular season and playoff splits. Contract data comes from nflverse's Over The Cap integration, filtered from 50,000+ raw records down to 655 active skill position players.

**Pipeline**

    data.py              pulls and processes seasonal stats
    scrape_contracts.py  pulls contract data from nflverse
    clean_contracts.py   filters and cleans contract records
    merge.py             joins stats and contracts on GSIS player ID
    model.py             trains XGBoost models and K-Means clusters
    app.py               Streamlit dashboard

**Feature engineering**

Stats for players with fewer than 12 games are scaled to a 17-game pace so injured players are compared fairly. Every counting stat gets a per-game rate version. Career peak stats capture a player's ceiling across all seasons. Guarantee percentage is included as a proxy for how much the team values the player.

**Rookie deal handling**

Contracts under $8M APY are flagged as rookie deals and excluded from leaderboards. CBA-priced contracts are not market-priced so comparing them against market predictions produces misleading results.

**Clustering**

K-Means runs separately per position on per-game production rates and age to discover player archetypes without predefined labels. Four clusters per position emerge naturally from the data, for example WR1/Alpha, Slot Technician, Deep Threat, and Role Player for wide receivers.

## Stack

Python, XGBoost, scikit-learn, pandas, Streamlit, Plotly, nfl_data_py

## Running Locally

    git clone https://github.com/ZainRizvi9/nfl-contract-analyzer.git
    cd nfl-contract-analyzer
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    streamlit run app.py

To rebuild the model from scratch:

    python data.py
    python scrape_contracts.py
    python clean_contracts.py
    python merge.py
    python model.py

## Limitations

Defensive players are not included since defensive value is hard to quantify with public stats. The model cannot capture future-value contracts signed on projected upside, which is why players like Ja'Marr Chase are excluded from the overpaid leaderboard. RB predictions are less reliable due to the small number of market-rate RB contracts in training data.

---
*Not affiliated with any NFL organization*
