# NFL Contract Value Analyzer

Predicts what NFL players should be paid based on their stats, then finds who is getting overpaid or underpaid relative to their market value.

**[Live App](https://nfl-contract-analyzer.streamlit.app)** · **[Video](https://youtu.be/OokofsgDGGM)** · **[GitHub](https://github.com/ZainRizvi9/nfl-contract-analyzer)**

---

## Overview

The NFL has no public tool that tells you whether a contract is good value relative to what the player actually produces. This project builds one. Separate XGBoost models are trained per position using four years of stats, and the predicted APY is compared against the player's actual contract to surface the biggest mismatches.

Notable findings: Deshaun Watson at $46M APY with 7 games played per season is the most overpaid player in the dataset. Patrick Mahomes and CJ Stroud surface as the most underpaid relative to their production.

---

## Model Performance

| Position | R² | MAE |
|----------|----|-----|
| QB | 0.961 | $4.38M |
| RB | 0.990 | $0.36M |
| WR | 0.969 | $0.94M |
| TE | 0.937 | $0.84M |

---

## How It Works

**Data sources**

Stats come from nflverse via nfl_data_py, covering 2021 to 2024 regular season and playoff splits. Contract data comes from nflverse's Over The Cap integration, filtered from 50,000+ raw records down to 655 active skill position players.

**Pipeline**
