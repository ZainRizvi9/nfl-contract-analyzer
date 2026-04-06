**Feature engineering**

Stats for players with fewer than 12 games are scaled to a 17-game pace so injured players are compared fairly. Every counting stat also gets a per-game rate version. Career peak stats capture a player's ceiling across all available seasons. Guarantee percentage is included as a proxy for how much the team actually values the player.

**Rookie deal handling**

Contracts under $8M APY are flagged as rookie deals and excluded from leaderboards. CBA-priced rookie contracts are not market-priced so comparing them against market predictions produces misleading results.

**Clustering**

K-Means runs separately per position on per-game production rates and age to discover player archetypes without predefined labels. Four clusters per position emerge naturally from the data, for example WR1/Alpha, Slot Technician, Deep Threat, and Role Player for wide receivers.

---

## Stack

Python, XGBoost, scikit-learn, pandas, Streamlit, Plotly, nfl_data_py

---

## Running Locally
```bash
git clone https://github.com/ZainRizvi9/nfl-contract-analyzer.git
cd nfl-contract-analyzer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To rebuild the model from scratch:
```bash
python data.py
python scrape_contracts.py
python clean_contracts.py
python merge.py
python model.py
```

---

## Limitations

Defensive players are not included since defensive value is hard to quantify with public stats. The model cannot capture future-value contracts signed on projected upside, which is why players like Ja'Marr Chase are excluded from the overpaid leaderboard. RB predictions are less reliable due to the small number of market-rate RB contracts in the training data.

---

*Not affiliated with any NFL organization*
