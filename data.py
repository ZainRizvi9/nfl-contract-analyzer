import nfl_data_py as nfl
import pandas as pd
import numpy as np

# ── Pull seasonal stats 2021-2024 ─────────────────────────────────────────────
print("Fetching seasonal stats...")
stats = nfl.import_seasonal_data(list(range(2021, 2025)))

stat_cols = [
    'player_id', 'season', 'season_type',
    'passing_yards', 'passing_tds', 'interceptions', 'attempts', 'completions',
    'rushing_yards', 'rushing_tds', 'carries',
    'receiving_yards', 'receiving_tds', 'receptions', 'targets',
    'games', 'fantasy_points_ppr'
]
stat_cols = [c for c in stat_cols if c in stats.columns]
stats = stats[stat_cols]

reg      = stats[stats['season_type'] == 'REG'].copy()
playoffs = stats[stats['season_type'] == 'POST'].copy()

playoffs = playoffs.rename(columns={
    'passing_yards':      'playoff_passing_yards',
    'passing_tds':        'playoff_passing_tds',
    'interceptions':      'playoff_interceptions',
    'rushing_yards':      'playoff_rushing_yards',
    'rushing_tds':        'playoff_rushing_tds',
    'receiving_yards':    'playoff_receiving_yards',
    'receiving_tds':      'playoff_receiving_tds',
    'receptions':         'playoff_receptions',
    'targets':            'playoff_targets',
    'games':              'playoff_games',
    'fantasy_points_ppr': 'playoff_fantasy_points',
})

playoff_cols = ['player_id', 'season'] + \
    [c for c in playoffs.columns if c.startswith('playoff_')]
playoffs = playoffs[playoff_cols]

merged = reg.merge(playoffs, on=['player_id', 'season'], how='left')
playoff_stat_cols = [c for c in merged.columns if c.startswith('playoff_')]
merged[playoff_stat_cols] = merged[playoff_stat_cols].fillna(0)
merged['made_playoffs'] = (merged['playoff_games'] > 0).astype(int)

# ── Injury adjustment — scale short seasons to 17-game pace ──────────────────
# Players with <12 games had their stats suppressed by injury
# We project what they would have done over a full season
FULL_SEASON = 17
INJURY_THRESHOLD = 12

raw_stat_cols = [
    'passing_yards', 'passing_tds', 'interceptions', 'attempts', 'completions',
    'rushing_yards', 'rushing_tds', 'carries',
    'receiving_yards', 'receiving_tds', 'receptions', 'targets',
    'fantasy_points_ppr'
]

for col in raw_stat_cols:
    if col in merged.columns:
        adj_col = f'{col}_adj'
        merged[adj_col] = np.where(
            merged['games'] < INJURY_THRESHOLD,
            merged[col] * (FULL_SEASON / merged['games'].clip(lower=1)),
            merged[col]
        )

# ── Per-game rates — more stable than totals for injury-affected players ──────
for col in raw_stat_cols:
    if col in merged.columns:
        merged[f'{col}_pg'] = merged[col] / merged['games'].clip(lower=1)

merged['games_played_pct'] = merged['games'] / FULL_SEASON

# ── Roster data for name, position, age, draft info ──────────────────────────
print("Fetching roster data...")
rosters = nfl.import_seasonal_rosters(list(range(2021, 2025)))
print("Roster columns:", rosters.columns.tolist())

name_col  = next((c for c in ['player_name', 'full_name', 'name'] if c in rosters.columns), None)
pos_col   = next((c for c in ['position', 'depth_chart_position'] if c in rosters.columns), None)
age_col   = next((c for c in ['age'] if c in rosters.columns), None)
draft_col = next((c for c in ['draft_number'] if c in rosters.columns), None)
exp_col   = next((c for c in ['years_exp'] if c in rosters.columns), None)

roster_keep = ['player_id', 'season']
rename_map = {}
if name_col:
    roster_keep.append(name_col)
    rename_map[name_col] = 'player_name'
if pos_col:
    roster_keep.append(pos_col)
    rename_map[pos_col] = 'position'
if age_col:
    roster_keep.append(age_col)
    rename_map[age_col] = 'age'
if draft_col:
    roster_keep.append(draft_col)
    rename_map[draft_col] = 'draft_number'
if exp_col:
    roster_keep.append(exp_col)
    rename_map[exp_col] = 'years_exp'

rosters = (rosters[roster_keep]
           .rename(columns=rename_map)
           .drop_duplicates(subset=['player_id', 'season']))

if 'age' in rosters.columns:
    rosters['age'] = pd.to_numeric(rosters['age'], errors='coerce')

merged = merged.merge(rosters, on=['player_id', 'season'], how='left')
merged = merged.dropna(subset=['player_name', 'position'])

print(f"\nFinal stats shape: {merged.shape}")
print(f"Age null count: {merged['age'].isna().sum()}")
print(f"Unique ages sample: {sorted(merged['age'].dropna().unique().tolist())[:10]}")
print(merged[['player_id', 'player_name', 'position', 'age', 'season', 'games', 'games_played_pct']].head(10))

merged.to_parquet('player_stats.parquet', index=False)
print("\nSaved player_stats.parquet")