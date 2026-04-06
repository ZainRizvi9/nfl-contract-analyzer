import pandas as pd
import numpy as np

def build_merged_dataset():
    stats     = pd.read_parquet('player_stats.parquet')
    contracts = pd.read_parquet('contracts_clean.parquet')

    print(f"Stats: {stats.shape}, Contracts: {contracts.shape}")
    print(f"Stats player_id sample: {stats['player_id'].head(3).tolist()}")
    print(f"Contracts gsis_id sample: {contracts['gsis_id'].head(3).tolist()}")

    skill_positions = ['QB', 'RB', 'WR', 'TE']
    stats = stats[stats['position'].isin(skill_positions)].copy()

    # ── Most recent season stats per player ───────────────────────────────────
    latest = (stats.sort_values('season')
              .groupby('player_id')
              .last()
              .reset_index())

    # ── Career aggregates ─────────────────────────────────────────────────────
    agg_dict = {
        'seasons':                       ('season', 'count'),
        'career_games':                  ('games', 'sum'),
        'career_playoffs':               ('made_playoffs', 'sum'),
        'career_playoff_games':          ('playoff_games', 'sum'),
        'career_playoff_passing_yards':  ('playoff_passing_yards', 'sum'),
        'career_playoff_rushing_yards':  ('playoff_rushing_yards', 'sum'),
        'career_playoff_receiving_yards':('playoff_receiving_yards', 'sum'),
        'career_playoff_passing_tds':    ('playoff_passing_tds', 'sum'),
        'career_playoff_rushing_tds':    ('playoff_rushing_tds', 'sum'),
        'career_playoff_receiving_tds':  ('playoff_receiving_tds', 'sum'),
        'career_playoff_receptions':     ('playoff_receptions', 'sum'),
        # Peak raw
        'peak_passing_yards':            ('passing_yards', 'max'),
        'peak_passing_tds':              ('passing_tds', 'max'),
        'peak_rushing_yards':            ('rushing_yards', 'max'),
        'peak_rushing_tds':              ('rushing_tds', 'max'),
        'peak_receiving_yards':          ('receiving_yards', 'max'),
        'peak_receiving_tds':            ('receiving_tds', 'max'),
        'peak_receptions':               ('receptions', 'max'),
        'peak_targets':                  ('targets', 'max'),
        # Peak injury-adjusted
        'peak_passing_yards_adj':        ('passing_yards_adj', 'max'),
        'peak_passing_tds_adj':          ('passing_tds_adj', 'max'),
        'peak_rushing_yards_adj':        ('rushing_yards_adj', 'max'),
        'peak_rushing_tds_adj':          ('rushing_tds_adj', 'max'),
        'peak_receiving_yards_adj':      ('receiving_yards_adj', 'max'),
        'peak_receiving_tds_adj':        ('receiving_tds_adj', 'max'),
        'peak_receptions_adj':           ('receptions_adj', 'max'),
        # Best per-game rates
        'best_passing_yards_pg':         ('passing_yards_pg', 'max'),
        'best_passing_tds_pg':           ('passing_tds_pg', 'max'),
        'best_rushing_yards_pg':         ('rushing_yards_pg', 'max'),
        'best_rushing_tds_pg':           ('rushing_tds_pg', 'max'),
        'best_receiving_yards_pg':       ('receiving_yards_pg', 'max'),
        'best_receiving_tds_pg':         ('receiving_tds_pg', 'max'),
        'best_receptions_pg':            ('receptions_pg', 'max'),
        # Average per-game rates
        'avg_passing_yards_pg':          ('passing_yards_pg', 'mean'),
        'avg_rushing_yards_pg':          ('rushing_yards_pg', 'mean'),
        'avg_receiving_yards_pg':        ('receiving_yards_pg', 'mean'),
        'avg_receptions_pg':             ('receptions_pg', 'mean'),
        # Health signal
        'avg_games_played_pct':          ('games_played_pct', 'mean'),
        'min_games_played_pct':          ('games_played_pct', 'min'),
    }

    # Only aggregate columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if v[0] in stats.columns}
    career = stats.groupby('player_id').agg(**agg_dict).reset_index()
    agg = latest.merge(career, on='player_id')

    # Override single-season playoff stats with career totals
    for stat in ['playoffs', 'playoff_games', 'playoff_passing_yards',
                 'playoff_rushing_yards', 'playoff_receiving_yards',
                 'playoff_passing_tds', 'playoff_rushing_tds',
                 'playoff_receiving_tds', 'playoff_receptions']:
        career_col = f'career_{stat}'
        base_col = stat if stat != 'playoffs' else 'made_playoffs'
        if career_col in agg.columns:
            agg[base_col] = agg[career_col]

    # ── Age fix ───────────────────────────────────────────────────────────────
    agg['age'] = pd.to_numeric(agg['age'], errors='coerce')
    if 'years_exp' in agg.columns:
        agg['age'] = agg['age'].fillna(22 + agg['years_exp'])
    for pos, avg in {'QB': 27, 'RB': 25, 'WR': 26, 'TE': 26}.items():
        agg.loc[agg['position'] == pos, 'age'] = \
            agg.loc[agg['position'] == pos, 'age'].fillna(avg)

    print(f"Age — mean: {agg['age'].mean():.1f}, range: {agg['age'].min():.0f}-{agg['age'].max():.0f}")

    # ── Age bucket ────────────────────────────────────────────────────────────
    agg['age_bucket'] = pd.cut(
        agg['age'],
        bins=[0, 23, 26, 29, 33, 99],
        labels=['rookie_age', 'early_prime', 'prime', 'late_prime', 'veteran']
    ).astype(str)

# ── Min games filter ──────────────────────────────────────────────────────
    # First do a temp join to get APY, keep anyone with 8+ games OR big contract
    temp = agg.merge(contracts[['gsis_id','apy']], 
                     left_on='player_id', right_on='gsis_id', how='left')
    big_contract = temp['apy'].fillna(0) >= 20_000_000
    before = len(agg)
    agg = agg[(temp['games'].values >= 8) | big_contract.values].reset_index(drop=True)
    print(f"Removed {before - len(agg)} players with <8 games → {len(agg)} remaining")
    # ── DETERMINISTIC JOIN on player_id = gsis_id ─────────────────────────────
    # No fuzzy matching — clean exact join
    merged = agg.merge(
        contracts,
        left_on='player_id',
        right_on='gsis_id',
        how='inner'
    )

    print(f"\nDeterministic join: {len(merged)} players matched ({len(merged)/len(agg)*100:.1f}%)")
    print(f"Position breakdown:\n{merged['position_x'].value_counts()}")

    # Clean up duplicate position column from join
    merged['position'] = merged['position_x']
    merged = merged.drop(columns=['position_x', 'position_y'], errors='ignore')

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print(f"\nAge in final dataset: mean={merged['age'].mean():.1f}, range={merged['age'].min():.0f}-{merged['age'].max():.0f}")
    print(f"APY range: ${merged['apy'].min()/1e6:.1f}M - ${merged['apy'].max()/1e6:.1f}M")
    print(f"Guarantee pct: mean={merged['guarantee_pct'].mean():.2f}")
    print(f"\nSample:")
    print(merged[['player_name', 'position', 'age', 'apy', 'games',
                  'guarantee_pct', 'draft_round', 'seasons']].head(12).to_string())

    merged.to_parquet('merged_data.parquet', index=False)
    merged.to_csv('merged_data.csv', index=False)
    print(f"\nSaved merged_data.parquet + .csv ({len(merged)} players)")

if __name__ == '__main__':
    build_merged_dataset()