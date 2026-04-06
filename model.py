import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

MIN_SALARY = 1.1

# Production minimums to qualify as legitimate starter
# These filter garbage time stats and pure backups
MIN_ATTEMPTS_QB = 100   # ~6+ att/game over 17 games
MIN_CARRIES_RB  = 80    # ~5+ carries/game
MIN_TARGETS_WR  = 40    # ~2.5+ targets/game
MIN_TARGETS_TE  = 30    # slightly lower for TEs

# Rookie deal = CBA-priced, not market-priced
# Use draft_year + year_signed to detect — NOT age/seasons which misclassifies
# young stars like Lawrence/Hurts on big extensions
ROOKIE_APY_THRESHOLD = 8_000_000  # under $8M = likely rookie/minimum deal

POSITION_FEATURES = {
    'QB': [
        'age', 'seasons', 'games', 'games_played_pct', 'avg_games_played_pct',
        'passing_yards_adj', 'passing_tds_adj', 'interceptions',
        'completion_pct', 'yards_per_attempt', 'passing_td_rate',
        'rushing_yards_adj', 'rushing_tds_adj',
        'passing_yards_pg', 'passing_tds_pg',
        'best_passing_yards_pg', 'best_passing_tds_pg', 'avg_passing_yards_pg',
        'peak_passing_yards_adj', 'peak_passing_tds_adj',
        'playoff_games', 'playoff_rate',
        'playoff_yards_per_game', 'playoff_tds_per_game',
        'guarantee_pct', 'fa_year', 'is_rookie_deal',
        'inflated_apy_ref', 'draft_round',
    ],
    'RB': [
        'age', 'seasons', 'games', 'games_played_pct', 'avg_games_played_pct',
        'rushing_yards_adj', 'rushing_tds_adj', 'carries',
        'receiving_yards_adj', 'receiving_tds_adj', 'receptions_adj', 'targets',
        'rushing_yards_pg', 'rushing_tds_pg',
        'receiving_yards_pg', 'receptions_pg',
        'best_rushing_yards_pg', 'avg_rushing_yards_pg', 'avg_receiving_yards_pg',
        'peak_rushing_yards_adj', 'peak_rushing_tds_adj', 'peak_receptions_adj',
        'yards_per_carry', 'catch_rate',
        'playoff_games', 'playoff_rate',
        'playoff_yards_per_game', 'playoff_tds_per_game',
        'guarantee_pct', 'fa_year', 'is_rookie_deal',
        'inflated_apy_ref', 'draft_round',
    ],
    'WR': [
        'age', 'seasons', 'games', 'games_played_pct', 'avg_games_played_pct',
        'receiving_yards_adj', 'receiving_tds_adj', 'receptions_adj', 'targets',
        'receiving_yards_pg', 'receiving_tds_pg', 'receptions_pg',
        'best_receiving_yards_pg', 'avg_receiving_yards_pg',
        'best_receptions_pg', 'avg_receptions_pg',
        'peak_receiving_yards_adj', 'peak_receiving_tds_adj', 'peak_receptions_adj',
        'yards_per_target', 'catch_rate',
        'playoff_games', 'playoff_rate',
        'playoff_yards_per_game', 'playoff_tds_per_game',
        'guarantee_pct', 'fa_year', 'is_rookie_deal',
        'inflated_apy_ref', 'draft_round',
    ],
    'TE': [
        'age', 'seasons', 'games', 'games_played_pct', 'avg_games_played_pct',
        'receiving_yards_adj', 'receiving_tds_adj', 'receptions_adj', 'targets',
        'receiving_yards_pg', 'receptions_pg',
        'best_receiving_yards_pg', 'avg_receiving_yards_pg',
        'peak_receiving_yards_adj', 'peak_receiving_tds_adj', 'peak_receptions_adj',
        'yards_per_target', 'catch_rate',
        'playoff_games', 'playoff_rate',
        'playoff_yards_per_game', 'playoff_tds_per_game',
        'guarantee_pct', 'fa_year', 'is_rookie_deal',
        'inflated_apy_ref', 'draft_round',
    ],
}

ARCHETYPE_LABELS = {
    'QB': {0: 'Franchise Cornerstone', 1: 'Dual-Threat Playmaker',
           2: 'Game Manager', 3: 'Backup / Bridge'},
    'RB': {0: 'Workhorse Back', 1: 'Receiving Back',
           2: 'Power Runner', 3: 'Depth / Specialist'},
    'WR': {0: 'WR1 / Alpha', 1: 'Slot Technician',
           2: 'Deep Threat', 3: 'Role Player'},
    'TE': {0: 'Receiving Weapon', 1: 'Balanced Threat',
           2: 'Blocking TE', 3: 'Backup'},
}

CLUSTER_FEATURES = {
    'QB': ['best_passing_yards_pg', 'best_passing_tds_pg', 'completion_pct',
           'yards_per_attempt', 'rushing_yards_pg', 'playoff_rate', 'age'],
    'RB': ['best_rushing_yards_pg', 'yards_per_carry', 'avg_receiving_yards_pg',
           'receptions_pg', 'catch_rate', 'playoff_rate', 'age'],
    'WR': ['best_receiving_yards_pg', 'best_receptions_pg', 'yards_per_target',
           'catch_rate', 'peak_receiving_tds_adj', 'playoff_rate', 'age'],
    'TE': ['best_receiving_yards_pg', 'receptions_pg', 'yards_per_target',
           'receiving_tds_pg', 'catch_rate', 'playoff_rate', 'age'],
}

def build_features(df):
    df = df.copy()

    df['completion_pct']    = np.where(df['attempts'] > 0, df['completions'] / df['attempts'], 0)
    df['yards_per_attempt'] = np.where(df['attempts'] > 0, df['passing_yards'] / df['attempts'], 0)
    df['passing_td_rate']   = np.where(df['attempts'] > 0, df['passing_tds'] / df['attempts'], 0)
    df['yards_per_carry']   = np.where(df['carries'] > 0, df['rushing_yards'] / df['carries'], 0)
    df['yards_per_target']  = np.where(df['targets'] > 0, df['receiving_yards'] / df['targets'], 0)
    df['catch_rate']        = np.where(df['targets'] > 0, df['receptions'] / df['targets'], 0)

    for base, col in [('passing_yards', 'passing_yards_pg'),
                      ('passing_tds', 'passing_tds_pg'),
                      ('rushing_yards', 'rushing_yards_pg'),
                      ('rushing_tds', 'rushing_tds_pg'),
                      ('receiving_yards', 'receiving_yards_pg'),
                      ('receiving_tds', 'receiving_tds_pg'),
                      ('receptions', 'receptions_pg')]:
        if col not in df.columns and base in df.columns:
            df[col] = df[base] / df['games'].clip(lower=1)

    df['playoff_rate'] = df['made_playoffs'] / df['seasons'].clip(lower=1)

    df['playoff_yards_per_game'] = np.where(
        df['playoff_games'] > 0,
        (df['playoff_passing_yards'] + df['playoff_rushing_yards'] +
         df['playoff_receiving_yards']) / df['playoff_games'], 0)

    df['playoff_tds_per_game'] = np.where(
        df['playoff_games'] > 0,
        (df['playoff_passing_tds'] + df['playoff_rushing_tds'] +
         df['playoff_receiving_tds']) / df['playoff_games'], 0)

    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(26)

    # ── Rookie deal detection — APY based only ────────────────────────────────
    # Do NOT use age or seasons — young stars on extensions get misclassified
    # $8M threshold: below this is almost always a CBA rookie/minimum contract
    df['is_rookie_deal'] = (df['apy'] < ROOKIE_APY_THRESHOLD).astype(int)

    # Guarantee pct
    if 'guarantee_pct' not in df.columns:
        df['guarantee_pct'] = 0.0
    df['guarantee_pct'] = pd.to_numeric(df['guarantee_pct'], errors='coerce').fillna(0)

    # Draft round — 8 = undrafted
    if 'draft_round' in df.columns:
        df['draft_round'] = pd.to_numeric(df['draft_round'], errors='coerce').fillna(8)
    else:
        df['draft_round'] = 8.0

    # Cap-inflation-adjusted reference APY
    if 'inflated_apy' in df.columns:
        df['inflated_apy_ref'] = pd.to_numeric(df['inflated_apy'], errors='coerce').fillna(0)
    else:
        df['inflated_apy_ref'] = 0.0

    # Fill missing adj cols
    for base in ['passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds',
                 'receiving_yards', 'receiving_tds', 'receptions']:
        adj = f'{base}_adj'
        if adj not in df.columns:
            df[adj] = df.get(base, 0)
        peak_adj = f'peak_{base}_adj'
        if peak_adj not in df.columns:
            df[peak_adj] = df.get(f'peak_{base}', df.get(base, 0))

    for rate in ['passing_yards_pg', 'passing_tds_pg', 'rushing_yards_pg',
                 'rushing_tds_pg', 'receiving_yards_pg', 'receiving_tds_pg',
                 'receptions_pg']:
        for prefix in ['best_', 'avg_']:
            col = f'{prefix}{rate}'
            if col not in df.columns:
                df[col] = df.get(rate, 0)

    for col in ['avg_receptions_pg', 'best_receptions_pg']:
        if col not in df.columns:
            df[col] = df.get('receptions_pg', 0)

    return df

def get_legitimate_market_mask(df_pos, pos):
    """
    Legitimate market deal = on a real market contract (APY >= $8M)
    AND meets minimum production threshold for position.
    This filters both rookie CBA deals AND garbage time backups.
    """
    not_rookie = df_pos['is_rookie_deal'] == 0

    if pos == 'QB':
        prod = df_pos.get('attempts', pd.Series(0, index=df_pos.index)) >= MIN_ATTEMPTS_QB
    elif pos == 'RB':
        prod = df_pos.get('carries', pd.Series(0, index=df_pos.index)) >= MIN_CARRIES_RB
    elif pos == 'WR':
        prod = df_pos.get('targets', pd.Series(0, index=df_pos.index)) >= MIN_TARGETS_WR
    elif pos == 'TE':
        prod = df_pos.get('targets', pd.Series(0, index=df_pos.index)) >= MIN_TARGETS_TE
    else:
        prod = pd.Series(True, index=df_pos.index)

    return not_rookie & prod

def cluster_players(df_pos, pos, n_clusters=4):
    feats = [f for f in CLUSTER_FEATURES[pos] if f in df_pos.columns]
    X = df_pos[feats].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    cluster_apy = pd.Series(df_pos['apy_m'].values).groupby(labels).mean()
    rank_map = {old: new for new, old in
                enumerate(cluster_apy.sort_values(ascending=False).index)}
    ranked = np.array([rank_map[l] for l in labels])
    archetype_map = ARCHETYPE_LABELS.get(pos, {})
    return [archetype_map.get(l, f'Cluster {l}') for l in ranked], km, scaler, feats

def train_position_model(df_pos, pos, features):
    print(f"\n── {pos} ({len(df_pos)} players) ──")
    market_mask  = get_legitimate_market_mask(df_pos, pos)
    market_count = market_mask.sum()
    print(f"  Legitimate market deals: {market_count} | Excluded: {len(df_pos) - market_count}")

    X = df_pos[features].fillna(0)
    y = df_pos['apy_m']

    if len(df_pos) < 15:
        model = XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        model.fit(X, y)
        preds = np.clip(model.predict(X), MIN_SALARY, None)
        print(f"  Train MAE: ${mean_absolute_error(y, preds):.2f}M (small sample)")
        return model, preds

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=500, learning_rate=0.02, max_depth=3,
        subsample=0.75, colsample_bytree=0.75,
        min_child_weight=5, reg_alpha=0.2, reg_lambda=2.5,
        gamma=0.1, random_state=42, verbosity=0)
    model.fit(X_train, y_train)

    y_pred = np.clip(model.predict(X_test), MIN_SALARY, None)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    try:
        cv = cross_val_score(model, X, y, cv=4, scoring='r2')
        print(f"  MAE: ${mae:.2f}M | R²: {r2:.3f} | CV R²: {cv.mean():.3f} ± {cv.std():.3f}")
    except Exception:
        print(f"  MAE: ${mae:.2f}M | R²: {r2:.3f}")

    importance = pd.Series(model.feature_importances_, index=features)
    print(f"  Top 5: {importance.nlargest(5).index.tolist()}")

    return model, np.clip(model.predict(X), MIN_SALARY, None)

def train_model():
    df = pd.read_parquet('merged_data.parquet')
    print(f"Training on {len(df)} players")
    print(df.groupby('position')['apy'].count())

    df = build_features(df)
    df['apy_m'] = df['apy'] / 1_000_000

    all_predictions, all_models, all_clusters = [], {}, {}

    for pos, features in POSITION_FEATURES.items():
        df_pos = df[df['position'] == pos].copy().reset_index(drop=True)
        if df_pos.empty:
            continue

        features = [f for f in features if f in df_pos.columns]

        # Cluster
        n_clusters = min(4, max(2, len(df_pos) // 8))
        archetypes, km, scaler, cfeats = cluster_players(df_pos, pos, n_clusters)
        df_pos['archetype'] = archetypes
        all_clusters[pos]   = (km, scaler, cfeats)

        print(f"\n  {pos} Archetypes:")
        arch = (df_pos.groupby('archetype')
                .agg(n=('player_name','count'), avg_apy=('apy_m','mean'),
                     avg_age=('age','mean'), example=('player_name', lambda x: x.iloc[0]))
                .sort_values('avg_apy', ascending=False))
        for a, r in arch.iterrows():
            print(f"    {a:<28} n={r['n']:>3}  avg=${r['avg_apy']:.1f}M  "
                  f"age={r['avg_age']:.0f}  e.g. {r['example']}")

        model, preds = train_position_model(df_pos, pos, features)
        all_models[pos] = (model, features)

        df_pos['predicted_apy_m'] = preds
        df_pos['predicted_apy']   = preds * 1_000_000
        df_pos['delta_m']         = df_pos['predicted_apy_m'] - df_pos['apy_m']
        df_pos['delta_pct']       = df_pos['delta_m'] / df_pos['apy_m'] * 100
        df_pos['is_legit_market'] = get_legitimate_market_mask(df_pos, pos).astype(int)
        all_predictions.append(df_pos)

    results = pd.concat(all_predictions, ignore_index=True)
    market  = results[results['is_legit_market'] == 1].copy()

    def fmt(r):
        delta = r['delta_m']
        s = f"+${delta:.1f}M" if delta >= 0 else f"-${abs(delta):.1f}M"
        return (f"  {r['player_name']:<22} [{r['archetype']:<28}] "
                f"age={r['age']:.0f}  actual=${r['apy']/1e6:.1f}M  "
                f"predicted=${r['predicted_apy']/1e6:.1f}M  delta={s}  "
                f"guarantee={r['guarantee_pct']*100:.0f}%")

    print("\n" + "="*80)
    print("TOP 15 MOST UNDERPAID (legitimate starters on market deals)")
    print("="*80)
    for _, r in market.nlargest(15, 'delta_m').iterrows():
        print(fmt(r))

    print("\n" + "="*80)
    print("TOP 15 MOST OVERPAID (legitimate starters on market deals)")
    print("="*80)
    for _, r in market.nsmallest(15, 'delta_m').iterrows():
        print(fmt(r))

    for label, fn in [("UNDERPAID", lambda x: x.nlargest(5, 'delta_m')),
                      ("OVERPAID",  lambda x: x.nsmallest(5, 'delta_m'))]:
        print(f"\n{'='*80}\nBY POSITION — {label}\n{'='*80}")
        for pos in ['QB', 'RB', 'WR', 'TE']:
            pos_df = fn(market[market['position'] == pos])
            print(f"\n{pos}:")
            for _, r in pos_df.iterrows():
                print(fmt(r))

    # Save
    joblib.dump(all_models,   'models.joblib')
    joblib.dump(all_clusters, 'clusters.joblib')

    save_cols = [c for c in [
        'player_name', 'position', 'archetype', 'seasons', 'games', 'age',
        'games_played_pct', 'avg_games_played_pct',
        'passing_yards_adj', 'passing_tds_adj', 'attempts', 'completions',
        'rushing_yards_adj', 'rushing_tds_adj', 'carries',
        'receiving_yards_adj', 'receiving_tds_adj', 'receptions_adj', 'targets',
        'passing_yards_pg', 'rushing_yards_pg', 'receiving_yards_pg',
        'best_passing_yards_pg', 'best_receiving_yards_pg', 'best_rushing_yards_pg',
        'peak_passing_yards_adj', 'peak_receiving_yards_adj', 'peak_rushing_yards_adj',
        'completion_pct', 'yards_per_attempt', 'yards_per_carry', 'yards_per_target',
        'catch_rate', 'playoff_games', 'playoff_yards_per_game', 'playoff_tds_per_game',
        'playoff_rate', 'made_playoffs',
        'guarantee_pct', 'fa_year', 'is_rookie_deal', 'is_legit_market',
        'draft_round', 'team',
        'apy', 'predicted_apy', 'delta_m', 'delta_pct'
    ] if c in results.columns]

    results[save_cols].to_parquet('predictions.parquet', index=False)
    results[save_cols].to_csv('predictions.csv', index=False)
    print(f"\nSaved predictions.parquet ({len(results)} players, "
          f"{len(market)} legitimate market deals)")

if __name__ == '__main__':
    train_model()