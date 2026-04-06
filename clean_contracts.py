import pandas as pd

def clean_contracts():
    df = pd.read_csv('contracts_raw.csv', low_memory=False)
    print(f"Raw: {df.shape}")

    skill_positions = ['QB', 'RB', 'WR', 'TE']
    df = df[df['position'].isin(skill_positions)].copy()
    print(f"Skill positions: {df.shape}")

    # Active contracts only
    df = df[df['is_active'] == True].copy()
    print(f"Active only: {df.shape}")

    # Clean numeric cols — apy/guaranteed already in $M from nflverse
    df['apy_m']         = pd.to_numeric(df['apy'], errors='coerce')
    df['guaranteed_m']  = pd.to_numeric(df['guaranteed'], errors='coerce')
    df['value_m']       = pd.to_numeric(df['value'], errors='coerce')
    df['year_signed']   = pd.to_numeric(df['year_signed'], errors='coerce')

    # Drop rows with no meaningful APY
    df = df[df['apy_m'] > 0].copy()

    # Convert to dollars for pipeline consistency
    df['apy']             = df['apy_m'] * 1_000_000
    df['total_value']     = df['value_m'] * 1_000_000
    df['total_guaranteed']= df['guaranteed_m'] * 1_000_000
    df['guarantee_pct']   = (df['guaranteed_m'] / df['value_m']).clip(0, 1)
    df['guarantee_pct']   = df['guarantee_pct'].fillna(0)

    # Draft features
    df['draft_round']   = pd.to_numeric(df['draft_round'], errors='coerce')
    df['draft_overall'] = pd.to_numeric(df['draft_overall'], errors='coerce')
    df['draft_year']    = pd.to_numeric(df['draft_year'], errors='coerce')

    # FA year
    df['years']   = pd.to_numeric(df['years'], errors='coerce').fillna(1)
    df['fa_year'] = df['year_signed'] + df['years']

    # Clean gsis_id
    df['gsis_id'] = df['gsis_id'].astype(str).str.strip()
    df = df[~df['gsis_id'].isin(['None', 'nan', '', 'NaN'])]
    print(f"After gsis_id filter: {df.shape}")

    # Keep most recent + highest APY contract per player
    # Sort by year_signed desc, then apy desc — take first (most recent, highest paid)
    df = (df.sort_values(['year_signed', 'apy_m'], ascending=[False, False])
            .groupby('gsis_id')
            .first()
            .reset_index())
    print(f"One contract per player: {df.shape}")

    keep = [
        'gsis_id', 'player', 'position', 'team',
        'year_signed', 'years', 'fa_year',
        'apy', 'total_value', 'total_guaranteed', 'guarantee_pct',
        'draft_round', 'draft_overall', 'draft_year',
        'inflated_apy',
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    print(f"\nFinal contracts: {df.shape}")
    print(f"Position breakdown:\n{df['position'].value_counts()}")
    print(f"APY range: ${df['apy'].min()/1e6:.1f}M - ${df['apy'].max()/1e6:.1f}M")
    print(df[['player','position','gsis_id','apy','guarantee_pct',
              'draft_round','year_signed']].head(15).to_string())

    df.to_parquet('contracts_clean.parquet', index=False)
    df.to_csv('contracts_clean.csv', index=False)
    print("\nSaved contracts_clean.parquet + .csv")

if __name__ == '__main__':
    clean_contracts()