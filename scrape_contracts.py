import pandas as pd
import requests
import io

def fetch_nflverse_contracts():
    # contracts release has parquet + csv.gz, no plain csv
    # try parquet first via nfl_data_py which wraps nflverse
    import nfl_data_py as nfl

    print("Fetching contracts via nfl_data_py...")
    try:
        contracts = nfl.import_contracts()
        print(f"Raw contracts: {contracts.shape}")
        print(f"Columns: {contracts.columns.tolist()}")
        print(contracts.head(5).to_string())
        contracts.to_csv('contracts_raw.csv', index=False)
        print("\nSaved contracts_raw.csv")
        return contracts
    except Exception as e:
        print(f"nfl_data_py failed: {e}")

    # fallback — direct parquet URL
    print("\nTrying direct parquet URL...")
    url = "https://github.com/nflverse/nflverse-data/releases/download/contracts/contracts.parquet"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    contracts = pd.read_parquet(io.BytesIO(response.content))
    print(f"Raw contracts: {contracts.shape}")
    print(f"Columns: {contracts.columns.tolist()}")
    print(contracts.head(5).to_string())
    contracts.to_csv('contracts_raw.csv', index=False)
    print("\nSaved contracts_raw.csv")
    return contracts

if __name__ == '__main__':
    fetch_nflverse_contracts()