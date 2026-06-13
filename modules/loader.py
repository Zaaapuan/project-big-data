import pandas as pd
import sys
from pathlib import Path

def load_data(filename):
    """
    Read CSV file into a pandas DataFrame from big_data directory.
    Only supports .csv extension.
    """
    ext = Path(filename).suffix.lower()
    if ext != '.csv':
        print(f"Error: Format file '{ext}' tidak dikenali. Program hanya mendukung .csv.")
        sys.exit(1)

    filepath = Path('big_data') / filename
    if not filepath.exists():
        print(f"Error: File '{filepath}' tidak ditemukan.")
        sys.exit(1)
    
    try:
        # Load CSV, treating ',' as thousands separator in case of numeric columns
        df = pd.read_csv(filepath, thousands=',', na_values=['-'])
    except Exception as e:
        print(f"Error saat membaca file CSV: {e}")
        sys.exit(1)
        
    initial_len = len(df)
    if df.isna().any().any():
        df = df.dropna().reset_index(drop=True)
        deleted_rows = initial_len - len(df)
        print(f"Informasi: Ditemukan baris kosong. Sebanyak {deleted_rows} baris dihapus.")
        
    return df
