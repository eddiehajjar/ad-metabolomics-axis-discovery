from pathlib import Path
import pandas as pd

from src.data.load_results_table import load_results_table

RAW_POS = Path("data/raw/lipidomics/results/ST002302_AN003761_Results.txt")
RAW_NEG = Path("data/raw/lipidomics/results/ST002302_AN003762_Results.txt")

OUT_DIR = Path("data/processed/lipidomics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df_pos = load_results_table(RAW_POS)
    df_neg = load_results_table(RAW_NEG)

    # Make mz_rt the index (feature id)
    df_pos = df_pos.set_index("mz_rt")
    df_neg = df_neg.set_index("mz_rt")

    df_pos.to_parquet(OUT_DIR / "pos.parquet")
    df_neg.to_parquet(OUT_DIR / "neg.parquet")

    print("Wrote:")
    print(" ", OUT_DIR / "pos.parquet", df_pos.shape)
    print(" ", OUT_DIR / "neg.parquet", df_neg.shape)

if __name__ == "__main__":
    main()