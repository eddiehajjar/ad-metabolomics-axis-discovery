from pathlib import Path
from src.data.load_mwtab import load_mwtab

pos = Path("data/raw/lipidomics/ST002302_lipidomics_pos.mwtab.txt")
neg = Path("data/raw/lipidomics/ST002302_lipidomics_neg.mwtab.txt")

for p in [pos, neg]:
    parsed = load_mwtab(p)
    print("\n===", p.name, "===")
    print("results_file:", parsed.results_path)
    print("sample_factors shape:", parsed.sample_factors.shape)
    print("data_table shape:", parsed.data_table.shape)
    if not parsed.data_table.empty:
        print("columns (first 8):", list(parsed.data_table.columns[:8]))