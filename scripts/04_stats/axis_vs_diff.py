from pathlib import Path
import numpy as np
import pandas as pd

POS_W = Path("data/processed/axis/pos_axis_weights.csv")
NEG_W = Path("data/processed/axis/neg_axis_weights.csv")

POS_D = Path("data/processed/differential/pos_paired_diff.csv")
NEG_D = Path("data/processed/differential/neg_paired_diff.csv")

OUT_DIR = Path("reports/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def merge(weights_path: Path, diff_path: Path, outname: str):
    w = pd.read_csv(weights_path)  # mz_rt, weight
    d = pd.read_csv(diff_path)     # feature, log2FC, ...
    d = d.rename(columns={"feature": "mz_rt"})
    m = w.merge(d, on="mz_rt", how="inner")

    # helpful derived fields
    m["abs_weight"] = m["weight"].abs()
    m["neglog10_padj"] = -np.log10(m["padj"].clip(lower=1e-300))

    # sort by absolute weight (axis drivers)
    m_sorted = m.sort_values("abs_weight", ascending=False)
    outpath = OUT_DIR / outname
    m_sorted.to_csv(outpath, index=False)
    print("Wrote:", outpath, "rows:", len(m_sorted))

    print("\nTop 10 axis drivers:")
    print(m_sorted[["mz_rt", "weight", "log2FC", "padj"]].head(10).to_string(index=False))

def main():
    merge(POS_W, POS_D, "pos_axis_vs_diff.csv")
    merge(NEG_W, NEG_D, "neg_axis_vs_diff.csv")

if __name__ == "__main__":
    main()