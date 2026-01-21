from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

POS_DIFF = Path("data/processed/differential/pos_paired_diff.csv")
NEG_DIFF = Path("data/processed/differential/neg_paired_diff.csv")
OUT_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def volcano(path: Path, title: str, outname: str, label_top=10):
    df = pd.read_csv(path)
    # expected columns: feature, log2FC, tstat, pval, padj
    df["neglog10_padj"] = -np.log10(df["padj"].clip(lower=1e-300))

    plt.figure(figsize=(10, 7))
    plt.scatter(df["log2FC"], df["neglog10_padj"], alpha=0.6)

    # thresholds (editable)
    plt.axhline(-np.log10(0.05), linestyle="--")
    plt.axvline(0.5, linestyle="--")
    plt.axvline(-0.5, linestyle="--")

    # label top by padj
    top = df.sort_values("padj").head(label_top)
    for _, r in top.iterrows():
        plt.text(r["log2FC"], r["neglog10_padj"], str(r["feature"]), fontsize=8)

    plt.xlabel("log2 fold change (After / Before)")
    plt.ylabel("-log10(adjusted p-value)")
    plt.title(title)
    plt.tight_layout()

    outpath = OUT_DIR / outname
    plt.savefig(outpath, dpi=200)
    print("Saved:", outpath)

def main():
    volcano(POS_DIFF, "Volcano (POS): dupilumab paired Before→After", "pos_volcano.png")
    volcano(NEG_DIFF, "Volcano (NEG): dupilumab paired Before→After", "neg_volcano.png")

if __name__ == "__main__":
    main()