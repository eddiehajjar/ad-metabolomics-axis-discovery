from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Paths
X_PATH = Path("data/processed/lipidomics/pos.parquet")
META_PATH = Path("data/processed/metadata/sample_metadata.csv")
OUT = Path("reports/figures/pca_pos_paired.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    # Load data
    X = pd.read_parquet(X_PATH)         
    X = X.T                             
    meta = pd.read_csv(META_PATH)

    # Align (and sanity-check)
    meta = meta.sort_values("sample_id")
    X = X.loc[meta["sample_id"].values]

    # Scale (critical)
    Xs = StandardScaler().fit_transform(X.values)

    # PCA
    pca = PCA(n_components=2)
    Z = pca.fit_transform(Xs)

    pca_df = pd.DataFrame(
        Z,
        columns=["PC1", "PC2"],
    )
    pca_df = pd.concat([meta.reset_index(drop=True), pca_df], axis=1)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = {"Before": "tab:blue", "After": "tab:red", "QC": "gray"}

    # Scatter
    for g, d in pca_df.groupby("group"):
        ax.scatter(
            d.PC1, d.PC2,
            label=g,
            alpha=0.7,
            s=40,
            color=colors[g]
        )

    # Paired arrows
    for sid, d in pca_df.groupby("subject_id"):
        if {"Before", "After"} <= set(d.group):
            b = d[d.group == "Before"].iloc[0]
            a = d[d.group == "After"].iloc[0]
            ax.arrow(
                b.PC1, b.PC2,
                a.PC1 - b.PC1,
                a.PC2 - b.PC2,
                color="black",
                alpha=0.4,
                width=0.002,
                length_includes_head=True
            )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend()
    ax.set_title("POS lipidomics PCA (paired Before→After)")

    plt.tight_layout()
    plt.savefig(OUT, dpi=300)
    plt.close()

    print("Wrote:", OUT)

if __name__ == "__main__":
    main()