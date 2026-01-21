import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

META = Path("data/processed/metadata/sample_metadata.csv")
XPATH = Path("data/processed/lipidomics_qc/pos.parquet")

def main():
    X = pd.read_parquet(XPATH)
    meta = pd.read_csv(META).set_index("sample_id")

    X = X[meta.index].T
    Xs = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    Z = pca.fit_transform(Xs)

    df = pd.DataFrame(Z, columns=["PC1", "PC2"], index=X.index)
    df["group"] = meta["group"]

    plt.figure()
    for g in ["Before", "After", "QC"]:
        sub = df[df["group"] == g]
        plt.scatter(sub["PC1"], sub["PC2"], label=g)

    plt.legend()
    plt.title("POS lipidomics PCA")
    plt.show()

if __name__ == "__main__":
    main()