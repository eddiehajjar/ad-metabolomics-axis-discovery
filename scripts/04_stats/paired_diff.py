from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

META = Path("data/processed/metadata/sample_metadata.csv")
IN_DIR = Path("data/processed/lipidomics_qc")
OUT_DIR = Path("data/processed/differential")
OUT_DIR.mkdir(exist_ok=True)

def run(mode):
    X = pd.read_parquet(IN_DIR / f"{mode}.parquet")
    meta = pd.read_csv(META)

    before = meta[meta["group"] == "Before"]
    after  = meta[meta["group"] == "After"]

    before = before.sort_values("subject_id")
    after  = after.sort_values("subject_id")

    assert all(before["subject_id"].values == after["subject_id"].values)

    Xb = X[before["sample_id"]]
    Xa = X[after["sample_id"]]

    # Paired stats
    tstat, pvals = ttest_rel(Xa.values, Xb.values, axis=1, nan_policy="omit")
    log2fc = np.log2((Xa.mean(axis=1) + 1e-8) / (Xb.mean(axis=1) + 1e-8))

    padj = multipletests(pvals, method="fdr_bh")[1]

    res = pd.DataFrame({
        "feature": X.index,
        "log2FC": log2fc,
        "tstat": tstat,
        "pval": pvals,
        "padj": padj,
    }).sort_values("padj")

    out = OUT_DIR / f"{mode}_paired_diff.csv"
    res.to_csv(out, index=False)

    print(f"{mode}: wrote {out} ({res.shape[0]} features)")
    print("Top hits:")
    print(res.head(5))

def main():
    for mode in ["pos", "neg"]:
        run(mode)

if __name__ == "__main__":
    main()