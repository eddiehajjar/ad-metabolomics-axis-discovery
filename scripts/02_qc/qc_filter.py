from pathlib import Path
import pandas as pd
import numpy as np

META = Path("data/processed/metadata/sample_metadata.csv")
IN_DIR = Path("data/processed/lipidomics")
OUT_DIR = Path("data/processed/lipidomics_qc")
OUT_DIR.mkdir(exist_ok=True)

QC_CV_THRESHOLD = 0.30

def qc_filter(mat_path):
    X = pd.read_parquet(mat_path)
    meta = pd.read_csv(META)

    qc_samples = meta.loc[meta["group"] == "QC", "sample_id"]
    X_qc = X[qc_samples]

    mean = X_qc.mean(axis=1)
    std = X_qc.std(axis=1)
    cv = std / mean

    keep = cv <= QC_CV_THRESHOLD
    X_filt = X.loc[keep]

    return X_filt, cv

def main():
    for mode in ["pos", "neg"]:
        Xf, cv = qc_filter(IN_DIR / f"{mode}.parquet")
        out = OUT_DIR / f"{mode}.parquet"
        Xf.to_parquet(out)

        print(f"{mode}: kept {Xf.shape[0]} features")

if __name__ == "__main__":
    main()