from pathlib import Path
import pandas as pd

POS = Path("data/processed/lipidomics/pos.parquet")
NEG = Path("data/processed/lipidomics/neg.parquet")
META = Path("data/processed/metadata/sample_groups.csv")

def check(mat_path):
    X = pd.read_parquet(mat_path)
    return set(X.columns)

def main():
    meta = pd.read_csv(META)
    meta_ids = set(meta["sample_id"])

    pos_ids = check(POS)
    neg_ids = check(NEG)

    print("Metadata samples:", len(meta_ids))
    print("POS samples:", len(pos_ids))
    print("NEG samples:", len(neg_ids))

    print("\nMissing in POS:", sorted(meta_ids - pos_ids))
    print("Extra in POS:", sorted(pos_ids - meta_ids))

    print("\nMissing in NEG:", sorted(meta_ids - neg_ids))
    print("Extra in NEG:", sorted(neg_ids - meta_ids))

    assert meta_ids == pos_ids == neg_ids, "Sample mismatch detected"
    print("\n✅ Sample alignment OK")

if __name__ == "__main__":
    main()