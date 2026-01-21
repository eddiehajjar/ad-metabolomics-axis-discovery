from pathlib import Path
import pandas as pd
import re

META = Path("data/processed/metadata/sample_groups.csv")
OUT = Path("data/processed/metadata/sample_metadata.csv")

def subject_from_sample(s):
    # A1 ↔ B1 → subject 1
    m = re.match(r"[AB](\d+)", s)
    return m.group(1) if m else s

def main():
    df = pd.read_csv(META)
    df["subject_id"] = df["sample_id"].map(subject_from_sample)

    # QC samples remain unique
    df.loc[df["group"] == "QC", "subject_id"] = df["sample_id"]

    assert df["sample_id"].nunique() == len(df)
    assert df[df["group"] != "QC"]["subject_id"].nunique() == 33

    df.to_csv(OUT, index=False)
    print("Wrote:", OUT)
    print(df.head())

if __name__ == "__main__":
    main()