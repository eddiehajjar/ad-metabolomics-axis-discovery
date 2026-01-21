from pathlib import Path
import pandas as pd

DT_POS = Path("data/raw/lipidomics/datatables/ST002302_AN003761_datatable.txt")
OUT_DIR = Path("data/processed/metadata")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_class(x: str) -> str:
    x = str(x).strip()
    xl = x.lower()

    # treat any "control" as QC
    if "control" in xl or xl.startswith("qc") or "quality control" in xl:
        return "QC"
    if "before" in xl:
        return "Before"
    if "after" in xl:
        return "After"
    return "Unknown"

def read_datatable_split_once(path: Path) -> pd.DataFrame:
    """
    Datatable format:
      Samples Class
      B1  Treatment:After dupilumab treatment
    The second column contains spaces, so we must split only once.
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        _ = next(f, None)  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)  # split on first whitespace only
            if len(parts) != 2:
                continue
            sample_id, class_raw = parts[0].strip(), parts[1].strip()
            rows.append((sample_id, class_raw))
    return pd.DataFrame(rows, columns=["sample_id", "class_raw"])

def main():
    df = read_datatable_split_once(DT_POS)
    df["group"] = df["class_raw"].map(normalize_class)

    # sanity checks
    assert df["sample_id"].nunique() == len(df), "Duplicate sample IDs in datatable"
    assert set(df["group"].unique()) <= {"QC", "Before", "After", "Unknown"}, df["group"].unique()

    out = OUT_DIR / "sample_groups.csv"
    df[["sample_id", "group"]].to_csv(out, index=False)
    print("Wrote:", out, df.shape)
    print(df["group"].value_counts())

if __name__ == "__main__":
    main()