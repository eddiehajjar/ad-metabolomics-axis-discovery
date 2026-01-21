from pathlib import Path
import pandas as pd

POS_DIFF = Path("data/processed/differential/pos_paired_diff.csv")
NEG_DIFF = Path("data/processed/differential/neg_paired_diff.csv")
OUT_DIR = Path("reports/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def write_top(diff_path: Path, outname: str, n=50):
    df = pd.read_csv(diff_path)
    df = df.sort_values("padj").head(n).copy()
    outpath = OUT_DIR / outname
    df.to_csv(outpath, index=False)
    print("Wrote:", outpath, "rows:", len(df))

def main():
    write_top(POS_DIFF, "pos_top50.csv", n=50)
    write_top(NEG_DIFF, "neg_top50.csv", n=50)

if __name__ == "__main__":
    main()
