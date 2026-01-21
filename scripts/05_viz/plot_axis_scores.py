from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

POS_SCORES = Path("data/processed/axis/pos_scores.csv")
NEG_SCORES = Path("data/processed/axis/neg_scores.csv")

OUT_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot(scores_csv: Path, title: str, outname: str) -> None:
    df = pd.read_csv(scores_csv)

    # Keep only the groups we care about
    df["group"] = df["group"].astype(str)
    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce")

    before_df = df[df["group"] == "Before"].dropna(subset=["subject_id"]).copy()
    after_df = df[df["group"] == "After"].dropna(subset=["subject_id"]).copy()
    qc_df = df[df["group"] == "QC"].copy()

    # subject_id as int index for clean alignment
    before_df["subject_id"] = before_df["subject_id"].astype(int)
    after_df["subject_id"] = after_df["subject_id"].astype(int)

    before = before_df.set_index("subject_id")["axis_score"].sort_index()
    after = after_df.set_index("subject_id")["axis_score"].sort_index()

    # Only draw paired lines for subjects present in both
    common = before.index.intersection(after.index)

    plt.figure(figsize=(10, 7))

    # paired lines
    for sid in common:
        plt.plot([0, 1], [before.loc[sid], after.loc[sid]], alpha=0.35)

    # points
    plt.scatter([0] * len(before), before.values, label="Before")
    plt.scatter([1] * len(after), after.values, label="After")

    if len(qc_df) > 0:
        plt.scatter([-0.25] * len(qc_df), qc_df["axis_score"].values, label="QC")

    plt.xticks([0, 1], ["Before", "After"])
    plt.ylabel("Dupilumab lipidomic axis score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    outpath = OUT_DIR / outname
    plt.savefig(outpath, dpi=200)
    plt.close()
    print("Wrote:", outpath)


def main():
    plot(POS_SCORES, "Paired lipidomic response to dupilumab (POS)", "pos_axis_scores.png")
    plot(NEG_SCORES, "Paired lipidomic response to dupilumab (NEG)", "neg_axis_scores.png")


if __name__ == "__main__":
    main()