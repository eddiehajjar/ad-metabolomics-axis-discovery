from pathlib import Path
import glob
import numpy as np
import pandas as pd

def pick_one(patterns):
    hits = []
    for pat in patterns:
        hits += glob.glob(pat)
    hits = sorted(set(hits))
    if len(hits) == 0:
        raise FileNotFoundError(f"No files found for patterns: {patterns}")
    if len(hits) > 1:
        print("Multiple matches, using first:")
        for h in hits[:10]:
            print(" ", h)
    return Path(hits[0])

POS_IN = pick_one([
    "data/processed/lipidomics/*pos*qc*.parquet",
    "data/processed/lipidomics/*pos*filter*.parquet",
    "data/processed/lipidomics/pos.parquet",
])

NEG_IN = pick_one([
    "data/processed/lipidomics/*neg*qc*.parquet",
    "data/processed/lipidomics/*neg*filter*.parquet",
    "data/processed/lipidomics/neg.parquet",
])

META = Path("data/processed/metadata/sample_metadata.csv")

OUT_DIR = Path("data/processed/axis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def zscore_by_feature(X: pd.DataFrame) -> pd.DataFrame:
    """
    X: features x samples
    Standardize each feature across samples (mean 0, std 1).
    """
    mu = X.mean(axis=1)
    sd = X.std(axis=1, ddof=0).replace(0, np.nan)
    Z = (X.sub(mu, axis=0)).div(sd, axis=0)
    return Z.fillna(0.0)


def compute_axis_and_scores(X: pd.DataFrame, meta: pd.DataFrame):
    """
    X: features x samples
    meta must include columns: sample_id, group (Before/After/QC), subject_id
    Returns:
      axis (pd.Series over features),
      scores (pd.Series over samples),
      deltas (pd.DataFrame subject_id x features) [After-Before in Z-space]
    """
    # keep only Before/After, ensure aligned samples
    meta_ba = meta[meta["group"].isin(["Before", "After"])].copy()

    # X columns are sample IDs (A1.., B1.., QC..)
    common = [s for s in meta_ba["sample_id"].tolist() if s in X.columns]
    meta_ba = meta_ba[meta_ba["sample_id"].isin(common)].copy()
    X = X[common]

    # standardize features so high-variance features don't dominate
    Z = zscore_by_feature(X)

    # build paired matrices indexed by subject_id
    before = meta_ba[meta_ba["group"] == "Before"][["sample_id", "subject_id"]].copy()
    after  = meta_ba[meta_ba["group"] == "After"][["sample_id", "subject_id"]].copy()

    # ensure 1 before and 1 after per subject
    before = before.drop_duplicates("subject_id")
    after  = after.drop_duplicates("subject_id")

    subjects = sorted(set(before["subject_id"]).intersection(set(after["subject_id"])))

    # matrices: features x subjects
    Zb = pd.DataFrame({sid: Z[ before.loc[before["subject_id"] == sid, "sample_id"].iloc[0] ] for sid in subjects})
    Za = pd.DataFrame({sid: Z[ after.loc[after["subject_id"] == sid, "sample_id"].iloc[0] ] for sid in subjects})

    # deltas: features x subjects
    dZ = Za - Zb  # After - Before
    axis = dZ.mean(axis=1)  # mean shift per feature

    # normalize axis to unit length (so scores are comparable)
    norm = float(np.sqrt((axis.values ** 2).sum()))
    if norm == 0:
        raise ValueError("Axis norm is 0 — no consistent paired shift found.")
    axis_unit = axis / norm

    # scores for each sample: projection onto axis
    # score(sample) = axis_unit · Z[:, sample]
    scores = (Z.T @ axis_unit).rename("axis_score")  # samples x 1

    # also subject-level shift score: axis dot (After-Before)
    subject_shift = (dZ.T @ axis_unit).rename("axis_shift")  # subjects x 1

    return axis_unit, scores, dZ.T, subject_shift


def main():
    meta = pd.read_csv(META)
    assert {"sample_id", "group", "subject_id"} <= set(meta.columns), meta.columns.tolist()

    outputs = []

    for name, path in [("pos", POS_IN), ("neg", NEG_IN)]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. If qc_filter wrote different names, update POS_IN/NEG_IN.")

        X = pd.read_parquet(path)  # expect features x samples
        axis, sample_scores, deltas, subject_shift = compute_axis_and_scores(X, meta)

        # save axis weights (feature importance)
        axis_out = OUT_DIR / f"{name}_axis_weights.csv"
        axis.to_frame("weight").to_csv(axis_out)

        # save sample scores w/ metadata
        scores_df = meta.merge(sample_scores.reset_index().rename(columns={"index": "sample_id"}),
                               on="sample_id", how="left")
        scores_out = OUT_DIR / f"{name}_scores.csv"
        scores_df.to_csv(scores_out, index=False)

        # save subject-level shift
        subj_df = subject_shift.reset_index().rename(columns={"index": "subject_id"})
        subj_out = OUT_DIR / f"{name}_subject_shift.csv"
        subj_df.to_csv(subj_out, index=False)

        print(f"[{name}] wrote:")
        print(" ", axis_out)
        print(" ", scores_out, f"(rows={len(scores_df)})")
        print(" ", subj_out, f"(subjects={len(subj_df)})")

        outputs.append((name, scores_out))

    print("\nDone.")


if __name__ == "__main__":
    main()