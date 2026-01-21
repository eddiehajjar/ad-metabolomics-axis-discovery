from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

# ====== EDIT THESE TWO PATHS TO MATCH YOUR treatment_axis.py ======
POS_IN = Path("data/processed/lipidomics/pos.parquet")  # <-- set to your qc-filtered POS parquet
NEG_IN = Path("data/processed/lipidomics/neg.parquet")  # <-- set to your qc-filtered NEG parquet
# ================================================================

META = Path("data/processed/metadata/sample_metadata.csv")
POS_SHIFT = Path("data/processed/axis/pos_subject_shift.csv")
NEG_SHIFT = Path("data/processed/axis/neg_subject_shift.csv")

OUT_DIR = Path("reports/modeling")
FIG_DIR = Path("reports/figures")
TABLE_DIR = Path("reports/tables")
for d in [OUT_DIR, FIG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_baseline_matrix(mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X_baseline_by_subject, meta_baseline).

    X_baseline_by_subject:
        rows = subject_id_num (int)
        cols = features (mz_rt)
    meta_baseline:
        baseline rows from sample_metadata.csv for those subjects
    """
    meta = pd.read_csv(META)

    # Parse subject ids; QC rows become NaN and get dropped
    meta["subject_id_num"] = pd.to_numeric(meta["subject_id"], errors="coerce")

    if mode == "pos":
        X = pd.read_parquet(POS_IN)
    elif mode == "neg":
        X = pd.read_parquet(NEG_IN)
    else:
        raise ValueError(mode)

    # Keep baseline (Before) samples only
    base = meta[meta["group"] == "Before"].copy()

    # Drop non-subject rows (e.g., QC) and enforce integer ids
    base = base.dropna(subset=["subject_id_num"]).copy()
    base["subject_id_num"] = base["subject_id_num"].astype(int)

    # Ensure X is samples x features (rows=samples, cols=features)
    # Common cases:
    # 1) X already has samples as rows -> index contains sample_ids
    # 2) X has samples as columns -> columns contain sample_ids, rows are mz_rt -> needs transpose
    sample_ids = base["sample_id"].astype(str).tolist()

    if all(sid in X.columns for sid in sample_ids) and not all(sid in X.index for sid in sample_ids):
        X = X.T
    elif not all(sid in X.index for sid in sample_ids):
        missing_in_index = [sid for sid in sample_ids if sid not in X.index]
        missing_in_cols = [sid for sid in sample_ids if sid not in X.columns]
        raise KeyError(
            f"Could not align samples for mode={mode}. "
            f"Missing in X.index (first 10): {missing_in_index[:10]} | "
            f"Missing in X.columns (first 10): {missing_in_cols[:10]}"
        )

    # Sort baseline subjects for stable ordering
    base = base.sort_values("subject_id_num").reset_index(drop=True)

    # Subset baseline matrix in the same order as base
    Xb = X.reindex(base["sample_id"].astype(str)).copy()

    # Sanity checks
    if Xb.isna().any(axis=None):
        bad = Xb.index[Xb.isna().any(axis=1)].tolist()
        raise ValueError(f"NaNs after reindexing baseline samples (first 10): {bad[:10]}")

    # Make sure 1 baseline sample per subject
    assert base["subject_id_num"].nunique() == len(base), "Baseline subjects not unique"

    # Set subject_id as index (int)
    Xb.index = base["subject_id_num"].values

    return Xb, base


def perm_test_corr(y_true, y_pred, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    r_obs, _ = pearsonr(y_true, y_pred)
    cnt = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y_true)
        r_perm, _ = pearsonr(y_perm, y_pred)
        if abs(r_perm) >= abs(r_obs):
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)
    return r_obs, p


def fit_and_eval(X, y, model_name: str):
    loo = LeaveOneOut()

    if model_name == "ridge":
        model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("reg", RidgeCV(alphas=np.logspace(-3, 6, 60)))
        ])
    elif model_name == "lasso":
        model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("reg", LassoCV(alphas=np.logspace(-4, 1, 60), cv=5, max_iter=200000))
        ])
    elif model_name == "elasticnet":
        model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("reg", ElasticNetCV(l1_ratio=[0.1,0.3,0.5,0.7,0.9],
                                 alphas=np.logspace(-4, 1, 60),
                                 cv=5,
                                 max_iter=200000))
        ])
    else:
        raise ValueError(model_name)

    yhat = cross_val_predict(model, X, y, cv=loo)

    r = np.corrcoef(y, yhat)[0, 1]
    r2 = r2_score(y, yhat)
    mae = mean_absolute_error(y, yhat)

    r_perm, p_perm = perm_test_corr(y, yhat, n_perm=10000, seed=0)

    return yhat, {"r": r, "r2": r2, "mae": mae, "perm_r": r_perm, "perm_p": p_perm}


def main():
    # shifts
    pos_shift = pd.read_csv(POS_SHIFT)
    neg_shift = pd.read_csv(NEG_SHIFT)
    pos_shift["subject_id"] = pos_shift["subject_id"].astype(int)
    neg_shift["subject_id"] = neg_shift["subject_id"].astype(int)

    # baseline matrices
    Xpos, _ = load_baseline_matrix("pos")
    Xneg, _ = load_baseline_matrix("neg")

    # align subjects (should be identical)
    subj = sorted(set(Xpos.index).intersection(set(Xneg.index)))
    Xpos = Xpos.loc[subj]
    Xneg = Xneg.loc[subj]

    y_pos = pos_shift.set_index("subject_id").loc[subj, "axis_shift"].values
    y_neg = neg_shift.set_index("subject_id").loc[subj, "axis_shift"].values

    # combined outcome: z-score each shift then mean
    y_comb = ( (y_pos - y_pos.mean())/y_pos.std(ddof=1) + (y_neg - y_neg.mean())/y_neg.std(ddof=1) ) / 2

    # feature sets
    X_comb = pd.concat([Xpos.add_prefix("pos__"), Xneg.add_prefix("neg__")], axis=1)

    rows = []
    preds_out = pd.DataFrame({"subject_id": subj, "y_pos": y_pos, "y_neg": y_neg, "y_comb": y_comb})

    for model_name in ["ridge", "elasticnet", "lasso"]:
        # POS
        yhat, m = fit_and_eval(Xpos.values, y_pos, model_name)
        rows.append({"target":"pos", "model":model_name, **m})
        preds_out[f"yhat_pos_{model_name}"] = yhat

        # NEG
        yhat, m = fit_and_eval(Xneg.values, y_neg, model_name)
        rows.append({"target":"neg", "model":model_name, **m})
        preds_out[f"yhat_neg_{model_name}"] = yhat

        # COMBINED
        yhat, m = fit_and_eval(X_comb.values, y_comb, model_name)
        rows.append({"target":"combined", "model":model_name, **m})
        preds_out[f"yhat_comb_{model_name}"] = yhat

    perf = pd.DataFrame(rows).sort_values(["target","perm_p","mae"])
    out_perf = TABLE_DIR / "response_prediction_performance.csv"
    out_preds = TABLE_DIR / "response_prediction_predictions.csv"
    perf.to_csv(out_perf, index=False)
    preds_out.to_csv(out_preds, index=False)

    print("Wrote:", out_perf)
    print(perf)
    print("\nWrote:", out_preds)
    print(preds_out.head())


if __name__ == "__main__":
    main()