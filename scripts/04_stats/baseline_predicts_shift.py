from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AXIS_DIR = Path("data/processed/axis")
FIG_DIR = Path("reports/figures")
TAB_DIR = Path("reports/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

N_PERM = 10_000
RNG_SEED = 0


def zscore(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    return (x - x.mean()) / (x.std(ddof=0) + 1e-12)


def corr_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x**2).sum()) * np.sqrt((y**2).sum())
    return float((x * y).sum() / (denom + 1e-12))


def corr_spearman(x, y):
    # Spearman = Pearson on ranks
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return corr_pearson(rx, ry)


def permutation_pvalue(x, y, stat_fn, n_perm=N_PERM, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    obs = stat_fn(x, y)
    cnt = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        sp = stat_fn(x, yp)
        if abs(sp) >= abs(obs):
            cnt += 1
    # +1 smoothing
    p = (cnt + 1) / (n_perm + 1)
    return obs, p


def load_mode(mode: str) -> pd.DataFrame:
    scores = pd.read_csv(AXIS_DIR / f"{mode}_scores.csv")
    shift = pd.read_csv(AXIS_DIR / f"{mode}_subject_shift.csv")

    # force consistent dtypes
    scores["subject_id"] = pd.to_numeric(scores["subject_id"], errors="coerce").astype("Int64")
    shift["subject_id"] = pd.to_numeric(shift["subject_id"], errors="coerce").astype("Int64")

    # baseline axis score = BEFORE samples only
    base = (
        scores[scores["group"].str.lower().eq("before")]
        .loc[:, ["subject_id", "axis_score"]]
        .rename(columns={"axis_score": f"{mode}_baseline"})
    )

    df = base.merge(shift, on="subject_id", how="inner").rename(
        columns={"axis_shift": f"{mode}_shift"}
    )

    # drop any weird NA subject_ids just in case
    df = df.dropna(subset=["subject_id"]).copy()
    df["subject_id"] = df["subject_id"].astype(int)

    return df


def scatter_plot(df, xcol, ycol, title, outpath):
    x = df[xcol].astype(float).to_numpy()
    y = df[ycol].astype(float).to_numpy()

    r = corr_pearson(x, y)
    rho = corr_spearman(x, y)

    plt.figure()
    plt.scatter(x, y)
    # best-fit line (least squares)
    m, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    plt.plot(xx, m * xx + b)

    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(f"{title}\nPearson r={r:.2f} | Spearman ρ={rho:.2f}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def tertile_plot(df, xcol, ycol, title, outpath):
    tmp = df.copy()
    tmp["baseline_tertile"] = pd.qcut(tmp[xcol].astype(float), 3, labels=["Low", "Mid", "High"])
    grp = tmp.groupby("baseline_tertile")[ycol].agg(["mean", "std", "count"])
    grp["se"] = grp["std"] / np.sqrt(grp["count"].clip(lower=1))

    order = ["Low", "Mid", "High"]
    means = grp.loc[order, "mean"].to_numpy()
    ses = grp.loc[order, "se"].to_numpy()

    plt.figure()
    x = np.arange(3)
    plt.bar(x, means, yerr=ses, capsize=4)
    plt.xticks(x, order)
    plt.ylabel(ycol)
    plt.title(title)

    # overlay points (jitter) for transparency
    for i, lab in enumerate(order):
        ys = tmp.loc[tmp["baseline_tertile"] == lab, ycol].astype(float).to_numpy()
        xs = np.full_like(ys, i, dtype=float) + np.random.default_rng(0).normal(0, 0.04, size=len(ys))
        plt.scatter(xs, ys, alpha=0.7)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    return grp.reset_index()


def main():
    # --- POS + NEG ---
    pos = load_mode("pos")
    neg = load_mode("neg")

    # rename to common cols for merging
    pos = pos.rename(columns={"pos_baseline": "baseline", "pos_shift": "shift"}).assign(mode="pos")
    neg = neg.rename(columns={"neg_baseline": "baseline", "neg_shift": "shift"}).assign(mode="neg")

    # save per-mode merged tables
    pos.to_csv(TAB_DIR / "pos_baseline_vs_shift.csv", index=False)
    neg.to_csv(TAB_DIR / "neg_baseline_vs_shift.csv", index=False)

    # correlations + permutation tests (per mode)
    rows = []
    for mode, dfm in [("pos", pos), ("neg", neg)]:
        x = dfm["baseline"].to_numpy()
        y = dfm["shift"].to_numpy()

        pearson_r, pearson_pperm = permutation_pvalue(x, y, corr_pearson)
        spearman_rho, spearman_pperm = permutation_pvalue(x, y, corr_spearman)

        rows.append(
            {
                "mode": mode,
                "n_subjects": len(dfm),
                "pearson_r": pearson_r,
                "pearson_perm_p": pearson_pperm,
                "spearman_rho": spearman_rho,
                "spearman_perm_p": spearman_pperm,
            }
        )

        scatter_plot(
            dfm, "baseline", "shift",
            title=f"{mode.upper()}: baseline axis score vs response magnitude",
            outpath=FIG_DIR / f"{mode}_baseline_vs_shift.png",
        )

        tert = tertile_plot(
            dfm, "baseline", "shift",
            title=f"{mode.upper()}: response magnitude by baseline tertile",
            outpath=FIG_DIR / f"{mode}_tertiles_shift.png",
        )
        tert.to_csv(TAB_DIR / f"{mode}_tertiles_shift.csv", index=False)

    # --- COMBINED ---
    # Merge POS + NEG on subject_id (keep only subjects present in both)
    pos2 = load_mode("pos").rename(columns={"pos_baseline": "pos_baseline", "pos_shift": "pos_shift"})
    neg2 = load_mode("neg").rename(columns={"neg_baseline": "neg_baseline", "neg_shift": "neg_shift"})
    comb = pos2.merge(neg2, on="subject_id", how="inner")

    # Combined baseline/shift = sum of z-scores (scale-free)
    comb["baseline_combined"] = zscore(comb["pos_baseline"]) + zscore(comb["neg_baseline"])
    comb["shift_combined"] = zscore(comb["pos_shift"]) + zscore(comb["neg_shift"])

    comb_out = comb.loc[:, ["subject_id", "pos_baseline", "neg_baseline", "baseline_combined",
                            "pos_shift", "neg_shift", "shift_combined"]]
    comb_out.to_csv(TAB_DIR / "combined_baseline_vs_shift.csv", index=False)

    x = comb["baseline_combined"].to_numpy()
    y = comb["shift_combined"].to_numpy()
    pearson_r, pearson_pperm = permutation_pvalue(x, y, corr_pearson)
    spearman_rho, spearman_pperm = permutation_pvalue(x, y, corr_spearman)

    rows.append(
        {
            "mode": "combined",
            "n_subjects": len(comb),
            "pearson_r": pearson_r,
            "pearson_perm_p": pearson_pperm,
            "spearman_rho": spearman_rho,
            "spearman_perm_p": spearman_pperm,
        }
    )

    scatter_plot(
        comb, "baseline_combined", "shift_combined",
        title="COMBINED (POS+NEG): baseline vs response magnitude",
        outpath=FIG_DIR / "combined_baseline_vs_shift.png",
    )

    tert = tertile_plot(
        comb, "baseline_combined", "shift_combined",
        title="COMBINED (POS+NEG): response magnitude by baseline tertile",
        outpath=FIG_DIR / "combined_tertiles_shift.png",
    )
    tert.to_csv(TAB_DIR / "combined_tertiles_shift.csv", index=False)

    # Save summary stats
    summary = pd.DataFrame(rows)
    summary.to_csv(TAB_DIR / "baseline_predicts_shift_stats.csv", index=False)
    print("Wrote:")
    print(" ", TAB_DIR / "baseline_predicts_shift_stats.csv")
    print(" ", FIG_DIR / "pos_baseline_vs_shift.png")
    print(" ", FIG_DIR / "neg_baseline_vs_shift.png")
    print(" ", FIG_DIR / "combined_baseline_vs_shift.png")
    print(" ", FIG_DIR / "pos_tertiles_shift.png")
    print(" ", FIG_DIR / "neg_tertiles_shift.png")
    print(" ", FIG_DIR / "combined_tertiles_shift.png")
    print("\nStats:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()