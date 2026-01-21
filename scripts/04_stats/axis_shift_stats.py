from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

POS_SHIFT = Path("data/processed/axis/pos_subject_shift.csv")
NEG_SHIFT = Path("data/processed/axis/neg_subject_shift.csv")
OUT_DIR = Path("data/processed/axis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def summarize(shifts: pd.Series) -> dict:
    x = shifts.dropna().astype(float).values
    n = len(x)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n-1) if n > 1 else np.nan
    ci_low = mean - tcrit * se if n > 1 else np.nan
    ci_high = mean + tcrit * se if n > 1 else np.nan

    # Paired shift test: H0 mean shift = 0
    tstat, pval = stats.ttest_1samp(x, popmean=0.0)

    # Also provide nonparametric sign test-ish via Wilcoxon (robust)
    try:
        wstat, wpval = stats.wilcoxon(x)
    except ValueError:
        wstat, wpval = np.nan, np.nan

    return {
        "n_subjects": n,
        "mean_shift": mean,
        "sd_shift": sd,
        "se_shift": se,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "ttest_t": float(tstat),
        "ttest_p": float(pval),
        "wilcoxon_stat": float(wstat) if np.isfinite(wstat) else np.nan,
        "wilcoxon_p": float(wpval) if np.isfinite(wpval) else np.nan,
        "frac_positive_shift": float(np.mean(x > 0)),
        "median_shift": float(np.median(x)),
    }

def main():
    rows = []
    for name, path in [("pos", POS_SHIFT), ("neg", NEG_SHIFT)]:
        df = pd.read_csv(path)
        stats_row = summarize(df["axis_shift"])
        stats_row["mode"] = name
        rows.append(stats_row)

    out = pd.DataFrame(rows).set_index("mode")
    out_path = OUT_DIR / "axis_shift_stats.csv"
    out.to_csv(out_path)
    print("Wrote:", out_path)
    print(out)

if __name__ == "__main__":
    main()