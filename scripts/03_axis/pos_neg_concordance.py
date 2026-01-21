from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

POS = Path("data/processed/axis/pos_subject_shift.csv")
NEG = Path("data/processed/axis/neg_subject_shift.csv")

OUT = Path("reports/figures/pos_neg_concordance.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    pos = pd.read_csv(POS)
    neg = pd.read_csv(NEG)

    df = pos.merge(neg, on="subject_id", suffixes=("_pos", "_neg"))

    # Correlations
    r_p, p_p = pearsonr(df["axis_shift_pos"], df["axis_shift_neg"])
    r_s, p_s = spearmanr(df["axis_shift_pos"], df["axis_shift_neg"])

    print(f"Pearson r = {r_p:.3f}, p = {p_p:.2e}")
    print(f"Spearman ρ = {r_s:.3f}, p = {p_s:.2e}")

    from pathlib import Path

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(df["axis_shift_pos"], df["axis_shift_neg"], s=60)

    lims = [
        min(df["axis_shift_pos"].min(), df["axis_shift_neg"].min()),
        max(df["axis_shift_pos"].max(), df["axis_shift_neg"].max()),
    ]
    plt.plot(lims, lims, "--", color="gray")

    plt.xlabel("POS treatment axis shift")
    plt.ylabel("NEG treatment axis shift")
    plt.title(
        f"POS–NEG concordance\n"
        f"Pearson r={r_p:.2f} (p={p_p:.1e}) | Spearman ρ={r_s:.2f} (p={p_s:.1e})"
    )

    # Save once
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pos_neg_concordance.png"

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()