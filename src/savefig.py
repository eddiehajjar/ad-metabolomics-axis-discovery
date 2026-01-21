from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def savefig(name: str, dpi: int = 200) -> Path:
    """
    Save the current matplotlib figure to reports/figures/<name>.
    Automatically appends .png if missing.
    """
    name = name.strip()
    if not name.lower().endswith((".png", ".pdf", ".svg")):
        name = f"{name}.png"
    out = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    return out