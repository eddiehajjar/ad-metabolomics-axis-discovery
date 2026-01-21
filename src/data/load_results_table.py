from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple, List

import pandas as pd


# Matches floats like 0.123, .123, 12, 1.2E-05, 7.23E-05, etc.
_FLOAT_RE = re.compile(r"""
    [+-]?
    (?:
        (?:\d+\.\d*)|   # 12. or 12.34
        (?:\.\d+)|      # .34
        (?:\d+)         # 12
    )
    (?:[eE][+-]?\d+)?   # optional exponent
""", re.VERBOSE)

# Matches mz_rt id like "452.3748819_157.972"
_MZRT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?_\d+(?:\.\d+)?)\s+")


def expected_sample_ids() -> List[str]:
    qcs = [f"QC{i:02d}" for i in range(1, 8)]
    a = [f"A{i}" for i in range(1, 34)]
    b = [f"B{i}" for i in range(1, 34)]
    return qcs + a + b


def load_results_table(path: str | Path) -> pd.DataFrame:
    """
    Robustly parse MWB Results.txt feature table even if delimiters are broken.
    Returns a DataFrame with columns: ['mz_rt'] + sample_ids (73 sample columns)
    """
    path = Path(path)
    sample_ids = expected_sample_ids()
    n_expected = len(sample_ids)

    rows = []
    bad_lines = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline()  # discard; it's unreliable

        for line_no, line in enumerate(f, start=2):
            m = _MZRT_RE.match(line)
            if not m:
                bad_lines += 1
                continue

            mzrt = m.group(1)
            rest = line[m.end():]

            nums = [float(x) for x in _FLOAT_RE.findall(rest)]

            # Guardrail: keep only lines with the exact expected number of values
            if len(nums) != n_expected:
                bad_lines += 1
                continue

            rows.append([mzrt] + nums)

    df = pd.DataFrame(rows, columns=["mz_rt"] + sample_ids)

    if df.empty:
        raise ValueError(
            f"Parsed 0 rows from {path}. Bad lines: {bad_lines}. "
            f"File may not match expected format."
        )

    # Helpful sanity check
    if bad_lines > 0:
        print(f"[load_results_table] Warning: skipped {bad_lines} malformed lines from {path.name}")

    return df