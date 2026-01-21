from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd


@dataclass
class MWTabParsed:
    header: Dict[str, str]
    sample_factors: pd.DataFrame
    results_path: Optional[str]
    data_table: pd.DataFrame


def _read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in f]


def _parse_key_value_lines(lines: List[str]) -> Dict[str, str]:
    """
    Parses top-of-file key/value lines like:
    VERSION\t1
    CREATED_ON\t...
    and sectioned keys like PR:..., ST:..., etc.
    """
    header: Dict[str, str] = {}
    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        if "\t" in ln:
            k, v = ln.split("\t", 1)
            k = k.strip()
            v = v.strip()
            # Keep duplicates by suffixing
            if k in header:
                i = 2
                while f"{k}__{i}" in header:
                    i += 1
                header[f"{k}__{i}"] = v
            else:
                header[k] = v
    return header


def _find_results_file(lines: List[str]) -> Optional[str]:
    for ln in lines:
        if ln.startswith("MS:MS_RESULTS_FILE"):
            # Format: MS:MS_RESULTS_FILE <tab> filename <tab> UNITS:... <tab> ...
            parts = ln.split("\t")
            if len(parts) >= 2:
                return parts[1].strip()
    return None


def _parse_subject_sample_factors(lines: List[str]) -> pd.DataFrame:
    rows = []
    for ln in lines:
        if ln.startswith("SUBJECT_SAMPLE_FACTORS"):
            parts = ln.split("\t")
            # Expected: SUBJECT_SAMPLE_FACTORS, SUBJECT, SAMPLE, FACTORS, extras...
            # Sometimes SUBJECT is '-' (as in your file).
            while len(parts) < 4:
                parts.append("")
            _, subject, sample, factors = parts[:4]
            extra = parts[4:] if len(parts) > 4 else []
            rows.append(
                {
                    "subject_id": subject.strip(),
                    "sample_id": sample.strip(),
                    "factors_raw": factors.strip(),
                    "extras_raw": " | ".join([x.strip() for x in extra if x.strip()]),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Parse factors like: "Treatment:Before dupilumab treatment|Other:..."
    def parse_factor_string(s: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if not s:
            return out
        for token in s.split("|"):
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                k, v = token.split(":", 1)
                out[k.strip()] = v.strip()
        return out

    factor_dicts = df["factors_raw"].apply(parse_factor_string)
    factor_df = pd.json_normalize(factor_dicts)
    df = pd.concat([df.drop(columns=["factors_raw"]), factor_df], axis=1)
    return df


def _parse_results_table(lines: List[str]) -> pd.DataFrame:
    """
    mwTab files typically contain a big tab-delimited results table near the end,
    often preceded by something like '#MS_METABOLITE_DATA' or similar.
    We will detect the first line that looks like a wide header row (contains 'm/z' or 'RT').
    """
    # Heuristic: find header row that contains "m/z" or "mz" and "RT" or "rt"
    start_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("#"):
            continue
        low = ln.lower()
        if "\t" in ln and (("m/z" in low) or (" mz" in low) or ("\tmz" in low)) and ("rt" in low):
            start_idx = i
            break

    # Fallback: find the first long tab-delimited row after MS_RESULTS_FILE
    if start_idx is None:
        for i, ln in enumerate(lines):
            if "\t" in ln and len(ln.split("\t")) >= 10:
                # avoid top header area by requiring it appears after a #END? No.
                # We'll just take the first very wide row past the metadata region.
                if i > 50:
                    start_idx = i
                    break

    if start_idx is None:
        return pd.DataFrame()

    header = lines[start_idx].split("\t")
    data_rows = []
    for ln in lines[start_idx + 1 :]:
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split("\t")
        # skip malformed rows
        if len(parts) != len(header):
            continue
        data_rows.append(parts)

    if not data_rows:
        return pd.DataFrame(columns=header)

    df = pd.DataFrame(data_rows, columns=header)

    # Try to coerce numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


def load_mwtab(path: str | Path) -> MWTabParsed:
    path = Path(path)
    lines = _read_lines(path)

    header = _parse_key_value_lines(lines[:400])  # metadata lives up top
    results_path = _find_results_file(lines)
    sample_factors = _parse_subject_sample_factors(lines)
    data_table = _parse_results_table(lines)

    return MWTabParsed(
        header=header,
        sample_factors=sample_factors,
        results_path=results_path,
        data_table=data_table,
    )