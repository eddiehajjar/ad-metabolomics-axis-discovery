#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.

python scripts/01_ingest/build_processed_lipidomics.py
python scripts/01_ingest/build_sample_metadata.py
python scripts/01_ingest/add_subject_ids.py
python scripts/01_ingest/check_alignment.py

python scripts/02_qc/qc_filter.py

python scripts/03_axis/treatment_axis.py
python scripts/04_stats/paired_diff.py
python scripts/04_stats/axis_shift_stats.py
python scripts/04_stats/axis_vs_diff.py

python scripts/03_axis/pos_neg_concordance.py
python scripts/05_viz/pca_plot.py
python scripts/05_viz/volcano_plot.py
# python scripts/05_viz/pca_paired_arrows.py   # leave commented until it’s fixed