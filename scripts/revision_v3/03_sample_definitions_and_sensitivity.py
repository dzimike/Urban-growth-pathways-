"""
Paper 2 v3 -- Script 03: Sample definitions and built-fraction threshold
sensitivity (Requirement 4)

Two DIFFERENT sample concepts are used throughout this revision, and are
kept explicitly distinct rather than conflated (a source of confusion the
v2 self-audit specifically flagged):

  National-footprint sample
      Cells with >=1 Google Open Buildings V3 footprint detected
      (building_count > 0 in grid_model_ready_500m.parquet). This is the
      sample every other script in the wider project (Paper 1, Paper 2 v2)
      calls "the 296,703 built-up cells". It says nothing about whether
      GHSL detected any height/volume there.

  Valid-height sample
      Cells where the GHSL-derived built_fraction (script 02 output,
      cell_built_fraction_mean) exceeds a stated threshold. This is a
      DIFFERENT population, defined purely from the GHSL side, used
      wherever a model needs AGBH/ANBH to be numerically reliable rather
      than just "some footprint exists". A cell can be in one sample and
      not the other: a newly-built single small structure can register in
      Open Buildings V3 (building_count > 0) while contributing a built
      fraction too small for GHSL's 100 m pixels to register confidently,
      and conversely a GHSL built-fraction signal can appear in a cell
      where Open Buildings V3's footprint detector missed the structure.

This script reports both samples, their overlap, and the sensitivity of
the valid-height sample's size and composition to five built-fraction
thresholds: >0 (encoded as a small epsilon), 0.005, 0.01, 0.02, 0.05
(config.BUILT_FRACTION_THRESHOLDS).

Outputs
-------
    02_processed_data/ghsl_layers_v3/sample_definitions.parquet
        Per-cell flags: in_national_footprint, in_valid_height_{thr} for
        each threshold, plus the underlying AGBH/ANBH/built_fraction values.
    04_outputs/paper2_v3/tables/T_sample_sensitivity.csv
        One row per threshold: n cells, % of national footprint sample,
        overlap with national-footprint sample, AGBH/ANBH descriptive
        stats within that valid-height sample.

Usage
-----
    python scripts/revision_v3/03_sample_definitions_and_sensitivity.py
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg

logging.basicConfig(level=cfg.LOG_LEVEL, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


def main() -> None:
    cfg.GHSL_LAYERS_V3.mkdir(parents=True, exist_ok=True)
    cfg.PAPER2_V3_TABLES.mkdir(parents=True, exist_ok=True)

    height = pd.read_parquet(cfg.GHSL_LAYERS_V3 / "agbh_anbh_grid_500m.parquet")
    footprint = pd.read_parquet(cfg.COVARIATES / "grid_model_ready_500m.parquet",
                                 columns=["grid_id", "building_count"])

    df = height.merge(footprint, on="grid_id", how="left")
    assert len(df) == 961_858

    df["in_national_footprint"] = df["building_count"].fillna(0) > 0
    n_footprint = int(df["in_national_footprint"].sum())
    log.info("National-footprint sample (Open Buildings V3, building_count>0): n=%d", n_footprint)

    n_footprint_nodata = int((df["in_national_footprint"] & df["cell_agbh_mean_m"].isna()).sum())
    log.info("  ...of which have NO overlapping GHSL raster pixel (true nodata, excluded from any "
             "height-based analysis): n=%d (%.4f%%)", n_footprint_nodata,
             100 * n_footprint_nodata / n_footprint)

    thresholds = cfg.BUILT_FRACTION_THRESHOLDS
    threshold_labels = [">0" if t < 1e-6 else str(t) for t in thresholds]
    rows = []
    for thr, label in zip(thresholds, threshold_labels):
        flag_col = f"in_valid_height_{label.replace('>', 'gt').replace('.', 'p')}"
        df[flag_col] = df["cell_built_fraction_mean"].fillna(-1) > thr
        sub = df[df[flag_col]]
        overlap_with_footprint = int((sub["in_national_footprint"]).sum())
        overlap_pct_of_footprint = 100 * overlap_with_footprint / n_footprint
        overlap_pct_of_valid = 100 * overlap_with_footprint / len(sub) if len(sub) else float("nan")
        rows.append({
            "threshold_label": label,
            "threshold_value": thr,
            "n_cells_valid_height": len(sub),
            "pct_of_national_961858": round(100 * len(sub) / 961_858, 4),
            "n_overlap_with_national_footprint": overlap_with_footprint,
            "pct_of_footprint_sample_covered": round(overlap_pct_of_footprint, 2),
            "pct_of_valid_height_that_is_footprint": round(overlap_pct_of_valid, 2),
            "agbh_mean": round(float(sub["cell_agbh_mean_m"].mean()), 4) if len(sub) else np.nan,
            "agbh_median": round(float(sub["cell_agbh_mean_m"].median()), 4) if len(sub) else np.nan,
            "anbh_derived_mean": round(float(sub["cell_anbh_derived_m"].mean()), 4) if len(sub) else np.nan,
            "anbh_derived_median": round(float(sub["cell_anbh_derived_m"].median()), 4) if len(sub) else np.nan,
            "anbh_derived_p99": round(float(sub["cell_anbh_derived_m"].quantile(0.99)), 4) if len(sub) else np.nan,
            "anbh_derived_max": round(float(sub["cell_anbh_derived_m"].max()), 4) if len(sub) else np.nan,
        })
        log.info("  threshold %-6s -> n=%7d (%.3f%% of national grid), covers %.2f%% of footprint sample, "
                  "ANBH mean=%.3f median=%.3f max=%.2f",
                  label, len(sub), 100 * len(sub) / 961_858, overlap_pct_of_footprint,
                  rows[-1]["anbh_derived_mean"] or float("nan"),
                  rows[-1]["anbh_derived_median"] or float("nan"),
                  rows[-1]["anbh_derived_max"] or float("nan"))

    sens = pd.DataFrame(rows)
    sens_path = cfg.PAPER2_V3_TABLES / "T_sample_sensitivity.csv"
    sens.to_csv(sens_path, index=False)
    log.info("Saved sensitivity table -> %s", sens_path.relative_to(cfg.PROJECT_ROOT))

    keep_cols = ["grid_id", "cell_agbh_mean_m", "cell_built_fraction_mean", "cell_anbh_derived_m",
                 "n_source_pixels", "in_national_footprint"] + \
                [c for c in df.columns if c.startswith("in_valid_height_")]
    out_path = cfg.GHSL_LAYERS_V3 / "sample_definitions.parquet"
    df[keep_cols].to_parquet(out_path, index=False)
    log.info("Saved per-cell sample flags -> %s", out_path.relative_to(cfg.PROJECT_ROOT))

    # ---------------------------------------------------------- QC print --
    print("\n=== Sample sensitivity summary ===")
    print(sens.to_string(index=False))
    print(f"\nNational-footprint sample: {n_footprint:,} cells "
          f"({n_footprint_nodata} lack any GHSL raster coverage -- true nodata)")


if __name__ == "__main__":
    main()
