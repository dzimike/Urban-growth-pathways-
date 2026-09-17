"""
Paper 2 v3 -- Script 06: Vertical-horizontal typology rebuild (Requirement 9)

Rebuilds the typology on:
  - Vertical axis: cell_anbh_derived_m (script 02) -- the PRIMARY vertical
    outcome per Requirement 5. NOT AGBH (that was v2's basis).
  - Horizontal axis: growth_1975_2015 = built_coverage_2015 -
    built_coverage_1975 -- the full prior-expansion trajectory truncated at
    the last pre-2018 GHSL epoch. This replaces v2's "long_term_growth"
    (1975-2020), which extended two years past the 2018 vertical outcome;
    2015 is the correct truncation point per Requirement 6's logic applied
    consistently to the typology, not just the regression models.

Classification rule (unchanged in spirit from the v2 typology, restated
and re-verified here rather than assumed to still be safe): a two-by-two
matrix on whether each cell's ANBH and growth_1975_2015 exceed the
NATIONAL MEDIAN. Every threshold value and every tie count is reported
exactly. Ties (cell value == median) are assigned to the LOW class on
that axis -- the same disclosed rule used throughout this revision.

Labels are literal and descriptive per Requirement 9 ("high/low height x
high/low prior expansion"), not evocative names like v2's "vertical-
horizontal intensification": High height / High expansion, High height /
Low expansion, Low height / High expansion, Low height / Low expansion.

Outputs
-------
    02_processed_data/ghsl_layers_v3/typology_v3.parquet
        Per-cell: grid_id, cell_anbh_derived_m, growth_1975_2015, class label
    04_outputs/paper2_v3/tables/T_typology_median.csv
    04_outputs/paper2_v3/tables/T_typology_tertile_full.csv
    04_outputs/paper2_v3/tables/T_typology_quartile_full.csv
    04_outputs/paper2_v3/tables/T_typology_sensitivity_summary.csv

Usage
-----
    python scripts/revision_v3/06_typology_v3.py
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

LABELS = {
    (True, True): "High height / High prior expansion",
    (True, False): "High height / Low prior expansion",
    (False, True): "Low height / High prior expansion",
    (False, False): "Low height / Low prior expansion",
}


def build_dataset() -> pd.DataFrame:
    samp = pd.read_parquet(cfg.GHSL_LAYERS_V3 / "sample_definitions.parquet",
                            columns=["grid_id", "cell_anbh_derived_m", "in_national_footprint"])
    cov = pd.read_parquet(cfg.COVARIATES / "grid_model_ready_500m.parquet",
                           columns=["grid_id", "built_coverage_1975", "built_coverage_2010",
                                    "built_coverage_2015", "distance_to_nearest_cbd_m"])
    df = samp[samp.in_national_footprint].merge(cov, on="grid_id", how="left")
    n0 = len(df)
    df = df[df.cell_anbh_derived_m.notna()].copy()
    n1 = len(df)
    df["growth_1975_2015"] = df["built_coverage_2015"] - df["built_coverage_1975"]
    # Same four covariates script 04's regression sample requires (Requirement 6/9.4.7),
    # kept identical here so the typology sample matches the regression sample exactly.
    needed = ["built_coverage_1975", "built_coverage_2010", "built_coverage_2015",
              "distance_to_nearest_cbd_m"]
    df = df.dropna(subset=needed).copy()
    n2 = len(df)
    df["growth_1975_2010"] = df["built_coverage_2010"] - df["built_coverage_1975"]
    df["growth_2010_2015"] = df["built_coverage_2015"] - df["built_coverage_2010"]
    log.info("Sample: national-footprint=%d -> drop %d (no ANBH, nodata) -> drop %d "
             "(missing coverage/CBD-distance covariates) -> final n=%d", n0, n0 - n1, n1 - n2, n2)
    return df


def crosstab_shares(vert: pd.Series, horz: pd.Series, v_edges, h_edges) -> pd.DataFrame:
    v_class = np.digitize(vert, v_edges)
    h_class = np.digitize(horz, h_edges)
    n = len(vert)
    table = pd.crosstab(v_class, h_class)
    return (table / n * 100).round(4)


def main() -> None:
    cfg.PAPER2_V3_TABLES.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    n = len(df)

    vert, horz = df["cell_anbh_derived_m"], df["growth_1975_2015"]
    v_med, h_med = float(vert.median()), float(horz.median())

    v_above, v_eq, v_below = int((vert > v_med).sum()), int((vert == v_med).sum()), int((vert < v_med).sum())
    h_above, h_eq, h_below = int((horz > h_med).sum()), int((horz == h_med).sum()), int((horz < h_med).sum())
    log.info("Vertical (ANBH) median=%.6f: above=%d equal=%d below=%d", v_med, v_above, v_eq, v_below)
    log.info("Horizontal (growth 1975-2015) median=%.6f: above=%d equal=%d below=%d", h_med, h_above, h_eq, h_below)

    v_high = vert > v_med   # ties -> low, disclosed rule
    h_high = horz > h_med
    df["typology_class"] = [LABELS[(vh, hh)] for vh, hh in zip(v_high, h_high)]

    counts = df["typology_class"].value_counts()
    median_table = pd.DataFrame({
        "class_label": counts.index, "n": counts.values,
        "pct_of_sample": (counts.values / n * 100).round(4),
    })
    median_table.to_csv(cfg.PAPER2_V3_TABLES / "T_typology_median.csv", index=False)
    log.info("Median-split typology:\n%s", median_table.to_string(index=False))

    profile = df.groupby("typology_class").agg(
        n=("grid_id", "size"),
        mean_anbh=("cell_anbh_derived_m", "mean"),
        median_anbh=("cell_anbh_derived_m", "median"),
        mean_growth_1975_2015=("growth_1975_2015", "mean"),
        median_growth_1975_2015=("growth_1975_2015", "median"),
    ).round(4)
    profile.to_csv(cfg.PAPER2_V3_TABLES / "T_typology_profile.csv")

    # ------------------------------------- Requirement 9 cross-check (Reviewer #2,
    # Major 6): the four Section 4.7 regression predictors, by typology class --
    # not previously carried into this typology rebuild, added here rather than
    # left as an asserted "similar pattern holds" in the manuscript text.
    predictor_profile = df.groupby("typology_class").agg(
        n=("grid_id", "size"),
        mean_built_coverage_2010=("built_coverage_2010", "mean"),
        mean_growth_1975_2010=("growth_1975_2010", "mean"),
        mean_growth_2010_2015=("growth_2010_2015", "mean"),
        mean_distance_to_cbd_km=("distance_to_nearest_cbd_m", lambda s: s.mean() / 1000.0),
    ).round(4)
    predictor_profile = predictor_profile.reindex(list(LABELS.values()))
    predictor_profile.to_csv(cfg.PAPER2_V3_TABLES / "T_typology_predictor_profile.csv")
    log.info("Predictor profile by typology class:\n%s", predictor_profile.to_string())

    # ---------------------------------------------- tertile / quartile ----
    tert_edges_v = vert.quantile([1 / 3, 2 / 3]).values
    tert_edges_h = horz.quantile([1 / 3, 2 / 3]).values
    tert = crosstab_shares(vert, horz, tert_edges_v, tert_edges_h)
    tert.index.name, tert.columns.name = "vertical_tertile", "horizontal_tertile"
    tert.to_csv(cfg.PAPER2_V3_TABLES / "T_typology_tertile_full.csv")

    quart_edges_v = vert.quantile([0.25, 0.5, 0.75]).values
    quart_edges_h = horz.quantile([0.25, 0.5, 0.75]).values
    quart = crosstab_shares(vert, horz, quart_edges_v, quart_edges_h)
    quart.index.name, quart.columns.name = "vertical_quartile", "horizontal_quartile"
    quart.to_csv(cfg.PAPER2_V3_TABLES / "T_typology_quartile_full.csv")

    median_diag = median_table.loc[
        median_table.class_label.isin([LABELS[(True, True)], LABELS[(False, False)]]),
        "pct_of_sample",
    ].sum()
    tert_diag = float(np.trace(tert.values))
    quart_diag = float(np.trace(quart.values))

    sens_summary = pd.DataFrame([
        {"scheme": "median (2x2)", "n_diagonal_cells_summed": "2 of 4", "diagonal_pct": round(median_diag, 2),
         "off_diagonal_pct": round(100 - median_diag, 2)},
        {"scheme": "tertile (3x3)", "n_diagonal_cells_summed": "3 of 9", "diagonal_pct": round(tert_diag, 2),
         "off_diagonal_pct": round(100 - tert_diag, 2)},
        {"scheme": "quartile (4x4)", "n_diagonal_cells_summed": "4 of 16", "diagonal_pct": round(quart_diag, 2),
         "off_diagonal_pct": round(100 - quart_diag, 2)},
    ])
    sens_summary.to_csv(cfg.PAPER2_V3_TABLES / "T_typology_sensitivity_summary.csv", index=False)
    log.info("Sensitivity summary:\n%s", sens_summary.to_string(index=False))

    # ------------------------------------- AGBH-matched comparison typology
    # Same sample, same horizontal axis (growth_1975_2015), same tie rule --
    # ONLY the vertical variable changes (AGBH in place of ANBH). This is
    # the cleanest possible like-for-like check of how much of the ANBH
    # typology's weak diagonal concordance (above) is attributable
    # specifically to ANBH's retrieval-floor problem (script 02/audit_report
    # Section 3) rather than to the choice of horizontal axis or sample.
    agbh_series = pd.read_parquet(cfg.GHSL_LAYERS_V3 / "agbh_anbh_grid_500m.parquet",
                                   columns=["grid_id", "cell_agbh_mean_m"])
    df_agbh = df[["grid_id", "growth_1975_2015"]].merge(agbh_series, on="grid_id", how="left")
    df_agbh = df_agbh.dropna(subset=["cell_agbh_mean_m"])
    assert len(df_agbh) == n, f"AGBH-matched sample ({len(df_agbh)}) must equal ANBH sample ({n})"

    vert_a, horz_a = df_agbh["cell_agbh_mean_m"], df_agbh["growth_1975_2015"]
    v_med_a, h_med_a = float(vert_a.median()), float(horz_a.median())
    v_high_a, h_high_a = vert_a > v_med_a, horz_a > h_med_a
    median_diag_a = 100 * ((v_high_a & h_high_a).sum() + (~v_high_a & ~h_high_a).sum()) / n

    tert_edges_va = vert_a.quantile([1 / 3, 2 / 3]).values
    tert_a = crosstab_shares(vert_a, horz_a, tert_edges_va, tert_edges_h)  # same horizontal edges as ANBH version
    tert_diag_a = float(np.trace(tert_a.values))

    agbh_comparison = pd.DataFrame([
        {"vertical_var": "ANBH (primary)", "scheme": "median", "diagonal_pct": round(median_diag, 2)},
        {"vertical_var": "AGBH (matched comparison)", "scheme": "median", "diagonal_pct": round(median_diag_a, 2)},
        {"vertical_var": "ANBH (primary)", "scheme": "tertile", "diagonal_pct": round(tert_diag, 2)},
        {"vertical_var": "AGBH (matched comparison)", "scheme": "tertile", "diagonal_pct": round(tert_diag_a, 2)},
    ])
    agbh_comparison.to_csv(cfg.PAPER2_V3_TABLES / "T_typology_agbh_matched_comparison.csv", index=False)
    log.info("AGBH-matched comparison (same sample, same horizontal axis, ANBH swapped for AGBH):\n%s",
              agbh_comparison.to_string(index=False))

    out_path = cfg.GHSL_LAYERS_V3 / "typology_v3.parquet"
    df[["grid_id", "cell_anbh_derived_m", "growth_1975_2015", "typology_class"]].to_parquet(out_path, index=False)
    log.info("Saved %s (n=%d)", out_path.relative_to(cfg.PROJECT_ROOT), len(df))

    import json
    with open(cfg.PAPER2_V3_TABLES / "typology_construction_report.json", "w") as f:
        json.dump({
            "n_final": n, "vertical_var": "cell_anbh_derived_m", "horizontal_var": "growth_1975_2015",
            "vertical_median": v_med, "horizontal_median": h_med,
            "vertical_ties_n": v_eq, "horizontal_ties_n": h_eq,
            "tie_rule": "cells exactly equal to a median are assigned to the LOW class on that axis",
        }, f, indent=2)


if __name__ == "__main__":
    main()
