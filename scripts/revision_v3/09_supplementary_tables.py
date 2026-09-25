"""
Paper 2 v3 -- Script 09: Supplementary statistics for the Appendix

Produces the reviewer-requested statistics that were previously only
summarised in the text:

  1. Full-sample descriptive statistics (Reviewer #2, Major 4): mean, SD,
     min, p1/p5/p10/p25/p50/p75/p90/p95/p99, max for ANBH, AGBH, built
     fraction, and the four Section 4.7 predictors, on the final analytical
     sample (n = 296,668).
  2. Per-region statistics for the 15 named study areas (Reviewer #4,
     point 1 / Reviewer #6): cells, ANBH and AGBH summaries, 2010 built
     coverage, prior expansion, and typology shares, plus rows for cells
     outside all named regions and for the national sample. Region
     membership is the single `urban_region_name` assignment used
     everywhere else in this analysis (script 07), so the 15 rows sum to
     the 28,129 named-region cells of the final sample (28,133 in the
     national-footprint sample before the 4 nodata/missing-covariate
     exclusions that fall in named regions).
  3. The full tertile and quartile typology cross-tabulations (Reviewer #4,
     point 4), re-expressed with row/column marginals and the cut-points
     used, from script 06's CSVs.

Outputs
-------
    04_outputs/paper2_v3/tables/T_descriptive_statistics.csv
    04_outputs/paper2_v3/tables/T_regional_statistics.csv
    04_outputs/paper2_v3/tables/T_typology_crosstabs_with_marginals.csv

Usage
-----
    python scripts/revision_v3/09_supplementary_tables.py
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

QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

# Order used in Section 3 / Table 1 of the manuscript.
REGION_ORDER = [
    "Accra-Tema", "Kumasi", "Sekondi-Takoradi", "Tamale", "Cape Coast", "Koforidua",
    "Sunyani", "Ho", "Wa", "Bolgatanga", "Accra-Kasoa", "Accra-Prampram",
    "Accra-Dodowa", "Kumasi-Ejisu", "Tamale-Savelugu",
]


def build_final_sample() -> pd.DataFrame:
    """Same construction as scripts 04 and 06 (final n = 296,668)."""
    samp = pd.read_parquet(cfg.GHSL_LAYERS_V3 / "sample_definitions.parquet",
                           columns=["grid_id", "cell_agbh_mean_m", "cell_anbh_derived_m",
                                    "cell_built_fraction_mean", "in_national_footprint"])
    cov = pd.read_parquet(cfg.COVARIATES / "grid_model_ready_500m.parquet",
                          columns=["grid_id", "urban_region_name", "built_coverage_1975",
                                   "built_coverage_2010", "built_coverage_2015",
                                   "distance_to_nearest_cbd_m"])
    df = samp[samp.in_national_footprint].merge(cov, on="grid_id", how="left")
    df = df[df.cell_anbh_derived_m.notna()]
    df = df.dropna(subset=["built_coverage_1975", "built_coverage_2010",
                           "built_coverage_2015", "distance_to_nearest_cbd_m"]).copy()
    df["growth_1975_2010"] = df.built_coverage_2010 - df.built_coverage_1975
    df["growth_2010_2015"] = df.built_coverage_2015 - df.built_coverage_2010
    df["growth_1975_2015"] = df.built_coverage_2015 - df.built_coverage_1975
    typ = pd.read_parquet(cfg.GHSL_LAYERS_V3 / "typology_v3.parquet", columns=["grid_id", "typology_class"])
    df = df.merge(typ, on="grid_id", how="left")
    assert len(df) == 296_668 and df.typology_class.notna().all(), len(df)
    return df


def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    variables = {
        "cell_anbh_derived_m": "ANBH derived (m)",
        "cell_agbh_mean_m": "AGBH mean (m)",
        "cell_built_fraction_mean": "Built fraction, 2018",
        "built_coverage_2010": "Built coverage 2010",
        "growth_1975_2010": "Growth 1975-2010",
        "growth_2010_2015": "Growth 2010-2015",
        "distance_to_nearest_cbd_m": "Distance to nearest CBD (km)",
    }
    rows = []
    for col, label in variables.items():
        s = df[col] / 1000.0 if col == "distance_to_nearest_cbd_m" else df[col]
        q = s.quantile(QUANTILES)
        row = {"variable": label, "n": int(s.notna().sum()), "mean": s.mean(), "sd": s.std(),
               "min": s.min()}
        for p_, v in zip(QUANTILES, q.values):
            row[f"p{int(round(p_ * 100))}"] = v
        row["max"] = s.max()
        rows.append(row)
    return pd.DataFrame(rows)


def regional_table(df: pd.DataFrame) -> pd.DataFrame:
    def summarise(sub: pd.DataFrame, name: str) -> dict:
        high_h = sub.typology_class.str.startswith("High height")
        diag = sub.typology_class.isin(["High height / High prior expansion",
                                        "Low height / Low prior expansion"])
        return {
            "region": name, "n_cells": len(sub),
            "anbh_mean_m": sub.cell_anbh_derived_m.mean(),
            "anbh_sd_m": sub.cell_anbh_derived_m.std(),
            "anbh_median_m": sub.cell_anbh_derived_m.median(),
            "anbh_p90_m": sub.cell_anbh_derived_m.quantile(0.90),
            "anbh_p99_m": sub.cell_anbh_derived_m.quantile(0.99),
            "agbh_mean_m": sub.cell_agbh_mean_m.mean(),
            "agbh_median_m": sub.cell_agbh_mean_m.median(),
            "built_coverage_2010_mean": sub.built_coverage_2010.mean(),
            "growth_1975_2015_mean": sub.growth_1975_2015.mean(),
            "pct_high_height": 100 * high_h.mean(),
            "pct_diagonal": 100 * diag.mean(),
        }

    rows = [summarise(df[df.urban_region_name == r], r) for r in REGION_ORDER]
    named = df[df.urban_region_name.notna()]
    rows.append(summarise(named, "All 15 named regions"))
    rows.append(summarise(df[df.urban_region_name.isna()], "Outside named regions"))
    rows.append(summarise(df, "National analytical sample"))
    out = pd.DataFrame(rows)
    assert out.loc[out.region.isin(REGION_ORDER), "n_cells"].sum() == len(named)
    return out


def crosstabs_with_marginals() -> pd.DataFrame:
    """Long-format tertile/quartile tables (percent of the full sample) with
    marginals. Rows = ANBH class (1 = lowest), columns = growth_1975_2015
    class. Script 06 uses np.digitize, which assigns a value equal to a
    cut-point to the UPPER class -- unlike the median split, where ties go
    to the low class. This is disclosed in the Appendix table note."""
    out = []
    for scheme, fname in [("tertile", "T_typology_tertile_full.csv"),
                          ("quartile", "T_typology_quartile_full.csv")]:
        t = pd.read_csv(cfg.PAPER2_V3_TABLES / fname, index_col=0)
        t.columns = [int(c) + 1 for c in t.columns]
        t.index = [int(i) + 1 for i in t.index]
        t["Row total"] = t.sum(axis=1)
        t.loc["Column total"] = t.sum(axis=0)
        t.columns = [str(c) for c in t.columns]
        t = t.reset_index().rename(columns={"index": "anbh_class"})
        t.insert(0, "scheme", scheme)
        out.append(t)
    ct = pd.concat(out, ignore_index=True)
    class_cols = sorted([c for c in ct.columns if c.isdigit()], key=int)
    return ct[["scheme", "anbh_class"] + class_cols + ["Row total"]]


def main() -> None:
    df = build_final_sample()
    log.info("Final analytical sample: n=%d", len(df))

    desc = descriptive_table(df)
    desc.to_csv(cfg.PAPER2_V3_TABLES / "T_descriptive_statistics.csv", index=False)
    log.info("Descriptive statistics:\n%s", desc.round(4).to_string(index=False))

    reg = regional_table(df)
    reg.to_csv(cfg.PAPER2_V3_TABLES / "T_regional_statistics.csv", index=False)
    log.info("Regional statistics:\n%s", reg.round(3).to_string(index=False))

    ct = crosstabs_with_marginals()
    ct.to_csv(cfg.PAPER2_V3_TABLES / "T_typology_crosstabs_with_marginals.csv", index=False)
    log.info("Cross-tabs with marginals:\n%s", ct.round(2).to_string(index=False))

    n_zero = int((df.cell_anbh_derived_m == 0).sum())
    iqr_lo, iqr_hi = df.cell_anbh_derived_m.quantile([0.25, 0.75])
    log.info("Cells with ANBH == 0 (footprint present, no GHSL built signal): %d (%.2f%%); "
             "ANBH IQR %.4f-%.4f m (width %.4f m)", n_zero, 100 * n_zero / len(df),
             iqr_lo, iqr_hi, iqr_hi - iqr_lo)
    for cls, sub in df.groupby("typology_class"):
        log.info("  zero-ANBH cells in %s: %d", cls, int((sub.cell_anbh_derived_m == 0).sum()))

    # Cut-points for the table note (same construction as script 06).
    v, h = df.cell_anbh_derived_m, df.growth_1975_2015
    for name, qs in [("tertile", [1 / 3, 2 / 3]), ("quartile", [0.25, 0.5, 0.75])]:
        log.info("%s cut-points: ANBH %s | growth %s", name,
                 np.round(v.quantile(qs).values, 4), np.round(h.quantile(qs).values, 6))
        for e in h.quantile(qs).values:
            log.info("  growth ties at %s cut-point %.6f: %d cells", name, e, int((h == e).sum()))


if __name__ == "__main__":
    main()
