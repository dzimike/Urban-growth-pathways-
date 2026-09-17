"""
Paper 2 v3 -- Script 07: Validation module (Requirement 11)

This pipeline has no independent ground-truth building-height dataset to
validate against directly (no LiDAR, no field survey already exists for
this project). Three complementary, evidence-based checks are run
instead, following the existing project pattern in
scripts/15_validation_sampling.py rather than inventing a new one:

  A. Spatial plausibility: does the ANBH-derived typology concentrate in
     the places independently known (from administrative/geographic
     region definitions, not from GHSL data) to have taller buildings --
     Accra-Tema and Kumasi cores -- and not in secondary cities?

  B. Cross-dataset consistency: does cell-level ANBH/AGBH correlate
     sensibly with INDEPENDENT building-size metrics derived from a
     different dataset and a different measurement process (Google Open
     Buildings V3 footprint geometry: median_building_area,
     building_area_gini, share_very_large)? A positive, non-trivial
     correlation is expected if GHSL height data is behaving sensibly
     (taller cells should tend to also have larger, more unequal
     building footprints -- commercial/institutional towers are large
     buildings, not just tall ones).

  C. Structured manual-validation sample + coding template: a stratified
     sample (region tier x ANBH-derived height class) with GHSL-derived
     values pre-filled, so a human validator (field visit, street-level
     imagery, or high-resolution satellite imagery) can directly compare
     an independently observed storey count against what the model
     implies. This produces the INFRASTRUCTURE for validation; it does
     not itself constitute completed field validation, which is outside
     the scope of what this pipeline can execute.

Outputs
-------
    04_outputs/paper2_v3/tables/T_validation_A_spatial_plausibility.csv
    04_outputs/paper2_v3/tables/T_validation_B_cross_dataset.csv
    02_processed_data/validation_samples/v3_manual_validation_sample.parquet
    04_outputs/paper2_v3/tables/v3_manual_validation_coding_template.csv

Usage
-----
    python scripts/revision_v3/07_validation_module.py
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg

logging.basicConfig(level=cfg.LOG_LEVEL, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

VALIDATION_DIR = cfg.PROCESSED / "validation_samples"

MANUAL_CODING_FIELDS = [
    "grid_id", "stratum", "longitude_centroid", "latitude_centroid",
    "rs_anbh_derived_m", "rs_agbh_mean_m", "rs_implied_storeys_3m_rule",
    "validator_name", "validation_date", "imagery_source",
    "observed_dominant_storeys",   # 1 / 2 / 3 / 4plus / mixed / unclear
    "height_class_match",          # matches_ghsl / ghsl_overestimates / ghsl_underestimates / unclear
    "dominant_building_use",       # residential / commercial / industrial / institutional / mixed / unknown
    "settlement_type",             # formal / informal / compound / estate / industrial / other
    "validation_notes",
    "data_quality_flag",           # ok / suspect / exclude
]


def load_base() -> pd.DataFrame:
    typ = pd.read_parquet(cfg.GHSL_LAYERS_V3 / "typology_v3.parquet")
    cov = pd.read_parquet(cfg.COVARIATES / "grid_model_ready_500m.parquet",
                           columns=["grid_id", "urban_region_name", "median_building_area",
                                    "building_area_gini", "share_very_large", "building_count"])
    agg = pd.read_parquet(cfg.GHSL_LAYERS_V3 / "agbh_anbh_grid_500m.parquet",
                           columns=["grid_id", "cell_agbh_mean_m"])
    df = typ.merge(cov, on="grid_id", how="left").merge(agg, on="grid_id", how="left")
    return df


def part_a_spatial_plausibility(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab typology class against the INDEPENDENT urban_region_name
    classification (built from administrative boundaries, not from GHSL
    height data at all -- see scripts/02_download_boundaries.py)."""
    named = df[df.urban_region_name.notna()].copy()
    named["region_tier"] = np.where(
        named.urban_region_name.isin(["Accra-Tema", "Kumasi"]), "Metro core (Accra-Tema, Kumasi)",
        np.where(named.urban_region_name.isin(["Accra-Kasoa", "Accra-Dodowa", "Accra-Prampram",
                                                  "Kumasi-Ejisu", "Tamale-Savelugu"]),
                 "Peri-urban corridor", "Secondary city"),
    )
    ct = pd.crosstab(named.region_tier, named.typology_class, normalize="index").round(4) * 100
    log.info("Part A -- typology class share (%%) by independent region tier:\n%s", ct.to_string())

    high_height_share = ct[[c for c in ct.columns if c.startswith("High height")]].sum(axis=1)
    log.info("Part A -- 'High height' share by region tier:\n%s", high_height_share.round(1).to_string())
    return ct


def part_b_cross_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Correlate ANBH/AGBH against building-size metrics from an entirely
    different dataset (Open Buildings V3 footprint geometry)."""
    built = df[df.building_count.fillna(0) > 0].copy()
    rows = []
    for h_col in ["cell_anbh_derived_m", "cell_agbh_mean_m"]:
        for m_col in ["median_building_area", "building_area_gini", "share_very_large"]:
            sub = built[[h_col, m_col]].dropna()
            if len(sub) < 100:
                continue
            pr, pp = st.pearsonr(sub[h_col], sub[m_col])
            sr, sp = st.spearmanr(sub[h_col], sub[m_col])
            rows.append({"height_var": h_col, "building_size_metric": m_col, "n": len(sub),
                         "pearson_r": round(pr, 4), "pearson_p": pp,
                         "spearman_r": round(sr, 4), "spearman_p": sp})
    out = pd.DataFrame(rows)
    log.info("Part B -- cross-dataset correlations:\n%s", out.to_string(index=False))
    return out


def part_c_manual_validation_sample(df: pd.DataFrame, per_stratum: int = 20, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    named = df[df.urban_region_name.notna() & df.cell_anbh_derived_m.notna()].copy()
    named["region_tier"] = np.where(
        named.urban_region_name.isin(["Accra-Tema", "Kumasi"]), "Metro_core",
        np.where(named.urban_region_name.isin(["Accra-Kasoa", "Accra-Dodowa", "Accra-Prampram",
                                                  "Kumasi-Ejisu", "Tamale-Savelugu"]),
                 "Periurban_corridor", "Secondary_city"),
    )
    med_anbh = named.cell_anbh_derived_m.median()
    named["height_tier"] = np.where(named.cell_anbh_derived_m > med_anbh, "HighANBH", "LowANBH")
    named["stratum"] = named.region_tier + "_" + named.height_tier

    rng = np.random.default_rng(seed)
    frames = []
    for stratum, group in named.groupby("stratum"):
        n = min(per_stratum, len(group))
        if n == 0:
            continue
        idx = rng.choice(group.index, size=n, replace=False)
        frames.append(group.loc[idx])
    sample = pd.concat(frames, ignore_index=True) if frames else named.iloc[0:0]
    log.info("Part C -- stratified sample: %d cells across %d strata (target %d/stratum)",
              len(sample), sample.stratum.nunique() if len(sample) else 0, per_stratum)

    grid = gpd.read_parquet(cfg.GRIDS / "grid_500m.parquet")[["grid_id", "geometry"]]
    sample_geo = grid.merge(sample, on="grid_id", how="inner")
    centroids = sample_geo.geometry.centroid
    sample_geo["longitude_centroid"] = centroids.x.round(6)
    sample_geo["latitude_centroid"] = centroids.y.round(6)

    template_rows = []
    for _, row in sample_geo.iterrows():
        implied_storeys = round(row["cell_anbh_derived_m"] / 3.0, 1) if pd.notna(row["cell_anbh_derived_m"]) else ""
        entry = {
            "grid_id": row["grid_id"], "stratum": row["stratum"],
            "longitude_centroid": row["longitude_centroid"], "latitude_centroid": row["latitude_centroid"],
            "rs_anbh_derived_m": round(row["cell_anbh_derived_m"], 3) if pd.notna(row["cell_anbh_derived_m"]) else "",
            "rs_agbh_mean_m": round(row["cell_agbh_mean_m"], 4) if pd.notna(row["cell_agbh_mean_m"]) else "",
            "rs_implied_storeys_3m_rule": implied_storeys,
        }
        for f in MANUAL_CODING_FIELDS[7:]:
            entry[f] = ""
        template_rows.append(entry)
    template = pd.DataFrame(template_rows, columns=MANUAL_CODING_FIELDS)
    return sample_geo, template


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    cfg.PAPER2_V3_TABLES.mkdir(parents=True, exist_ok=True)

    df = load_base()

    ct = part_a_spatial_plausibility(df)
    ct.to_csv(cfg.PAPER2_V3_TABLES / "T_validation_A_spatial_plausibility.csv")

    b = part_b_cross_dataset(df)
    b.to_csv(cfg.PAPER2_V3_TABLES / "T_validation_B_cross_dataset.csv", index=False)

    sample_geo, template = part_c_manual_validation_sample(df)
    sample_out = sample_geo.drop(columns=["geometry"])
    sample_out.to_parquet(VALIDATION_DIR / "v3_manual_validation_sample.parquet", index=False)
    template.to_csv(cfg.PAPER2_V3_TABLES / "v3_manual_validation_coding_template.csv", index=False)
    log.info("Saved manual validation sample (n=%d) and blank coding template.", len(sample_out))


if __name__ == "__main__":
    main()
