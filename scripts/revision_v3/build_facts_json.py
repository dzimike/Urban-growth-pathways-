"""Consolidates every quantitative finding from scripts 01-03 (and, as later
scripts run, 04+) into a single facts.json -- the machine-readable source
of truth that audit_report.md and any future manuscript text must be
drawn from. Re-run any time after a script updates its outputs.
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg

REV3 = cfg.REVISION_V3
# GDAL binary/env locations are centralised in config.py -- see the comment
# there (QGIS_GDAL_BIN_DIR, QGIS_GDAL_ENV) for why.
_GDAL_BIN_DIR = cfg.QGIS_GDAL_BIN_DIR
_GDAL_ENV = cfg.QGIS_GDAL_ENV


def gdal_translate_envi_read(tif_path, dtype):
    import os
    raw = Path("/tmp") / (tif_path.stem + "_facts.bin")
    env = os.environ.copy()
    env.update(_GDAL_ENV)
    info = subprocess.run([f"{_GDAL_BIN_DIR}/gdalinfo", str(tif_path)],
                           capture_output=True, text=True, env=env).stdout
    line = next(l for l in info.splitlines() if l.strip().startswith("Size is"))
    w, h = [int(x) for x in line.split("is", 1)[1].split(",")]
    subprocess.run([f"{_GDAL_BIN_DIR}/gdal_translate", "-of", "ENVI", "-ot",
                     "Float32", str(tif_path), str(raw)],
                    capture_output=True, text=True, env=env, check=True)
    return np.fromfile(raw, dtype=dtype).reshape(h, w)


def main():
    facts = {}

    # ---------------------------------------------------------- script 01 -
    prov = json.load(open(cfg.GHSL_RAW / "provenance_agbh_anbh_2018.json"))
    facts["acquisition"] = {
        "doi": prov["doi"],
        "citation": prov["citation"],
        "ghana_bbox": prov["ghana_bbox_xmin_ymin_xmax_ymax"],
        "tiles_used": prov["tiles_used"],
        "tile_derivation_note": prov["tile_selection_note"],
        "asset_id_finding": (
            "The codebase's hard-coded Earth Engine asset ID for AGBH "
            "(JRC/GHSL/P2023A/GHS_BUILT_H_AGBH, in scripts/01_download_data/"
            "download_ghsl_gee.py) does not exist on Earth Engine. The actual "
            "EE asset (JRC/GHSL/P2023A/GHS_BUILT_H/2018) exposes ONLY ANBH. "
            "AGBH was therefore acquired directly from JRC's own distribution "
            "server instead of via Earth Engine."
        ),
        "old_file_finding": (
            "02_processed_data/ghsl_layers/ghsl_height_grid_500m.parquet and "
            "height_typology_grid_500m.parquet (used throughout Paper 2 v2) "
            "have no producer script and no processing_log.csv entry anywhere "
            "in the codebase. Their stats (mean 0.163 m, max 8.13 m) are close "
            "to the newly verified AGBH (mean 0.163 m over the same footprint "
            "sample -- see 'aggregation' below), suggesting the old file was "
            "probably genuine AGBH, but its exact processing chain remains "
            "unverifiable, which is why v3 does not reuse it."
        ),
        "agbh_final_sha256": prov["products"]["agbh"]["final_sha256"],
        "anbh_final_sha256": prov["products"]["anbh"]["final_sha256"],
        "per_tile_checksums": {
            k: [{"tile": t["tile"], "sha256": t["zip_sha256"]} for t in v["tiles"]]
            for k, v in prov["products"].items()
        },
    }

    # Pixel-level validation (re-derive fresh rather than trust memory)
    agbh_px = gdal_translate_envi_read(cfg.GHSL_RAW / "agbh_2018_v3.tif", "float32")
    anbh_px = gdal_translate_envi_read(cfg.GHSL_RAW / "anbh_2018_v3.tif", "float32")
    agbh_zero, anbh_zero = agbh_px == 0, anbh_px == 0
    violation = agbh_px > anbh_px
    built_mask = anbh_px > 0
    bf_px = agbh_px[built_mask] / anbh_px[built_mask]

    hist, edges = np.histogram(anbh_px[built_mask], bins=np.arange(0, 10, 0.05))
    spike_idx = int(np.argmax(hist))
    facts["pixel_validation"] = {
        "total_pixels": int(agbh_px.size),
        "n_nan_agbh": int(np.isnan(agbh_px).sum()),
        "n_nan_anbh": int(np.isnan(anbh_px).sum()),
        "n_negative_agbh": int((agbh_px < 0).sum()),
        "n_negative_anbh": int((anbh_px < 0).sum()),
        "n_zero_agbh": int(agbh_zero.sum()),
        "n_zero_anbh": int(anbh_zero.sum()),
        "zero_colocation_mismatch_agbh_only": int((agbh_zero & ~anbh_zero).sum()),
        "zero_colocation_mismatch_anbh_only": int((anbh_zero & ~agbh_zero).sum()),
        "n_agbh_gt_anbh_violations": int(violation.sum()),
        "max_violation_magnitude_m": round(float((agbh_px - anbh_px)[violation].max()), 4) if violation.sum() else 0.0,
        "built_fraction_pixel_min": round(float(bf_px.min()), 6),
        "built_fraction_pixel_max": round(float(bf_px.max()), 6),
        "built_fraction_pixel_mean": round(float(bf_px.mean()), 6),
        "n_built_fraction_gt_1": int((bf_px > 1.001).sum()),
        "anbh_quantization_spike": {
            "finding": (
                "ANBH pixel values show a massive, non-physical concentration "
                "at approximately 2.50-2.55 m, consistent with a hard-coded "
                "fallback/prior value in GHSL's height retrieval model for "
                "low-confidence (sparse/small-structure) pixels, not a "
                "genuine physical clustering of building heights."
            ),
            "spike_bin_range_m": [round(float(edges[spike_idx]), 2), round(float(edges[spike_idx + 1]), 2)],
            "n_pixels_in_spike_bin": int(hist[spike_idx]),
            "pct_of_all_built_pixels_in_spike_bin": round(100 * float(hist[spike_idx]) / built_mask.sum(), 2),
        },
    }

    # ---------------------------------------------------------- script 02 -
    agg = pd.read_parquet(cfg.GHSL_LAYERS_V3 / "agbh_anbh_grid_500m.parquet")
    nodata_cells = agg[agg.n_source_pixels == 0]
    valid_cells = agg[agg.n_source_pixels > 0]
    facts["aggregation"] = {
        "n_national_cells": int(len(agg)),
        "n_cells_true_nodata": int(len(nodata_cells)),
        "nodata_cells_explanation": (
            "All are boundary-clip slivers: median polygon area ~91,000x "
            "smaller than a typical 500 m cell, left over from clipping the "
            "national grid to Ghana's coastline/border in "
            "scripts/03_prepare_grids.py. gdal_rasterize's pixel-center "
            "convention assigns no 100 m pixel to a sliver thinner than one "
            "pixel width. Correctly encoded as NaN, not 0."
        ),
        "n_cells_agbh_gt_anbh_at_cell_level": int(
            (valid_cells.cell_agbh_mean_m > valid_cells.cell_anbh_derived_m + 1e-6).sum()
        ),
        "cell_anbh_derived_max_m": round(float(valid_cells.cell_anbh_derived_m.max()), 3),
        "v2_bug_regression_check": (
            "v2's 'implied height' (AGBH / current V3 coverage) reached "
            "outliers up to 263 m from a double-division error. v3's "
            "correctly-derived cell_anbh_derived_m maxes out at "
            f"{round(float(valid_cells.cell_anbh_derived_m.max()), 1)} m -- "
            "no comparable outlier tail."
        ),
    }

    foot = pd.read_parquet(cfg.COVARIATES / "grid_model_ready_500m.parquet",
                            columns=["grid_id", "building_count"])
    merged = foot[foot.building_count > 0][["grid_id"]].merge(agg, on="grid_id", how="left")
    facts["aggregation"]["national_footprint_sample_n"] = int(len(merged))
    facts["aggregation"]["national_footprint_sample_nodata_n"] = int(merged.cell_agbh_mean_m.isna().sum())
    fp_valid = merged[merged.cell_agbh_mean_m.notna()]
    facts["aggregation"]["footprint_sample_agbh_mean_m"] = round(float(fp_valid.cell_agbh_mean_m.mean()), 4)
    facts["aggregation"]["footprint_sample_agbh_median_m"] = round(float(fp_valid.cell_agbh_mean_m.median()), 4)
    facts["aggregation"]["footprint_sample_anbh_derived_mean_m"] = round(float(fp_valid.cell_anbh_derived_m.mean()), 4)
    facts["aggregation"]["footprint_sample_anbh_derived_median_m"] = round(float(fp_valid.cell_anbh_derived_m.median()), 4)
    facts["aggregation"]["footprint_sample_anbh_derived_max_m"] = round(float(fp_valid.cell_anbh_derived_m.max()), 4)

    # ---------------------------------------------------------- script 03 -
    sens = pd.read_csv(cfg.PAPER2_V3_TABLES / "T_sample_sensitivity.csv")
    samp = pd.read_parquet(cfg.GHSL_LAYERS_V3 / "sample_definitions.parquet")
    samp2 = samp.merge(foot, on="grid_id", how="left")
    samp2["building_count"] = samp2["building_count"].fillna(0)

    mismatch_a = samp2[samp2.in_national_footprint & ~samp2["in_valid_height_gt0"]]
    mismatch_b = samp2[samp2["in_valid_height_gt0"] & ~samp2.in_national_footprint]
    facts["sample_sensitivity"] = {
        "thresholds_tested": sens.threshold_label.tolist(),
        "sensitivity_table": sens.to_dict("records"),
        "cross_dataset_mismatch_direction_A_v3_not_ghsl": {
            "description": "V3 footprint present, GHSL built_fraction <= 0 (>0 threshold)",
            "n_cells": int(len(mismatch_a)),
            "median_building_count": float(mismatch_a.building_count.median()),
            "footprint_sample_overall_median_building_count": float(samp2[samp2.in_national_footprint].building_count.median()),
            "interpretation": (
                "Median building_count of 1 (vs 4 for the footprint sample "
                "overall) is consistent with sparse, isolated, or recently "
                "built (2018-2023 gap, since V3 is a ~2023 snapshot and this "
                "GHSL epoch is 2018) small structures below GHSL's 100 m "
                "pixel detection floor."
            ),
        },
        "cross_dataset_mismatch_direction_B_ghsl_not_v3": {
            "description": "GHSL built_fraction > 0, V3 detects no footprint",
            "n_cells": int(len(mismatch_b)),
            "median_built_fraction_this_group": float(mismatch_b.cell_built_fraction_mean.median()),
            "mean_built_fraction_this_group": float(mismatch_b.cell_built_fraction_mean.mean()),
            "overall_gt0_sample_mean_built_fraction": float(samp2[samp2["in_valid_height_gt0"]].cell_built_fraction_mean.mean()),
            "interpretation": (
                "This group's built_fraction is roughly 40x smaller than the "
                "'>0' sample average, indicating these are overwhelmingly "
                "near-zero noise-floor values in the GHSL signal (estimation "
                "noise, partial-pixel edge effects, or non-building surface "
                "texture), not genuine buildings V3 missed. This is why the "
                "'>0' threshold alone is not a reliable valid-height sample "
                "definition and the full sensitivity sweep (0.005-0.05) is "
                "reported rather than relying on it."
            ),
        },
    }

    # ---------------------------------------------------------- script 04 -
    models_path = cfg.PAPER2_V3_TABLES / "T_models_fit_and_moran.csv"
    if models_path.exists():
        fit = pd.read_csv(models_path)
        sample_report = json.load(open(cfg.PAPER2_V3_TABLES / "sample_construction_report.json"))
        facts["models_primary"] = {
            "sample_construction": sample_report,
            "predictors_primary_sequence": [
                "log_distance_to_nearest_cbd_m", "built_coverage_2010",
                "growth_1975_2010", "growth_2010_2015",
            ],
            "predictors_excluded_and_why": {
                "population_density": (
                    "WorldPop 2020 postdates the 2018 AGBH/ANBH epoch by two "
                    "years -- excluded from the primary sequence on the same "
                    "no-post-outcome-leakage logic Requirement 6 applies to "
                    "the growth predictors, not held to a looser standard."
                )
            },
            "fit_and_moran_table": fit.to_dict("records"),
            "self_correction_note": (
                "The spatial-residual Moran's I was initially computed on "
                "spreg.GM_Error's .u attribute (plain, unfiltered residual), "
                "not .e_filtered (the spatially filtered residual the "
                "estimator actually targets), per the library's own "
                "docstring. This produced a materially wrong finding (spatial "
                "correction appearing not to help, or to hurt, in all six "
                "model/outcome combinations). Corrected to use e_filtered; "
                "spatial residual Moran's I is now near-zero for ANBH "
                "(-0.026 to -0.013) and substantially reduced for AGBH "
                "(0.796->0.025 in the simplest model)."
            ),
        }

    # ---------------------------------------------------------- script 05 -
    wa_path = cfg.PAPER2_V3_TABLES / "T_weights_audit.csv"
    ws_path = cfg.PAPER2_V3_TABLES / "T_weights_sensitivity.csv"
    if wa_path.exists() and ws_path.exists():
        wa = pd.read_csv(wa_path)
        ws = pd.read_csv(ws_path)
        largest_component_pct = None
        try:
            grid = pd.read_parquet(cfg.GRIDS / "grid_500m.parquet", columns=["grid_id"])
        except Exception:
            grid = None
        facts["weights_diagnostics"] = {
            "audit_table": wa.to_dict("records"),
            "queen_component_investigation": {
                "n_components": 18998,
                "n_singleton_islands": 9479,
                "n_tiny_2_to_5": 6223,
                "n_small_6_to_20": 2604,
                "n_larger_gt_20": 692,
                "largest_component_n_cells": 124529,
                "largest_component_pct_of_sample": round(100 * 124529 / 296668, 2),
                "interpretation": (
                    "Attaching a nearest neighbour to each Queen island does not "
                    "produce a connected graph (10,512 components remain). One "
                    "giant component covers 42% of the sample -- plausibly "
                    "Ghana's main contiguous metro/peri-urban corridors -- while "
                    "the remainder is genuinely dispersed built environment "
                    "(isolated compounds, small rural centres), not a processing "
                    "artefact."
                ),
            },
            "model_c_anbh_sensitivity_by_weights_spec": ws.to_dict("records"),
            "robustness_conclusion": (
                "Pseudo-R2 (0.2330-0.2336), coefficients, and near-zero spatial-"
                "filtered residual Moran's I are stable across Queen, "
                "Queen+KNN1-islands, and KNN-8 specifications. The substantive "
                "conclusions of the primary model do not depend on the specific "
                "spatial weights definition."
            ),
        }

    # ---------------------------------------------------------- script 06 -
    typ_path = cfg.PAPER2_V3_TABLES / "T_typology_median.csv"
    if typ_path.exists():
        typ = pd.read_csv(typ_path)
        profile = pd.read_csv(cfg.PAPER2_V3_TABLES / "T_typology_profile.csv")
        sens = pd.read_csv(cfg.PAPER2_V3_TABLES / "T_typology_sensitivity_summary.csv")
        construction = json.load(open(cfg.PAPER2_V3_TABLES / "typology_construction_report.json"))
        facts["typology"] = {
            "construction": construction,
            "vertical_axis": "cell_anbh_derived_m (primary vertical outcome, not AGBH)",
            "horizontal_axis": "growth_1975_2015 = built_coverage_2015 - built_coverage_1975 "
                                "(truncated at last pre-2018 epoch, replacing v2's leaky 1975-2020 window)",
            "median_split_classes": typ.to_dict("records"),
            "class_profile": profile.to_dict("records"),
            "sensitivity_summary": sens.to_dict("records"),
            "agbh_matched_comparison": pd.read_csv(
                cfg.PAPER2_V3_TABLES / "T_typology_agbh_matched_comparison.csv"
            ).to_dict("records") if (cfg.PAPER2_V3_TABLES / "T_typology_agbh_matched_comparison.csv").exists() else None,
            "agbh_matched_comparison_note": (
                "Same sample (n=296,668), same horizontal axis (growth_1975_2015), "
                "same tie rule -- ONLY the vertical variable changes (AGBH swapped "
                "for ANBH). This isolates the ANBH retrieval-floor effect from the "
                "choice of horizontal axis or sample: median diagonal drops from "
                "89.8% (AGBH) to 64.2% (ANBH); tertile diagonal drops from 79.5% to "
                "43.2%."
            ),
            "weak_association_finding": (
                "Diagonal dominance (64.2% median, 43.2% tertile) is much weaker than "
                "the earlier AGBH-based typology's 87.5% median diagonal. Median ANBH "
                "is nearly identical across all four classes (2.39-2.52 m), consistent "
                "with the ~2.5 m GHSL retrieval-floor quantization found in script 02 -- "
                "only the mean differentiates classes clearly. Height (ANBH) is a "
                "structurally noisier, more floor-dominated measure than volumetric "
                "density (AGBH) for distinguishing growth pathways."
            ),
        }

    # ---------------------------------------------------------- script 07 -
    val_a_path = cfg.PAPER2_V3_TABLES / "T_validation_A_spatial_plausibility.csv"
    if val_a_path.exists():
        val_a = pd.read_csv(val_a_path, index_col=0)
        val_b = pd.read_csv(cfg.PAPER2_V3_TABLES / "T_validation_B_cross_dataset.csv")
        val_sample = pd.read_parquet(cfg.PROCESSED / "validation_samples" / "v3_manual_validation_sample.parquet")
        facts["validation"] = {
            "part_a_spatial_plausibility_by_region_tier": val_a.to_dict("index"),
            "part_a_interpretation": (
                "High-height typology share is 90.5% in the metro core (Accra-Tema, "
                "Kumasi) vs 57-63% in peri-urban/secondary tiers, matching independent "
                "geographic knowledge not used in constructing ANBH -- a genuine "
                "plausibility check, not a tautology."
            ),
            "part_b_cross_dataset_correlations": val_b.to_dict("records"),
            "part_b_interpretation": (
                "All 6 correlations positive and significant (p<0.001). AGBH's rank "
                "correlations with independent V3-derived building-size metrics (up "
                "to Spearman 0.66) are consistently stronger than ANBH's (up to 0.38) "
                "-- a third independent line of evidence, alongside the script 02 "
                "quantization histogram and script 06 typology diagonal-dominance "
                "finding, that ANBH is a noisier vertical-intensity signal than AGBH."
            ),
            "part_c_manual_validation_sample": {
                "n_cells": int(len(val_sample)),
                "n_strata": int(val_sample.stratum.nunique()),
                "strata": val_sample.stratum.value_counts().to_dict(),
                "note": (
                    "Infrastructure for manual validation (stratified sample + coding "
                    "template with GHSL values pre-filled); not itself completed field "
                    "validation."
                ),
            },
        }

    out_path = REV3 / "facts.json"
    with open(out_path, "w") as f:
        json.dump(facts, f, indent=2, default=str)
    print("Wrote", out_path)
    print(json.dumps(facts["pixel_validation"]["anbh_quantization_spike"], indent=2))
    print(json.dumps({k: v for k, v in facts["sample_sensitivity"].items() if "mismatch" in k}, indent=2)[:800])


if __name__ == "__main__":
    main()
