"""
Paper 2 v3 -- Script 04: Nested OLS and spatial models (Requirements 5-7)

Design decisions, stated explicitly per the revision instructions:

  Requirement 5 -- ANBH (cell_anbh_derived_m, script 02) is the PRIMARY
  vertical outcome. AGBH (cell_agbh_mean_m) is fit identically but labelled
  throughout as the volumetric-density ROBUSTNESS outcome, never as a
  second primary result.

  Requirement 6 -- only pre-2018 horizontal predictors in the primary
  nested sequence: built_coverage_2010, growth_1975_2010
  (= built_coverage_2010 - built_coverage_1975), and growth_2010_2015
  (= built_coverage_2015 - built_coverage_2010). Population density
  (WorldPop 2020) is NOT included in the primary sequence: it is
  measured two years AFTER the 2018 AGBH/ANBH epoch, exactly the kind
  of post-outcome leakage Requirement 6 rules out for the growth
  predictors. distance_to_nearest_cbd_m is included as it is a
  time-invariant structural covariate (CBD locations), not a growth
  measure, so it carries no analogous leakage risk.

  Requirement 7 -- nested model sequence (A -> B -> C), OLS then a
  spatial error specification for each, with standardized coefficients,
  SEs, 95% CIs, VIFs, model fit, and three separate Moran's I values
  (outcome itself, OLS residuals, spatial-model residuals) all reported
  rather than only the fullest model.

Weights specification: standard row-standardized Queen contiguity here
(matching the one already validated to work in this environment). The
full Queen / Queen+KNN-for-islands / KNN-8 sensitivity comparison
(Requirement 8) is a separate analysis in script 05, run once, not
duplicated six times across every nested model here.

Sample: the national-footprint sample (Open Buildings V3, building_count
> 0), MINUS cells with no GHSL raster coverage (nodata) or missing
covariates -- both counted and reported explicitly, not silently dropped.

Outputs
-------
    04_outputs/paper2_v3/tables/T_models_ols.csv
    04_outputs/paper2_v3/tables/T_models_spatial.csv
    04_outputs/paper2_v3/tables/T_models_fit_and_moran.csv
    04_outputs/paper2_v3/tables/T_models_vif.csv

Usage
-----
    python scripts/revision_v3/04_models_primary.py
"""
from __future__ import annotations
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg

logging.basicConfig(level=cfg.LOG_LEVEL, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

OUTCOMES = {
    "anbh_primary": {"col": "cell_anbh_derived_m", "label": "log1p(ANBH derived, m) -- PRIMARY vertical outcome"},
    "agbh_robustness": {"col": "cell_agbh_mean_m", "label": "log1p(AGBH mean, m) -- ROBUSTNESS volumetric-density outcome"},
}

# Nested predictor sequence -- pre-2018 only, per Requirement 6.
NESTED_SPECS = {
    "A_cbd_only": ["log_distance_to_nearest_cbd_m"],
    "B_plus_legacy": ["log_distance_to_nearest_cbd_m", "built_coverage_2010"],
    "C_full_pre2018": ["log_distance_to_nearest_cbd_m", "built_coverage_2010",
                        "growth_1975_2010", "growth_2010_2015"],
}


def build_dataset() -> pd.DataFrame:
    samp = pd.read_parquet(cfg.GHSL_LAYERS_V3 / "sample_definitions.parquet",
                            columns=["grid_id", "cell_agbh_mean_m", "cell_anbh_derived_m",
                                     "n_source_pixels", "in_national_footprint"])
    cov = pd.read_parquet(cfg.COVARIATES / "grid_model_ready_500m.parquet",
                           columns=["grid_id", "built_coverage_1975", "built_coverage_2010",
                                    "built_coverage_2015", "distance_to_nearest_cbd_m"])
    df = samp[samp.in_national_footprint].merge(cov, on="grid_id", how="left")
    n0 = len(df)

    n_nodata = int(df.cell_anbh_derived_m.isna().sum())
    df = df[df.cell_anbh_derived_m.notna()].copy()
    n1 = len(df)

    df["growth_1975_2010"] = df["built_coverage_2010"] - df["built_coverage_1975"]
    df["growth_2010_2015"] = df["built_coverage_2015"] - df["built_coverage_2010"]
    df["log_distance_to_nearest_cbd_m"] = np.log1p(df["distance_to_nearest_cbd_m"])

    needed = ["built_coverage_1975", "built_coverage_2010", "built_coverage_2015",
              "distance_to_nearest_cbd_m"]
    n_missing_cov = int(df[needed].isna().any(axis=1).sum())
    df = df.dropna(subset=needed).copy()
    n_final = len(df)

    log.info("Sample construction: national-footprint=%d -> drop %d (no GHSL raster coverage, nodata) "
             "-> drop %d (missing covariates) -> final n=%d",
             n0, n_nodata, n_missing_cov, n_final)
    return df, {"n_national_footprint": n0, "n_dropped_nodata": n_nodata,
                "n_dropped_missing_covariates": n_missing_cov, "n_final": n_final}


def fit_ols(y: pd.Series, X: pd.DataFrame):
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    return model


def standardized_coefs(model, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    sd_y = y.std()
    rows = []
    for name in model.params.index:
        coef, se = model.params[name], model.bse[name]
        if name == "const":
            std_coef, std_se = np.nan, np.nan
        else:
            sd_x = X[name].std()
            std_coef = coef * sd_x / sd_y
            std_se = se * sd_x / sd_y
        ci_lo, ci_hi = coef - 1.96 * se, coef + 1.96 * se
        rows.append({"variable": name, "coefficient": coef, "std_error": se,
                     "ci_95_lo": ci_lo, "ci_95_hi": ci_hi, "p_value": model.pvalues[name],
                     "standardized_coefficient": std_coef, "standardized_se": std_se})
    return pd.DataFrame(rows)


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    Xc = sm.add_constant(X)
    vals = Xc.values
    rows = []
    for i, col in enumerate(Xc.columns):
        if col == "const":
            continue
        v = variance_inflation_factor(vals, i)
        rows.append({"variable": col, "vif": round(v, 3), "flag_high_vif": v > 10})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    cfg.PAPER2_V3_TABLES.mkdir(parents=True, exist_ok=True)
    df, sample_report = build_dataset()

    grid = gpd.read_parquet(cfg.GRIDS / "grid_500m.parquet")[["grid_id", "geometry"]]
    gdf = grid.merge(df, on="grid_id", how="inner")
    assert len(gdf) == len(df)
    log.info("Building Queen contiguity weights on n=%d ...", len(gdf))
    from libpysal import weights as lps
    from esda.moran import Moran
    import spreg

    tW = time.time()
    w = lps.Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)
    w.transform = "R"
    log.info("Weights built in %.1fs (islands=%d)", time.time() - tW, len(w.islands))

    ols_rows, spatial_rows, fit_rows, vif_rows = [], [], [], []

    for outcome_key, spec in OUTCOMES.items():
        y_raw = gdf[spec["col"]]
        y = np.log1p(y_raw)
        y.name = spec["col"]

        # Outcome Moran's I is a property of the outcome, computed once per outcome
        mi_outcome = Moran(y.values, w, permutations=99)

        for model_name, predictors in NESTED_SPECS.items():
            X = gdf[predictors].copy()
            valid = y.notna() & X.notna().all(axis=1)
            y_v, X_v = y[valid], X[valid]

            ols = fit_ols(y_v, X_v)
            coefs = standardized_coefs(ols, X_v, y_v)
            coefs["outcome"] = outcome_key
            coefs["model"] = model_name
            coefs["spec"] = "OLS"
            ols_rows.append(coefs)

            if len(predictors) > 1:
                vif = compute_vif(X_v)
                vif["outcome"] = outcome_key
                vif["model"] = model_name
                vif_rows.append(vif)

            mi_ols_resid = Moran(ols.resid.values, w, permutations=99)

            # Spatial error model
            Xc_arr = X_v.values
            err = spreg.GM_Error(y_v.values.reshape(-1, 1), Xc_arr, w=w,
                                  name_y=spec["col"], name_x=predictors, name_w="queen")
            beta_names = ["const"] + predictors
            betas_flat = err.betas.flatten()
            vm_diag = np.sqrt(np.diag(err.vm))
            n_beta = len(beta_names)
            sp_coef = betas_flat[:n_beta]
            sp_se = vm_diag[:n_beta] if len(vm_diag) >= n_beta else [np.nan] * n_beta
            lam = float(betas_flat[-1]) if len(betas_flat) > n_beta else np.nan

            for name, coef, se in zip(beta_names, sp_coef, sp_se):
                z = coef / se if se and not np.isnan(se) else np.nan
                p = 2 * (1 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan
                spatial_rows.append({
                    "outcome": outcome_key, "model": model_name, "spec": "spatial_error",
                    "variable": name, "coefficient": coef, "std_error": se,
                    "ci_95_lo": coef - 1.96 * se if se else np.nan,
                    "ci_95_hi": coef + 1.96 * se if se else np.nan,
                    "z_statistic": z, "p_value": p,
                })
            spatial_rows.append({
                "outcome": outcome_key, "model": model_name, "spec": "spatial_error",
                "variable": "lambda", "coefficient": lam, "std_error": np.nan,
                "ci_95_lo": np.nan, "ci_95_hi": np.nan, "z_statistic": np.nan, "p_value": np.nan,
            })

            # IMPORTANT: err.u is the PLAIN (unfiltered) residual y - X*beta_GMErr,
            # evaluated on the original scale -- it is NOT the residual the spatial
            # error estimator actually targets. spreg exposes the correct one
            # separately as e_filtered (the spatially-filtered residual, i.e. the
            # Cochrane-Orcutt-style residual after the lambda*W transform). Using
            # .u here was an initial mistake in this script: it produced Moran's I
            # values that did not improve on OLS residuals in any of the six
            # outcome/model combinations, which would have been reported as a
            # genuine "spatial correction doesn't help" finding -- but the correct
            # diagnostic (below) tells a different story. See audit_report.md.
            mi_spatial_resid = Moran(err.e_filtered.flatten(), w, permutations=99)

            fit_rows.append({
                "outcome": outcome_key, "model": model_name, "n": int(valid.sum()),
                "ols_r2": ols.rsquared, "ols_adj_r2": ols.rsquared_adj, "ols_aic": ols.aic,
                "spatial_pseudo_r2": float(err.pr2) if hasattr(err, "pr2") else np.nan,
                "spatial_lambda": lam,
                "moran_i_outcome": mi_outcome.I, "moran_i_outcome_p_sim": mi_outcome.p_sim,
                "moran_i_ols_residuals": mi_ols_resid.I, "moran_i_ols_residuals_p_sim": mi_ols_resid.p_sim,
                "moran_i_spatial_residuals": mi_spatial_resid.I, "moran_i_spatial_residuals_p_sim": mi_spatial_resid.p_sim,
            })
            log.info("[%s / %s] n=%d  OLS R2=%.4f  spatial pseudo-R2=%.4f  lambda=%.4f  "
                     "Moran outcome=%.3f -> OLS resid=%.3f -> spatial resid=%.3f",
                     outcome_key, model_name, int(valid.sum()), ols.rsquared,
                     fit_rows[-1]["spatial_pseudo_r2"], lam,
                     mi_outcome.I, mi_ols_resid.I, mi_spatial_resid.I)

    pd.concat(ols_rows, ignore_index=True).to_csv(cfg.PAPER2_V3_TABLES / "T_models_ols.csv", index=False)
    pd.DataFrame(spatial_rows).to_csv(cfg.PAPER2_V3_TABLES / "T_models_spatial.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(cfg.PAPER2_V3_TABLES / "T_models_fit_and_moran.csv", index=False)
    if vif_rows:
        pd.concat(vif_rows, ignore_index=True).to_csv(cfg.PAPER2_V3_TABLES / "T_models_vif.csv", index=False)

    import json
    with open(cfg.PAPER2_V3_TABLES / "sample_construction_report.json", "w") as f:
        json.dump(sample_report, f, indent=2)

    log.info("All models complete in %.1fs. Tables written to %s", time.time() - t0,
              cfg.PAPER2_V3_TABLES.relative_to(cfg.PROJECT_ROOT))


if __name__ == "__main__":
    main()
