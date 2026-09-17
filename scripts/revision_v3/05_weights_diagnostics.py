"""
Paper 2 v3 -- Script 05: Spatial weights audit and sensitivity (Requirement 8)

Audits the Queen contiguity weights used in script 04 (islands, connected
components), then re-fits the headline specification (Model C, full
pre-2018 predictor set) under three weights specifications:

  1. Queen contiguity, unmodified (as used in script 04).
  2. Queen contiguity with each island attached to its single nearest
     neighbour (libpysal.weights.util.attach_islands), removing islands
     without discarding the underlying contiguity structure elsewhere.
  3. K-nearest-neighbours, k=8 -- a structurally different (non-contiguity)
     definition of "neighbour", used here purely as a sensitivity check on
     whether the substantive conclusions (coefficient signs/magnitudes,
     lambda, residual Moran's I) depend on the specific weights choice.

Uses the SAME sample (n=296,668, built in script 04's build_dataset) for
a like-for-like comparison across weights specifications.

Outputs
-------
    04_outputs/paper2_v3/tables/T_weights_audit.csv
        Per specification: n islands, n components, min/mean/max neighbours.
    04_outputs/paper2_v3/tables/T_weights_sensitivity.csv
        Per specification: Model C spatial-error coefficients, lambda,
        pseudo-R2, and Moran's I (outcome / OLS residual / spatial-filtered
        residual, all recomputed under that specification's own W).

Usage
-----
    python scripts/revision_v3/05_weights_diagnostics.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg

logging.basicConfig(level=cfg.LOG_LEVEL, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

PREDICTORS_C = ["log_distance_to_nearest_cbd_m", "built_coverage_2010",
                "growth_1975_2010", "growth_2010_2015"]


def build_dataset() -> pd.DataFrame:
    """Identical construction to script 04's build_dataset, kept in sync
    deliberately (not imported, to keep this script runnable standalone;
    see test_identity_range.py-style guard below for a consistency check)."""
    samp = pd.read_parquet(cfg.GHSL_LAYERS_V3 / "sample_definitions.parquet",
                            columns=["grid_id", "cell_agbh_mean_m", "cell_anbh_derived_m",
                                     "in_national_footprint"])
    cov = pd.read_parquet(cfg.COVARIATES / "grid_model_ready_500m.parquet",
                           columns=["grid_id", "built_coverage_1975", "built_coverage_2010",
                                    "built_coverage_2015", "distance_to_nearest_cbd_m"])
    df = samp[samp.in_national_footprint].merge(cov, on="grid_id", how="left")
    df = df[df.cell_anbh_derived_m.notna()].copy()
    df["growth_1975_2010"] = df["built_coverage_2010"] - df["built_coverage_1975"]
    df["growth_2010_2015"] = df["built_coverage_2015"] - df["built_coverage_2010"]
    df["log_distance_to_nearest_cbd_m"] = np.log1p(df["distance_to_nearest_cbd_m"])
    needed = ["built_coverage_1975", "built_coverage_2010", "built_coverage_2015",
              "distance_to_nearest_cbd_m"]
    df = df.dropna(subset=needed).copy()
    return df


def audit_weights(name: str, w) -> dict:
    return {
        "spec": name,
        "n_units": w.n,
        "n_islands": len(w.islands),
        "n_components": w.n_components,
        "min_neighbors": w.min_neighbors,
        "max_neighbors": w.max_neighbors,
        "mean_neighbors": round(w.mean_neighbors, 3),
    }


def main() -> None:
    t0 = time.time()
    cfg.PAPER2_V3_TABLES.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    log.info("Sample: n=%d (matches script 04's final n; consistency asserted below)", len(df))
    assert len(df) == 296_668, (
        f"Sample size {len(df)} does not match script 04's n=296,668 -- "
        "build_dataset() has drifted out of sync between scripts 04 and 05."
    )

    grid = gpd.read_parquet(cfg.GRIDS / "grid_500m.parquet")[["grid_id", "geometry"]]
    gdf = grid.merge(df, on="grid_id", how="inner").reset_index(drop=True)
    assert len(gdf) == len(df)

    from libpysal import weights as lps
    from libpysal.weights.util import attach_islands
    from esda.moran import Moran
    import spreg

    # ---------------------------------------------------- 1. Queen (base) -
    log.info("Building Queen contiguity weights ...")
    tW = time.time()
    w_queen = lps.Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)
    log.info("  done in %.1fs", time.time() - tW)

    # ----------------------------------------- 2. Queen + KNN-1 for islands
    log.info("Attaching nearest neighbour to %d Queen islands ...", len(w_queen.islands))
    tW = time.time()
    w_knn1 = lps.KNN.from_dataframe(gdf, k=1, use_index=False, silence_warnings=True)
    w_queen_fixed = attach_islands(w_queen, w_knn1)
    log.info("  done in %.1fs (islands remaining: %d)", time.time() - tW, len(w_queen_fixed.islands))

    # -------------------------------------------------------- 3. KNN-8 ----
    log.info("Building KNN-8 weights ...")
    tW = time.time()
    w_knn8 = lps.KNN.from_dataframe(gdf, k=8, use_index=False, silence_warnings=True)
    log.info("  done in %.1fs", time.time() - tW)

    specs = {"queen": w_queen, "queen_knn1_islands": w_queen_fixed, "knn8": w_knn8}

    audit_rows = [audit_weights(name, w) for name, w in specs.items()]
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(cfg.PAPER2_V3_TABLES / "T_weights_audit.csv", index=False)
    log.info("Weights audit:\n%s", audit_df.to_string(index=False))

    # ----------------------------------- Re-fit Model C under each spec ---
    y = np.log1p(gdf["cell_anbh_derived_m"])
    X = gdf[PREDICTORS_C]
    Xc = sm.add_constant(X)
    ols = sm.OLS(y, Xc).fit()

    sens_rows = []
    for name, w in specs.items():
        w.transform = "R"
        mi_outcome = Moran(y.values, w, permutations=99)
        mi_ols = Moran(ols.resid.values, w, permutations=99)

        err = spreg.GM_Error(y.values.reshape(-1, 1), X.values, w=w,
                              name_y="log1p_anbh", name_x=list(X.columns), name_w=name)
        beta_names = ["const"] + list(X.columns)
        betas_flat = err.betas.flatten()
        lam = float(betas_flat[-1]) if len(betas_flat) > len(beta_names) else np.nan
        mi_spatial = Moran(err.e_filtered.flatten(), w, permutations=99)

        for bname, bval in zip(beta_names, betas_flat[: len(beta_names)]):
            sens_rows.append({
                "weights_spec": name, "variable": bname, "coefficient": bval,
                "lambda": lam, "spatial_pseudo_r2": float(err.pr2) if hasattr(err, "pr2") else np.nan,
                "moran_i_outcome": mi_outcome.I, "moran_i_ols_residuals": mi_ols.I,
                "moran_i_spatial_residuals": mi_spatial.I,
            })
        log.info("[%s] lambda=%.4f  pseudo-R2=%.4f  Moran outcome=%.3f -> OLS resid=%.3f -> "
                 "spatial resid=%.4f", name, lam, sens_rows[-1]["spatial_pseudo_r2"],
                 mi_outcome.I, mi_ols.I, mi_spatial.I)

    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(cfg.PAPER2_V3_TABLES / "T_weights_sensitivity.csv", index=False)
    log.info("Saved weights sensitivity table. Total time %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
