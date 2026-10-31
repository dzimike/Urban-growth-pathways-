"""
Paper 2 v3 -- Script 12: Checks for the second-round review (Reviewer #4)

Reviewer #4 supported acceptance but asked for four qualifications. Each
section below produces the evidence for one of them.

  A. Pixel statistics over Ghana itself (point 2). The 40.1% concentration
     statistic and the AGBH <= ANBH check in build_facts_json.py are computed
     over the rectangular acquisition envelope (the Ghana bounding box), which
     includes parts of neighbouring countries and the Gulf of Guinea. Here the
     same statistics are recomputed with a Ghana boundary mask (GADM level 0,
     rasterised onto the 100 m EPSG:4326 pixel grid of the GHSL clips; a pixel
     is "in Ghana" if its centre falls inside the boundary).

  B. Cells without a GHSL height signal (point 1). 43,412 footprint cells
     have ANBH = 0. Model C (OLS and spatial error) and the median-split
     typology are refitted on the cells with ANBH > 0 only.

  C. Uncertainty of the OLS estimates (point 3). Table 3 / Fig. 5 show OLS
     with conventional standard errors. Here Model C is refitted with
     heteroskedasticity-robust (HC1) and spatially clustered standard errors
     (10 km and 25 km square blocks), and set beside the spatial error model,
     so the effect of residual spatial dependence on the intervals is visible.

  D. Chance baselines for the typology's diagonal shares (point 4). A
     two-class split has an independence baseline near 50%, a three-class
     split near 33%, a four-class split near 25%, so raw diagonal shares are
     not comparable across schemes. Here each scheme's expected diagonal under
     independence (from its own margins) and Cohen's kappa are computed, for
     ANBH and for the AGBH-matched comparison.

Outputs
-------
    04_outputs/paper2_v3/tables/T_round2_pixel_mask.csv
    04_outputs/paper2_v3/tables/T_round2_zero_signal.csv
    04_outputs/paper2_v3/tables/T_round2_inference.csv
    04_outputs/paper2_v3/tables/T_round2_typology_agreement.csv

Usage
-----
    python scripts/revision_v3/12_round2_checks.py
"""
from __future__ import annotations
import importlib.util
import logging
import re
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


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s02 = _load("s02", "02_process_agbh_anbh_grid.py")
s09 = _load("s09", "09_supplementary_tables.py")
gdal_run, gdalinfo_dict, read_envi_array = s02.gdal_run, s02.gdalinfo_dict, s02.read_envi_array

T = cfg.PAPER2_V3_TABLES
WORK = cfg.GHSL_LAYERS_V3 / "_utm_intermediate" / "round2"
PREDICTORS = ["log_distance_to_nearest_cbd_m", "built_coverage_2010",
              "growth_1975_2010", "growth_2010_2015"]


# ------------------------------------------------------------------ A -------
def pixel_mask_check() -> pd.DataFrame:
    WORK.mkdir(parents=True, exist_ok=True)
    anbh_tif, agbh_tif = cfg.GHSL_RAW / "anbh_2018_v3.tif", cfg.GHSL_RAW / "agbh_2018_v3.tif"
    meta = gdalinfo_dict(anbh_tif)
    W, H = meta["width"], meta["height"]
    info = gdal_run(["gdalinfo", str(anbh_tif)])
    def _xy(key):
        line = next(l for l in info.splitlines() if key in l)
        nums = re.findall(r"[-\d.]+", line.split("(", 1)[1].split(")")[0])
        return float(nums[0]), float(nums[1])
    (ulx, uly), (lrx, lry) = _xy("Upper Left"), _xy("Lower Right")

    mask_tif = WORK / "ghana_mask_4326.tif"
    if not mask_tif.exists():
        gpkg = cfg.BOUNDARIES_RAW / "gadm41_GHA.gpkg"
        gdal_run(["gdal_rasterize", "-l", "ADM_ADM_0", "-burn", "1", "-init", "0", "-ot", "Byte",
                  "-te", str(ulx), str(lry), str(lrx), str(uly), "-ts", str(W), str(H),
                  str(gpkg), str(mask_tif)])
    ghana = read_envi_array(mask_tif, W, H, "int32").ravel() == 1
    anbh = read_envi_array(anbh_tif, W, H, "float32").ravel()
    agbh = read_envi_array(agbh_tif, W, H, "float32").ravel()

    rows = []
    for label, m in [("Acquisition envelope (Ghana bounding box)", np.ones_like(ghana)),
                     ("Ghana boundary mask (GADM level 0)", ghana)]:
        a, g = anbh[m], agbh[m]
        built = a > 0
        # Same binning as the published statistic (build_facts_json.py):
        # np.histogram over 0.05 m bins from 0 to 10 m, modal bin.
        hist, edges = np.histogram(a[built], bins=np.arange(0, 10, 0.05))
        k = int(np.argmax(hist))
        n_spike = int(hist[k])
        viol = g > a
        rows.append({
            "domain": label, "n_pixels": int(m.sum()), "n_built_pixels_anbh_gt0": int(built.sum()),
            "modal_bin_m": f"{edges[k]:.2f}-{edges[k + 1]:.2f}",
            "n_built_in_modal_bin": n_spike,
            "pct_built_in_modal_bin": 100 * n_spike / built.sum(),
            "n_zero_both": int(((a == 0) & (g == 0)).sum()),
            "n_zero_mismatch": int(((a == 0) != (g == 0)).sum()),
            "n_agbh_gt_anbh": int(viol.sum()),
            "pct_agbh_le_anbh": 100 * (1 - viol.mean()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(T / "T_round2_pixel_mask.csv", index=False)
    log.info("A. Pixel statistics, envelope vs Ghana mask:\n%s", out.to_string(index=False))
    return out


# ------------------------------------------------------------- shared -------
def model_data() -> gpd.GeoDataFrame:
    df = s09.build_final_sample()
    df["log_distance_to_nearest_cbd_m"] = np.log1p(df.distance_to_nearest_cbd_m)
    grid = gpd.read_parquet(cfg.GRIDS / "grid_500m.parquet")[["grid_id", "geometry"]]
    gdf = grid.merge(df, on="grid_id", how="inner").to_crs(cfg.PROJ_CRS)
    assert len(gdf) == 296_668
    c = gdf.geometry.centroid
    gdf["block10"] = (np.floor(c.x / 10_000).astype(int).astype(str) + "_" +
                      np.floor(c.y / 10_000).astype(int).astype(str))
    gdf["block25"] = (np.floor(c.x / 25_000).astype(int).astype(str) + "_" +
                      np.floor(c.y / 25_000).astype(int).astype(str))
    return gdf.reset_index(drop=True)


def fit_ols_sem(gdf: gpd.GeoDataFrame, label: str) -> list[dict]:
    from libpysal import weights as lps
    from esda.moran import Moran
    import spreg
    y = np.log1p(gdf.cell_anbh_derived_m)
    X = gdf[PREDICTORS]
    ols = sm.OLS(y, sm.add_constant(X)).fit()
    w = lps.Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)
    w.transform = "R"
    sem = spreg.GM_Error(y.values.reshape(-1, 1), X.values, w=w, name_x=PREDICTORS)
    rows = []
    sd_y = y.std()
    sem_b = sem.betas.flatten()
    sem_se = np.sqrt(np.diag(sem.vm))
    for i, v in enumerate(PREDICTORS):
        rows.append({"sample": label, "n": len(gdf), "variable": v,
                     "ols_coef": ols.params[v], "ols_se": ols.bse[v],
                     "ols_standardized": ols.params[v] * X[v].std() / sd_y,
                     "sem_coef": sem_b[i + 1], "sem_se": sem_se[i + 1],
                     "ols_r2": ols.rsquared, "sem_pseudo_r2": float(sem.pr2),
                     "sem_lambda": float(sem_b[-1]),
                     "moran_i_ols_resid": Moran(ols.resid.values, w, permutations=0).I,
                     "moran_i_sem_filtered": Moran(sem.e_filtered.flatten(), w, permutations=0).I})
    return rows


# ------------------------------------------------------------------ B -------
def zero_signal_check(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    zero = gdf.cell_anbh_derived_m == 0
    comp = gdf.assign(zero=zero).groupby("typology_class").zero.agg(["size", "sum", "mean"])
    log.info("B. Zero-signal cells by typology class:\n%s", comp.to_string())
    rows = fit_ols_sem(gdf, "All cells (as published)")
    sig = gdf[~zero].reset_index(drop=True)
    rows += fit_ols_sem(sig, "Cells with a GHSL height signal (ANBH > 0)")
    out = pd.DataFrame(rows)
    # Median-split typology on signal-only cells (same rule: ties to low class)
    v, h = sig.cell_anbh_derived_m, sig.growth_1975_2015
    vh, hh = v > v.median(), h > h.median()
    diag = ((vh & hh) | (~vh & ~hh)).mean()
    cls = (pd.Series(np.where(vh, "High height", "Low height")) + " / " +
           pd.Series(np.where(hh, "High prior expansion", "Low prior expansion"))).values
    med = pd.Series(v.values).groupby(cls).median()
    out.attrs["typology_signal_only"] = {"n": len(sig), "anbh_median": v.median(), "diagonal_pct": 100 * diag,
                                         "class_median_min": med.min(), "class_median_max": med.max()}
    extra = pd.DataFrame([
        {"sample": "Composition", "variable": f"zero-signal cells in {k}", "n": int(r["size"]),
         "ols_coef": int(r["sum"]), "ols_se": 100 * r["mean"]} for k, r in comp.iterrows()
    ] + [{"sample": "Typology on ANBH > 0 cells", "variable": k, "ols_coef": val}
         for k, val in out.attrs["typology_signal_only"].items()])
    full = pd.concat([out, extra], ignore_index=True)
    full.to_csv(T / "T_round2_zero_signal.csv", index=False)
    log.info("B. Model C, all cells vs signal-only:\n%s", out.round(4).to_string(index=False))
    log.info("B. Typology on signal-only cells: %s", out.attrs["typology_signal_only"])
    return full


# ------------------------------------------------------------------ C -------
def inference_check(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    y = np.log1p(gdf.cell_anbh_derived_m)
    X = sm.add_constant(gdf[PREDICTORS])
    fits = {
        "OLS, conventional SE (Table 3, Fig. 5)": sm.OLS(y, X).fit(),
        "OLS, HC1 robust SE": sm.OLS(y, X).fit(cov_type="HC1"),
        "OLS, SE clustered by 10 km blocks": sm.OLS(y, X).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(gdf.block10)[0]}),
        "OLS, SE clustered by 25 km blocks": sm.OLS(y, X).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(gdf.block25)[0]}),
    }
    rows = []
    for name, f in fits.items():
        for v in PREDICTORS:
            rows.append({"estimator": name, "variable": v, "coef": f.params[v], "se": f.bse[v],
                         "ci_lo": f.params[v] - 1.96 * f.bse[v], "ci_hi": f.params[v] + 1.96 * f.bse[v],
                         "se_ratio_to_conventional": f.bse[v] / fits["OLS, conventional SE (Table 3, Fig. 5)"].bse[v]})
    sem = pd.read_csv(T / "T_models_spatial.csv")
    sem = sem[(sem.outcome == "anbh_primary") & (sem.model == "C_full_pre2018") & sem.variable.isin(PREDICTORS)]
    conv = {r["variable"]: r["se"] for r in rows if r["estimator"].startswith("OLS, conventional")}
    for _, r in sem.iterrows():
        rows.append({"estimator": "Spatial error model (GM_Error, Queen)", "variable": r.variable,
                     "coef": r.coefficient, "se": r.std_error, "ci_lo": r.ci_95_lo, "ci_hi": r.ci_95_hi,
                     "se_ratio_to_conventional": r.std_error / conv[r.variable]})
    out = pd.DataFrame(rows)
    out.attrs["n_clusters"] = {"10km": int(gdf.block10.nunique()), "25km": int(gdf.block25.nunique())}
    out.to_csv(T / "T_round2_inference.csv", index=False)
    log.info("C. Clusters: %s", out.attrs["n_clusters"])
    log.info("C. Model C uncertainty by estimator:\n%s", out.round(4).to_string(index=False))
    return out


# ------------------------------------------------------------------ D -------
def agreement(vert: pd.Series, horz: pd.Series, scheme: str) -> dict:
    if scheme == "median (2x2)":
        vc = (vert > vert.median()).astype(int).values      # ties -> low, as in Section 4.9
        hc = (horz > horz.median()).astype(int).values
    else:
        qs = [1 / 3, 2 / 3] if scheme.startswith("tertile") else [0.25, 0.5, 0.75]
        vc = np.digitize(vert, vert.quantile(qs).values)    # same rule as script 06
        hc = np.digitize(horz, horz.quantile(qs).values)
    k = int(max(vc.max(), hc.max())) + 1
    ct = np.zeros((k, k))
    np.add.at(ct, (vc, hc), 1)
    p = ct / ct.sum()
    po = np.trace(p)
    pe = float((p.sum(axis=1) * p.sum(axis=0)).sum())
    return {"scheme": scheme, "observed_diagonal_pct": 100 * po,
            "expected_diagonal_under_independence_pct": 100 * pe,
            "excess_over_chance_pct_points": 100 * (po - pe), "cohens_kappa": (po - pe) / (1 - pe)}


def typology_agreement(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []
    for vert_name, col in [("ANBH (primary)", "cell_anbh_derived_m"), ("AGBH (matched comparison)", "cell_agbh_mean_m")]:
        for scheme in ["median (2x2)", "tertile (3x3)", "quartile (4x4)"]:
            rows.append({"vertical_axis": vert_name, **agreement(gdf[col], gdf.growth_1975_2015, scheme)})
    out = pd.DataFrame(rows)
    out.to_csv(T / "T_round2_typology_agreement.csv", index=False)
    log.info("D. Diagonal shares against their chance baselines:\n%s", out.round(3).to_string(index=False))
    return out


def main() -> None:
    t0 = time.time()
    pixel_mask_check()
    gdf = model_data()
    typology_agreement(gdf)
    inference_check(gdf)
    zero_signal_check(gdf)
    log.info("Round-2 checks complete in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
