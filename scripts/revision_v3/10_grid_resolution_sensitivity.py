"""
Paper 2 v3 -- Script 10: Grid-resolution sensitivity (250 m / 500 m / 1 km)

Repeats the core analysis at the two alternative resolutions the project
design specifies (CLAUDE.md Section 4, Level 2: 250 m for intra-urban
detail, 1 km for national comparison and sensitivity testing), which
reviewers requested and which earlier revisions did not report.

One code path, three resolutions
--------------------------------
All three resolutions -- including 500 m -- are recomputed here by the SAME
code, so that any difference between rows reflects resolution, not a
difference in processing. The 500 m row then doubles as a reproduction
check against the published 500 m results (scripts 02, 04, 06): the ANBH/
AGBH aggregation is identical by construction (same UTM pixel lattice, same
zone-rasterisation rule), while built coverage is recomputed on the UTM
lattice rather than taken from the upstream 500 m covariate file (which was
aggregated in EPSG:4326 by scripts/06_process_ghsl.py). The agreement
between the two is logged and written to the output table.

Per resolution:
  1. Grid: the project's own grids -- grid_model_ready_{250m,1km}.parquet
     and grid_model_ready_500m.parquet -- restricted to building_count > 0
     (the national-footprint rule, Section 4.2), in EPSG:32630.
  2. Aggregation: grid polygons rasterised onto the 100 m UTM lattice of the
     AGBH/ANBH rasters (script 02's method); cell AGBH and built fraction are
     pixel means, cell ANBH = cell AGBH / cell built fraction. At 250 m a
     100 m pixel is assigned whole to the cell containing its centre, so a
     250 m cell averages 4-9 pixels (6.25 on average); this is a coarser
     approximation than at 500 m (25 pixels) or 1 km (100 pixels) and is
     stated in the manuscript.
  3. Built coverage 1975/2010/2015: GHSL BUILT-S (m2 per ~100 m pixel),
     nearest-neighbour warped onto the same UTM lattice, converted to a
     fraction (value / 10,000 m2, clipped to [0, 1]) and averaged per cell.
  4. CBD distance: cell centroid to the nearest of the 10 CBD points used
     by scripts/07_add_population_roads_pois.py (same coordinates).
  5. Model C (Section 4.7): OLS and spatial error (Queen, row-standardised),
     log(1+ANBH) on the four pre-2018 predictors; Moran's I (point estimate)
     on OLS and spatially filtered residuals.
  6. Median-split typology (Section 4.9): diagonal share, class median ANBH.

Outputs
-------
    04_outputs/paper2_v3/tables/T_grid_resolution_sensitivity.csv
    04_outputs/paper2_v3/tables/T_grid_resolution_coefficients.csv

Usage
-----
    python scripts/revision_v3/10_grid_resolution_sensitivity.py
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

# Reuse script 02's GDAL-CLI helpers (no GDAL Python bindings in this env).
_spec = importlib.util.spec_from_file_location(
    "s02", Path(__file__).with_name("02_process_agbh_anbh_grid.py"))
s02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s02)
gdal_run, gdalinfo_dict, read_envi_array = s02.gdal_run, s02.gdalinfo_dict, s02.read_envi_array

INTERMEDIATE = cfg.GHSL_LAYERS_V3 / "_utm_intermediate"
RES_DIR = INTERMEDIATE / "resolution_sensitivity"
EPOCHS = [1975, 2010, 2015]
PIXEL_AREA_M2 = 10_000.0

# Same CBD points as scripts/07_add_population_roads_pois.py (lon, lat).
CBD_COORDS = {
    "Accra": (-0.187, 5.556), "Kumasi": (-1.624, 6.688), "Tamale": (-0.839, 9.401),
    "Sekondi-Takoradi": (-1.714, 4.934), "Cape Coast": (-1.242, 5.105),
    "Koforidua": (-0.261, 6.088), "Sunyani": (-2.329, 7.340), "Ho": (0.472, 6.601),
    "Wa": (-2.503, 10.060), "Bolgatanga": (-0.851, 10.785),
}

GRID_FILES = {
    250: cfg.COVARIATES / "grid_model_ready_250m.parquet",
    500: cfg.COVARIATES / "grid_model_ready_500m.parquet",
    1000: cfg.COVARIATES / "grid_model_ready_1km.parquet",
}

PREDICTORS = ["log_distance_to_nearest_cbd_m", "built_coverage_2010",
              "growth_1975_2010", "growth_2010_2015"]


def lattice_extent(tif: Path) -> tuple[float, float, float, float]:
    info = gdal_run(["gdalinfo", str(tif)])
    def _xy(key):
        line = next(l for l in info.splitlines() if key in l)
        nums = re.findall(r"[-\d.]+", line.split("(", 1)[1].split(")")[0])
        return float(nums[0]), float(nums[1])
    (ulx, uly), (lrx, lry) = _xy("Upper Left"), _xy("Lower Right")
    return ulx, lry, lrx, uly


def warp_built_surface(te) -> dict[int, Path]:
    out = {}
    for yr in EPOCHS:
        dst = INTERMEDIATE / f"built_surface_{yr}_utm.tif"
        if not dst.exists():
            log.info("Warping GHSL BUILT-S %d onto the UTM height lattice (nearest) ...", yr)
            gdal_run(["gdalwarp", "-t_srs", cfg.PROJ_CRS, "-r", "near", "-tr", "100", "100",
                      "-te", *[str(v) for v in te], "-ot", "Float32", "-overwrite",
                      str(cfg.GHSL_RAW / f"built_surface_{yr}.tif"), str(dst)])
        out[yr] = dst
    return out


def cbd_distance(centroids: gpd.GeoSeries) -> np.ndarray:
    cbd = gpd.GeoSeries(gpd.points_from_xy([c[0] for c in CBD_COORDS.values()],
                                           [c[1] for c in CBD_COORDS.values()]),
                        crs=cfg.GEO_CRS).to_crs(cfg.PROJ_CRS)
    cxy = np.column_stack([cbd.x, cbd.y])
    pxy = np.column_stack([centroids.x, centroids.y])
    d = np.sqrt(((pxy[:, None, :] - cxy[None, :, :]) ** 2).sum(axis=2))
    return d.min(axis=1)


def aggregate(res: int, te, W: int, H: int, rasters: dict) -> pd.DataFrame:
    g = gpd.read_parquet(GRID_FILES[res], columns=["grid_id", "building_count", "geometry"])
    g = g[g.building_count > 0].to_crs(cfg.PROJ_CRS).reset_index(drop=True)
    g["idx"] = np.arange(len(g), dtype="int64")
    n = len(g)
    log.info("[%d m] national-footprint cells: %d", res, n)

    RES_DIR.mkdir(parents=True, exist_ok=True)
    gpkg = RES_DIR / f"grid_{res}m_footprint_utm.gpkg"
    zone_tif = RES_DIR / f"zone_{res}m.tif"
    if not zone_tif.exists():
        g[["idx", "geometry"]].to_file(gpkg, layer="grid", driver="GPKG")
        gdal_run(["gdal_rasterize", "-a", "idx", "-te", *[str(v) for v in te],
                  "-tr", "100", "100", "-ot", "Int32", "-a_nodata", "-1", "-init", "-1",
                  str(gpkg), str(zone_tif)])
    zone = read_envi_array(zone_tif, W, H, "int32").ravel()
    m = zone >= 0
    z = zone[m]

    n_pix = np.bincount(z, minlength=n)
    def cell_mean(arr):
        s = np.bincount(z, weights=arr[m], minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n_pix > 0, s / np.maximum(n_pix, 1), np.nan)

    agbh = cell_mean(rasters["agbh"])
    bf = cell_mean(rasters["bf"])
    with np.errstate(invalid="ignore", divide="ignore"):
        anbh = np.where(bf > 0, agbh / np.maximum(bf, 1e-12), 0.0)
    anbh = np.where(n_pix > 0, anbh, np.nan)

    df = pd.DataFrame({"grid_id": g.grid_id, "n_pix": n_pix, "agbh": agbh, "bf": bf, "anbh": anbh})
    for yr in EPOCHS:
        df[f"built_coverage_{yr}"] = np.where(n_pix > 0, cell_mean(rasters[f"cov{yr}"]), np.nan)
    df["distance_to_nearest_cbd_m"] = cbd_distance(g.geometry.centroid)
    df["geometry"] = g.geometry.values
    return gpd.GeoDataFrame(df, geometry="geometry", crs=cfg.PROJ_CRS)


def analyse(res: int, gdf: gpd.GeoDataFrame) -> tuple[dict, pd.DataFrame]:
    n_fp = len(gdf)
    gdf = gdf[gdf.anbh.notna()]
    n_nodata = n_fp - len(gdf)
    gdf = gdf.dropna(subset=[f"built_coverage_{y}" for y in EPOCHS]).copy()
    gdf["growth_1975_2010"] = gdf.built_coverage_2010 - gdf.built_coverage_1975
    gdf["growth_2010_2015"] = gdf.built_coverage_2015 - gdf.built_coverage_2010
    gdf["growth_1975_2015"] = gdf.built_coverage_2015 - gdf.built_coverage_1975
    gdf["log_distance_to_nearest_cbd_m"] = np.log1p(gdf.distance_to_nearest_cbd_m)
    gdf = gdf.reset_index(drop=True)
    n = len(gdf)

    y = np.log1p(gdf.anbh)
    X = gdf[PREDICTORS]
    ols = sm.OLS(y, sm.add_constant(X)).fit()

    from libpysal import weights as lps
    from esda.moran import Moran
    import spreg
    t = time.time()
    w = lps.Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)
    w.transform = "R"
    log.info("[%d m] Queen weights on n=%d in %.1fs (islands=%d)", res, n, time.time() - t, len(w.islands))
    err = spreg.GM_Error(y.values.reshape(-1, 1), X.values, w=w,
                         name_y="log1p_anbh", name_x=PREDICTORS, name_w="queen")
    mi_ols = Moran(ols.resid.values, w, permutations=0).I
    mi_filt = Moran(err.e_filtered.flatten(), w, permutations=0).I

    v, h = gdf.anbh, gdf.growth_1975_2015
    v_high, h_high = v > v.median(), h > h.median()     # ties -> low (Section 4.9)
    cls = np.where(v_high, "High height", "Low height") + " / " + \
        np.where(h_high, "High prior expansion", "Low prior expansion")
    med_by_class = pd.Series(v.values).groupby(cls).median()

    row = {
        "resolution_m": res, "n_national_footprint": n_fp, "n_dropped_nodata": n_nodata,
        "n_dropped_missing_covariates": n_fp - n_nodata - n, "n_final": n,
        "mean_pixels_per_cell": float(gdf.n_pix.mean()),
        "anbh_mean_m": v.mean(), "anbh_sd_m": v.std(), "anbh_p25_m": v.quantile(.25),
        "anbh_median_m": v.median(), "anbh_p75_m": v.quantile(.75), "anbh_p99_m": v.quantile(.99),
        "pct_anbh_zero": 100 * (v == 0).mean(),
        "agbh_mean_m": gdf.agbh.mean(), "agbh_median_m": gdf.agbh.median(),
        "ols_r2": ols.rsquared, "spatial_pseudo_r2": float(err.pr2),
        "spatial_lambda": float(err.betas.flatten()[-1]),
        "moran_i_ols_residuals": mi_ols, "moran_i_spatial_filtered_residuals": mi_filt,
        "n_islands_queen": len(w.islands),
        "typology_diagonal_pct": 100 * ((v_high & h_high) | (~v_high & ~h_high)).mean(),
        "typology_class_median_anbh_min_m": med_by_class.min(),
        "typology_class_median_anbh_max_m": med_by_class.max(),
    }
    for cls_name, med in med_by_class.items():
        key = cls_name.lower().replace(" / ", "_").replace(" ", "_")
        row[f"median_anbh_m__{key}"] = med
        row[f"pct_cells__{key}"] = 100 * float((cls == cls_name).mean())
    sd_y = y.std()
    coefs = pd.DataFrame({
        "resolution_m": res, "variable": PREDICTORS,
        "coefficient": ols.params[PREDICTORS].values,
        "std_error": ols.bse[PREDICTORS].values,
        "standardized_coefficient": [ols.params[p] * X[p].std() / sd_y for p in PREDICTORS],
    })
    log.info("[%d m] n=%d  OLS R2=%.4f  pseudo-R2=%.4f  lambda=%.3f  MI ols=%.3f filt=%.3f  "
             "diag=%.1f%%  class-median ANBH %.2f-%.2f m",
             res, n, row["ols_r2"], row["spatial_pseudo_r2"], row["spatial_lambda"], mi_ols, mi_filt,
             row["typology_diagonal_pct"], row["typology_class_median_anbh_min_m"],
             row["typology_class_median_anbh_max_m"])
    return row, coefs, gdf


def reproduction_check(gdf500: gpd.GeoDataFrame) -> dict:
    """Compare the 500 m row of this script with the published 500 m inputs."""
    pub = pd.read_parquet(cfg.GHSL_LAYERS_V3 / "sample_definitions.parquet",
                          columns=["grid_id", "cell_anbh_derived_m", "cell_agbh_mean_m"])
    up = pd.read_parquet(cfg.COVARIATES / "grid_model_ready_500m.parquet",
                         columns=["grid_id", "built_coverage_2010", "built_coverage_1975",
                                  "built_coverage_2015", "distance_to_nearest_cbd_m"])
    m = gdf500[["grid_id", "anbh", "agbh", "built_coverage_1975", "built_coverage_2010",
                "built_coverage_2015", "distance_to_nearest_cbd_m"]].merge(
        pub, on="grid_id").merge(up, on="grid_id", suffixes=("", "_upstream"))
    out = {
        "n_matched": len(m),
        "anbh_max_abs_diff_m": float((m.anbh - m.cell_anbh_derived_m).abs().max()),
        "agbh_max_abs_diff_m": float((m.agbh - m.cell_agbh_mean_m).abs().max()),
    }
    for yr in EPOCHS:
        a, b = m[f"built_coverage_{yr}"], m[f"built_coverage_{yr}_upstream"]
        ok = a.notna() & b.notna()
        out[f"built_coverage_{yr}_pearson_r_vs_upstream"] = float(np.corrcoef(a[ok], b[ok])[0, 1])
        out[f"built_coverage_{yr}_mean_abs_diff_vs_upstream"] = float((a[ok] - b[ok]).abs().mean())
    out["cbd_distance_max_abs_diff_m"] = float(
        (m.distance_to_nearest_cbd_m - m.distance_to_nearest_cbd_m_upstream).abs().max())
    log.info("500 m reproduction check vs published inputs: %s", out)
    return out


def main() -> None:
    t0 = time.time()
    agbh_utm = INTERMEDIATE / "agbh_utm.tif"
    bf_tif = INTERMEDIATE / "built_fraction_px.tif"
    assert agbh_utm.exists() and bf_tif.exists(), "Run script 02 first."
    meta = gdalinfo_dict(agbh_utm)
    W, H = meta["width"], meta["height"]
    te = lattice_extent(agbh_utm)
    cov_tifs = warp_built_surface(te)

    rasters = {"agbh": read_envi_array(agbh_utm, W, H, "float32").ravel(),
               "bf": read_envi_array(bf_tif, W, H, "float32").ravel()}
    for yr, p in cov_tifs.items():
        meta_c = gdalinfo_dict(p)
        assert (meta_c["width"], meta_c["height"]) == (W, H), f"BUILT-S {yr} not on the height lattice"
        a = read_envi_array(p, W, H, "float32").ravel()
        a = np.where(np.isfinite(a) & (a >= 0), a, 0.0)
        rasters[f"cov{yr}"] = np.clip(a / PIXEL_AREA_M2, 0, 1).astype("float32")

    rows, coef_tables = [], []
    for res in (250, 500, 1000):
        gdf = aggregate(res, te, W, H, rasters)
        row, coefs, final = analyse(res, gdf)
        if res == 500:
            row.update({f"repro_{k}": v for k, v in reproduction_check(final).items()})
        rows.append(row)
        coef_tables.append(coefs)

    pd.DataFrame(rows).to_csv(cfg.PAPER2_V3_TABLES / "T_grid_resolution_sensitivity.csv", index=False)
    pd.concat(coef_tables, ignore_index=True).to_csv(
        cfg.PAPER2_V3_TABLES / "T_grid_resolution_coefficients.csv", index=False)
    log.info("Grid-resolution sensitivity complete in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
