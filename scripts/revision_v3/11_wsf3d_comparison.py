"""
Paper 2 v3 -- Script 11: Cross-product comparison with WSF3D building height

Compares GHSL ANBH/AGBH (2018) with an independently produced building-height
product, the World Settlement Footprint 3D (WSF3D V02; Esch et al., 2022,
Remote Sensing of Environment 270, 112877; DLR; CC-BY-4.0).

What this is and is not
-----------------------
WSF3D is NOT ground truth. It is a second satellite-derived height estimate,
made from different inputs: building height from the TanDEM-X DEM (radar,
acquired 2011-2013) within the WSF-Imperviousness settlement mask (2017-2019),
whereas GHSL BUILT-H 2018 is derived from AW3D30, SRTM30 and Sentinel-2. Agreement between
the two is cross-product consistency evidence; disagreement cannot by
itself say which product is wrong. Buildings constructed after ~2012 have no
TanDEM-X height (DLR temporal-extent note), so WSF3D is expected to miss
recent construction that GHSL 2018 can see.

Why WSF3D answers a specific question in this paper
---------------------------------------------------
Section 5.1 argues that ANBH carries a retrieval floor: 40.1% of built pixels
in one 0.05 m bin at 2.50-2.55 m, and a cell-level interquartile range of
only 1.8 cm. If the floor is a retrieval artefact rather than real
uniformity of Ghana's building stock, an independent height product should
show substantial height variation among cells that ANBH places on the floor.
That is a test WSF3D can run and the earlier checks (typology, Open
Buildings footprint size) could not, because it compares height with height.

Method
------
1. Clip the global WSF3D BuildingHeight and BuildingFraction GeoTIFFs (2.8
   arcsec, ~90 m) to the same Ghana bounding box as the GHSL rasters. Height is
   Int16 with gain 0.1 (metres = value x 0.1); fraction is integer percent.
2. At native resolution compute hf = height_m x fraction (building-area-
   weighted height) and f = fraction (0-1); nodata/outside-settlement -> 0.
3. Warp hf and f onto the 100 m UTM lattice of the GHSL rasters (script 02)
   with AVERAGE resampling, which preserves area-weighted means.
4. Aggregate to the 500 m grid with script 02's zone raster:
     WSF3D net height  = sum(hf) / sum(f)   (ANBH analogue: height over building area)
     WSF3D gross height = sum(hf) / n_pix   (AGBH analogue: coverage x height)
     WSF3D fraction     = sum(f)  / n_pix
5. Compare on the final analytical sample (n = 296,668).

WSF3D's own placeholder value
-----------------------------
Over Ghana, 8.0% of WSF3D built pixels carry a height of exactly 0.2 m (raw
value 2), almost all in very sparse pixels (mean building fraction 1.9%),
and 16% carry exactly 2.8 m. A 0.2 m building height is not physical, so
the PRIMARY comparison excludes WSF3D pixels below MIN_HEIGHT_M = 1.0 m
(treated as "no usable height"); the all-pixels version is reported as a
sensitivity check (suffix _incl_sub1m).

Outputs
-------
    01_raw_data/wsf3d/wsf3d_{height,fraction}_ghana.tif    (clips; globals are not kept)
    01_raw_data/wsf3d/provenance_wsf3d.json
    02_processed_data/ghsl_layers_v3/wsf3d_grid_500m.parquet
    04_outputs/paper2_v3/tables/T_wsf3d_summary.csv
    04_outputs/paper2_v3/tables/T_wsf3d_floor_band.csv
    04_outputs/paper2_v3/tables/T_wsf3d_by_typology.csv
    04_outputs/paper2_v3/tables/T_wsf3d_by_region.csv

Usage
-----
    python scripts/revision_v3/11_wsf3d_comparison.py --global-dir /path/to/downloaded/globals
    (after the clips exist, --global-dir is no longer needed)
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg

logging.basicConfig(level=cfg.LOG_LEVEL, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

_spec = importlib.util.spec_from_file_location(
    "s02", Path(__file__).with_name("02_process_agbh_anbh_grid.py"))
s02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s02)
gdal_run, gdalinfo_dict, read_envi_array = s02.gdal_run, s02.gdalinfo_dict, s02.read_envi_array

WSF_RAW = cfg.GHSL_RAW.parent / "wsf3d"
INTERMEDIATE = cfg.GHSL_LAYERS_V3 / "_utm_intermediate"
WSF_INT = INTERMEDIATE / "wsf3d"
BASE_URL = "https://download.geoservice.dlr.de/WSF3D/files/global/"
LAYERS = {"height": "WSF3D_V02_BuildingHeight.tif", "fraction": "WSF3D_V02_BuildingFraction.tif"}
HEIGHT_GAIN = 0.1
MIN_HEIGHT_M = 1.0   # primary spec; 0.0 = keep WSF3D's 0.2 m placeholder pixels
N_CELLS = 961_858

HIGH_LABELS = ["High height / High prior expansion", "High height / Low prior expansion"]
CLASS_ORDER = ["High height / High prior expansion", "Low height / Low prior expansion",
               "High height / Low prior expansion", "Low height / High prior expansion"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def extent(tif: Path) -> tuple[float, float, float, float]:
    """(ulx, uly, lrx, lry) of a raster, from gdalinfo."""
    info = gdal_run(["gdalinfo", str(tif)])
    def _xy(key):
        line = next(l for l in info.splitlines() if key in l)
        nums = re.findall(r"[-\d.]+", line.split("(", 1)[1].split(")")[0])
        return float(nums[0]), float(nums[1])
    (ulx, uly), (lrx, lry) = _xy("Upper Left"), _xy("Lower Right")
    return ulx, uly, lrx, lry


def nodata_value(tif: Path):
    m = re.search(r"NoData Value=([-\d.eE+na]+)", gdal_run(["gdalinfo", str(tif)]))
    return None if m is None else m.group(1)


# ------------------------------------------------------------ 1. clip -------
def clip_globals(global_dir: Path | None) -> dict[str, Path]:
    WSF_RAW.mkdir(parents=True, exist_ok=True)
    clips = {k: WSF_RAW / f"wsf3d_{k}_ghana.tif" for k in LAYERS}
    if all(p.exists() for p in clips.values()):
        return clips
    assert global_dir is not None, "Clips missing: pass --global-dir with the downloaded global GeoTIFFs."
    ulx, uly, lrx, lry = extent(cfg.GHSL_RAW / "anbh_2018_v3.tif")   # same Ghana bbox as GHSL
    prov = {
        "product": "World Settlement Footprint 3D (WSF3D), version 2",
        "provider": "German Aerospace Center (DLR)",
        "citation": ("Esch, T. et al. (2022) World Settlement Footprint 3D - A first three-dimensional "
                     "survey of the global building stock. Remote Sensing of Environment, 270, 112877."),
        "license": "CC-BY-4.0",
        "temporal_extent_note": ("Building height from TanDEM-X DEM (2011-2013); settlement mask from "
                                 "WSF-Imperviousness (2017-2019). Settlements built after ~2012 have no "
                                 "height information (DLR temporalextent.txt)."),
        "encoding": {"height": "Int16 metres x 10 (gain 0.1)", "fraction": "integer percent 0-100"},
        "clip_bbox_epsg4326": {"ulx": ulx, "uly": uly, "lrx": lrx, "lry": lry,
                               "note": "identical to the GHSL anbh_2018_v3.tif clip"},
        "files": {},
    }
    dl = global_dir / "download_utc.txt"
    if dl.exists():
        prov["download_completed_utc"] = dl.read_text().strip()
    for key, fname in LAYERS.items():
        src = global_dir / fname
        assert src.exists(), f"missing {src}"
        log.info("Hashing %s (%.2f GB) ...", fname, src.stat().st_size / 1e9)
        g_hash = sha256(src)
        log.info("Clipping %s to Ghana ...", fname)
        gdal_run(["gdal_translate", "-projwin", str(ulx), str(uly), str(lrx), str(lry),
                  "-co", "COMPRESS=LZW", str(src), str(clips[key])])
        prov["files"][key] = {
            "source_url": BASE_URL + fname, "global_size_bytes": src.stat().st_size,
            "global_sha256": g_hash, "clip_path": str(clips[key].relative_to(cfg.PROJECT_ROOT)),
            "clip_sha256": sha256(clips[key]), "nodata_value": nodata_value(clips[key]),
        }
    with open(WSF_RAW / "provenance_wsf3d.json", "w") as f:
        json.dump(prov, f, indent=2)
    log.info("Wrote provenance_wsf3d.json")
    return clips


# ------------------------------------------ 2-4. weight, warp, aggregate ----
def aggregate_to_grid(clips: dict[str, Path], min_height_m: float, tag: str) -> pd.DataFrame:
    WSF_INT.mkdir(parents=True, exist_ok=True)
    h_nd, f_nd = nodata_value(clips["height"]), nodata_value(clips["fraction"])
    log.info("NoData values: height=%s fraction=%s", h_nd, f_nd)

    # Native-resolution products. Anything that is nodata, negative, or has
    # zero fraction contributes zero building area (and so zero weight).
    hf_tif, f_tif = WSF_INT / f"wsf3d_hf_native{tag}.tif", WSF_INT / f"wsf3d_f_native{tag}.tif"
    h_ok = f"(A != {h_nd})" if h_nd not in (None, "nan") else "1"
    f_ok = f"(B != {f_nd})" if f_nd not in (None, "nan") else "1"
    min_raw = int(round(min_height_m / HEIGHT_GAIN))
    valid = f"{h_ok} * {f_ok} * (A > 0) * (A >= {min_raw}) * (B > 0) * (B <= 100)"
    if not hf_tif.exists():
        gdal_run(["gdal_calc.py", "-A", str(clips["height"]), "-B", str(clips["fraction"]),
                  f"--calc={valid} * (A * {HEIGHT_GAIN}) * (B / 100.0)", "--type=Float32",
                  "--NoDataValue=none", f"--outfile={hf_tif}", "--overwrite"])
    if not f_tif.exists():
        gdal_run(["gdal_calc.py", "-A", str(clips["height"]), "-B", str(clips["fraction"]),
                  f"--calc={valid} * (B / 100.0)", "--type=Float32",
                  "--NoDataValue=none", f"--outfile={f_tif}", "--overwrite"])

    agbh_utm = INTERMEDIATE / "agbh_utm.tif"
    meta = gdalinfo_dict(agbh_utm)
    W, H = meta["width"], meta["height"]
    ulx, uly, lrx, lry = extent(agbh_utm)
    out = {}
    for name, src in [("hf", hf_tif), ("f", f_tif)]:
        dst = WSF_INT / f"wsf3d_{name}_utm100{tag}.tif"
        if not dst.exists():
            log.info("Warping WSF3D %s onto the GHSL 100 m UTM lattice (average) ...", name)
            gdal_run(["gdalwarp", "-t_srs", cfg.PROJ_CRS, "-r", "average", "-tr", "100", "100",
                      "-te", str(ulx), str(lry), str(lrx), str(uly), "-ot", "Float32",
                      "-dstnodata", "none", "-overwrite", str(src), str(dst)])
        m = gdalinfo_dict(dst)
        assert (m["width"], m["height"]) == (W, H), f"WSF3D {name} not on the GHSL lattice"
        a = read_envi_array(dst, W, H, "float32").ravel()
        out[name] = np.where(np.isfinite(a) & (a > 0), a, 0.0)

    zone = read_envi_array(INTERMEDIATE / "grid_zone_id.tif", W, H, "int32").ravel()
    m = zone >= 0
    z = zone[m]
    n_pix = np.bincount(z, minlength=N_CELLS)
    s_hf = np.bincount(z, weights=out["hf"][m], minlength=N_CELLS)
    s_f = np.bincount(z, weights=out["f"][m], minlength=N_CELLS)
    with np.errstate(invalid="ignore", divide="ignore"):
        net = np.where(s_f > 0, s_hf / np.maximum(s_f, 1e-12), np.nan)
        gross = np.where(n_pix > 0, s_hf / np.maximum(n_pix, 1), np.nan)
        frac = np.where(n_pix > 0, s_f / np.maximum(n_pix, 1), np.nan)
    df = pd.DataFrame({"grid_id": [f"500m_{i:08d}" for i in range(N_CELLS)],
                       "wsf3d_net_height_m": net, "wsf3d_gross_height_m": gross,
                       "wsf3d_building_fraction": frac})
    df.to_parquet(cfg.GHSL_LAYERS_V3 / f"wsf3d_grid_500m{tag}.parquet", index=False)
    return df


# ---------------------------------------------------------- 5. compare -----
def final_sample() -> pd.DataFrame:
    typ = pd.read_parquet(cfg.GHSL_LAYERS_V3 / "typology_v3.parquet")
    samp = pd.read_parquet(cfg.GHSL_LAYERS_V3 / "sample_definitions.parquet",
                           columns=["grid_id", "cell_agbh_mean_m", "cell_built_fraction_mean"])
    reg = pd.read_parquet(cfg.COVARIATES / "grid_model_ready_500m.parquet",
                          columns=["grid_id", "urban_region_name"])
    df = typ.merge(samp, on="grid_id").merge(reg, on="grid_id")
    assert len(df) == 296_668
    return df


def corr_row(label: str, x: pd.Series, y: pd.Series) -> dict:
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    rho, p_rho = stats.spearmanr(x, y)
    r, p_r = stats.pearsonr(np.log1p(x), np.log1p(y))
    return {"comparison": label, "n": int(ok.sum()), "spearman_rho": rho, "spearman_p": p_rho,
            "pearson_r_log1p": r, "pearson_p": p_r}


def quantiles(s: pd.Series, prefix: str) -> dict:
    q = s.quantile([.10, .25, .50, .75, .90])
    return {f"{prefix}_n": int(s.notna().sum()), f"{prefix}_mean": s.mean(),
            **{f"{prefix}_p{int(k * 100)}": v for k, v in q.items()}}


def compare(wsf: pd.DataFrame, tag: str = "") -> dict:
    T = cfg.PAPER2_V3_TABLES
    df = final_sample().merge(wsf, on="grid_id", how="left")
    anbh, agbh = df.cell_anbh_derived_m, df.cell_agbh_mean_m
    net, gross = df.wsf3d_net_height_m, df.wsf3d_gross_height_m
    has_wsf = net.notna()
    n = len(df)

    # ---- coverage and headline correlations
    rows = [
        corr_row("ANBH vs WSF3D net height (cells where both are defined)", anbh.where(has_wsf), net),
        corr_row("ANBH vs WSF3D net height (both > 0)", anbh.where((anbh > 0) & has_wsf), net),
        corr_row("AGBH vs WSF3D gross height (all sample cells)", agbh, gross),
    ]
    both = has_wsf & (anbh > 0)
    hi_a = anbh > anbh.median()
    hi_w = net > net[both].median()
    agree = (hi_a[both] == hi_w[both])
    po = agree.mean()
    pa, pw = hi_a[both].mean(), hi_w[both].mean()
    pe = pa * pw + (1 - pa) * (1 - pw)
    summary = {
        "n_final_sample": n,
        "n_with_wsf3d_buildings": int(has_wsf.sum()),
        "pct_with_wsf3d_buildings": 100 * has_wsf.mean(),
        "n_anbh_zero": int((anbh == 0).sum()),
        "n_anbh_zero_with_wsf3d_buildings": int(((anbh == 0) & has_wsf).sum()),
        "wsf3d_net_mean_m_where_defined": net.mean(),
        "wsf3d_net_median_m_where_defined": net.median(),
        "anbh_mean_m_same_cells": anbh[has_wsf].mean(),
        "anbh_median_m_same_cells": anbh[has_wsf].median(),
        "median_split_agreement_pct_both_positive": 100 * po,
        "median_split_cohens_kappa_both_positive": (po - pe) / (1 - pe),
    }

    # ---- retrieval-floor test: WSF3D height where ANBH sits on the floor
    lo, hi = anbh.quantile(0.25), anbh.quantile(0.75)
    bands = {
        "ANBH = 0": anbh == 0,
        f"ANBH floor band (IQR {lo:.4f}-{hi:.4f} m)": (anbh >= lo) & (anbh <= hi),
        "ANBH above floor, below p90": (anbh > hi) & (anbh <= anbh.quantile(0.90)),
        "ANBH >= p90": anbh > anbh.quantile(0.90),
    }
    band_rows = []
    for label, mask in bands.items():
        sub = df[mask & has_wsf]
        r = {"anbh_band": label, "n_cells": int(mask.sum()),
             "pct_with_wsf3d": 100 * float(has_wsf[mask].mean()),
             "anbh_min_m": anbh[mask].min(), "anbh_max_m": anbh[mask].max(),
             **quantiles(sub.wsf3d_net_height_m, "wsf3d_net")}
        if len(sub) > 10 and label.startswith("ANBH floor"):
            r["spearman_anbh_vs_wsf3d_within_band"] = stats.spearmanr(
                sub.cell_anbh_derived_m, sub.wsf3d_net_height_m)[0]
        band_rows.append(r)
    fb = pd.DataFrame(band_rows)
    fb.to_csv(T / f"T_wsf3d_floor_band{tag}.csv", index=False)
    log.info("Floor-band test:\n%s", fb.round(3).to_string(index=False))

    # ---- by typology class
    by_t = []
    for cls in CLASS_ORDER:
        mask = df.typology_class == cls
        by_t.append({"typology_class": cls, "n_cells": int(mask.sum()),
                     "pct_with_wsf3d": 100 * float(has_wsf[mask].mean()),
                     "anbh_median_m": anbh[mask].median(),
                     **quantiles(net[mask & has_wsf], "wsf3d_net")})
    bt = pd.DataFrame(by_t)
    hi_mask, lo_mask = df.typology_class.isin(HIGH_LABELS), ~df.typology_class.isin(HIGH_LABELS)
    u = stats.mannwhitneyu(net[hi_mask & has_wsf], net[lo_mask & has_wsf], alternative="greater")
    bt.to_csv(T / f"T_wsf3d_by_typology{tag}.csv", index=False)
    log.info("By typology class:\n%s", bt.round(3).to_string(index=False))
    summary["mannwhitney_wsf3d_net_high_vs_low_height_classes_U"] = float(u.statistic)
    summary["mannwhitney_wsf3d_net_high_vs_low_height_classes_p_one_sided"] = float(u.pvalue)

    # ---- by region
    regs = []
    for name, sub in df.groupby(df.urban_region_name.fillna("Outside named regions")):
        w = sub.wsf3d_net_height_m
        regs.append({"region": name, "n_cells": len(sub), "pct_with_wsf3d": 100 * w.notna().mean(),
                     "anbh_mean_m": sub.cell_anbh_derived_m.mean(),
                     "anbh_median_m": sub.cell_anbh_derived_m.median(),
                     "agbh_mean_m": sub.cell_agbh_mean_m.mean(),
                     "wsf3d_net_mean_m": w.mean(), "wsf3d_net_median_m": w.median(),
                     "wsf3d_gross_mean_m": sub.wsf3d_gross_height_m.mean()})
    rg = pd.DataFrame(regs)
    named = rg[rg.region != "Outside named regions"]
    rank = {
        "region_rank_spearman_anbh_mean_vs_wsf3d_net_mean": stats.spearmanr(named.anbh_mean_m, named.wsf3d_net_mean_m)[0],
        "region_rank_spearman_anbh_median_vs_wsf3d_net_median": stats.spearmanr(named.anbh_median_m, named.wsf3d_net_median_m)[0],
        "region_rank_spearman_agbh_mean_vs_wsf3d_gross_mean": stats.spearmanr(named.agbh_mean_m, named.wsf3d_gross_mean_m)[0],
    }
    rg.to_csv(T / f"T_wsf3d_by_region{tag}.csv", index=False)
    log.info("By region:\n%s", rg.round(3).to_string(index=False))
    log.info("Region-level rank agreement (15 named regions): %s", {k: round(v, 3) for k, v in rank.items()})
    summary.update(rank)

    # One summary table: correlation rows (with n and p-values), then scalar
    # quantities in the "value" column.
    summ = pd.concat([pd.DataFrame(rows), pd.DataFrame([{"comparison": k, "value": v}
                                                        for k, v in summary.items()])],
                     ignore_index=True)
    summ.to_csv(T / f"T_wsf3d_summary{tag}.csv", index=False)
    log.info("Summary%s:\n%s", tag, summ.to_string(index=False))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-dir", type=Path, default=None,
                    help="Directory holding the downloaded global WSF3D GeoTIFFs (first run only)")
    args = ap.parse_args()
    t0 = time.time()
    clips = clip_globals(args.global_dir)
    compare(aggregate_to_grid(clips, MIN_HEIGHT_M, ""), "")                     # primary
    compare(aggregate_to_grid(clips, 0.0, "_incl_sub1m"), "_incl_sub1m")       # sensitivity
    log.info("WSF3D comparison complete in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
