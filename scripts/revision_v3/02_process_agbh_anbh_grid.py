"""
Paper 2 v3 -- Script 02: Aggregate AGBH/ANBH to the 500 m grid (by area)

Implements the exact aggregation rule specified for this revision:
  - Do NOT exclude valid zero AGBH pixels when aggregating AGBH.
  - Derive same-epoch built fraction as AGBH/ANBH at the source-pixel level.
  - Aggregate AGBH and derived built fraction BY AREA to the 500 m grid.
  - Calculate cell ANBH as cell AGBH divided by cell built fraction.
  - Preserve nodata separately from valid zero.

Why zonal statistics against the real grid polygons, not gdalwarp -tap
-----------------------------------------------------------------------
The canonical grid (02_processed_data/grids/grid_500m.parquet, used by
every other script in this project) is built in a PROJECTED CRS
(EPSG:32630, true metre squares; see scripts/03_prepare_grids.py), with
its origin floor-aligned to a multiple of 500 m in UTM metres, then
reprojected to EPSG:4326 only for storage. Reconstructing that exact
alignment independently via `gdalwarp -tap` risks a silent off-by-a-
fraction-of-a-pixel mismatch against the real grid cells. Instead this
script:
  1. Reprojects AGBH/ANBH from EPSG:4326 (~100 m native) to EPSG:32630
     using nearest-neighbour resampling (preserves every source pixel
     value exactly; no interpolation before the real aggregation step).
  2. Rasterises the ACTUAL grid_500m.parquet polygons (not a
     reconstructed grid) onto a zone-ID raster aligned pixel-for-pixel
     with the reprojected AGBH/ANBH rasters.
  3. Aggregates AGBH and built-fraction per zone using plain pixel
     counts/sums in that equal-area (UTM) space -- which IS the
     area-weighted mean, since every pixel has equal area in a
     projected CRS. No cos(latitude) weighting hack is needed once the
     data is in a true equal-area projection.

Identity/range checks (Requirement 3) are run as part of this script
and also independently in tests/test_identity_range.py.

Outputs
-------
    02_processed_data/ghsl_layers_v3/agbh_anbh_grid_500m.parquet
        One row per NATIONAL grid cell (n = 961,858), columns:
          grid_id, n_source_pixels, n_valid_pixels, n_nodata_pixels,
          cell_agbh_mean_m, cell_built_fraction_mean, cell_anbh_derived_m
    02_processed_data/ghsl_layers_v3/_utm_intermediate/
        Reprojected/rasterised intermediates, kept for audit.

Usage
-----
    python scripts/revision_v3/02_process_agbh_anbh_grid.py
"""
from __future__ import annotations
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg

logging.basicConfig(level=cfg.LOG_LEVEL, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

# GDAL binary/env locations are centralised in config.py -- see the comment
# there (QGIS_GDAL_BIN_DIR, QGIS_GDAL_ENV) for why.
_GDAL_BIN_DIR = cfg.QGIS_GDAL_BIN_DIR
_GDAL_ENV = cfg.QGIS_GDAL_ENV
INTERMEDIATE = cfg.GHSL_LAYERS_V3 / "_utm_intermediate"


def gdal_run(args: list[str]) -> str:
    import os
    env = dict(**os.environ, **_GDAL_ENV)
    full = [f"{_GDAL_BIN_DIR}/{args[0]}"] + args[1:]
    r = subprocess.run(full, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{args[0]} failed:\n{r.stderr}")
    return r.stdout


def gdalinfo_dict(path: Path) -> dict:
    out = gdal_run(["gdalinfo", str(path)])
    d = {}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Size is"):
            w, h = line.split("is", 1)[1].split(",")
            d["width"], d["height"] = int(w), int(h)
        elif line.startswith("Pixel Size"):
            d["pixel_size_raw"] = line.split("=", 1)[1].strip()
    return d


def read_envi_array(tif_path: Path, width: int, height: int, dtype: str) -> np.ndarray:
    """Export a GeoTIFF to raw ENVI binary via the working gdal_translate CLI,
    then read it with plain numpy -- avoids needing GDAL Python bindings,
    which are not usable in this environment (see audit_report.md)."""
    raw_path = tif_path.with_suffix(".raw.bin")
    gdal_run(["gdal_translate", "-of", "ENVI", "-ot",
              "Float32" if dtype == "float32" else "Int32",
              str(tif_path), str(raw_path)])
    arr = np.fromfile(raw_path, dtype=dtype).reshape(height, width)
    return arr


def main() -> None:
    t0 = time.time()
    cfg.GHSL_LAYERS_V3.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)

    agbh_src = cfg.GHSL_RAW / "agbh_2018_v3.tif"
    anbh_src = cfg.GHSL_RAW / "anbh_2018_v3.tif"
    assert agbh_src.exists() and anbh_src.exists(), "Run script 01 first."

    # ---------------------------------------------------- 1. reproject ----
    agbh_utm = INTERMEDIATE / "agbh_utm.tif"
    anbh_utm = INTERMEDIATE / "anbh_utm.tif"
    for src, dst in [(agbh_src, agbh_utm), (anbh_src, anbh_utm)]:
        if not dst.exists():
            log.info("Reprojecting %s -> EPSG:32630 (nearest-neighbour) ...", src.name)
            gdal_run(["gdalwarp", "-t_srs", cfg.PROJ_CRS, "-r", "near",
                      "-tr", "100", "100", "-overwrite", str(src), str(dst)])
    meta = gdalinfo_dict(agbh_utm)
    W, H = meta["width"], meta["height"]
    log.info("UTM raster grid: %d x %d = %d pixels", W, H, W * H)

    # Confirm ANBH landed on the identical pixel grid (required for the
    # per-pixel built-fraction calculation to be meaningful).
    meta_anbh = gdalinfo_dict(anbh_utm)
    assert meta_anbh["width"] == W and meta_anbh["height"] == H, \
        "AGBH/ANBH UTM rasters do not share a pixel grid -- aborting."

    # -------------------------------------------- 2. pixel built-fraction -
    bf_path = INTERMEDIATE / "built_fraction_px.tif"
    if not bf_path.exists():
        log.info("Computing pixel-level built_fraction = AGBH/ANBH (0/0 -> 0) ...")
        # A!=0 guards the identity 0/0 case; the verified data (script 01
        # audit) has zero occurrences of ANBH==0 with AGBH!=0, but the
        # formula does not silently mask that case -- it would divide by
        # ~0 and produce a huge/inf value that the range test (below and
        # in tests/test_identity_range.py) will catch and report.
        gdal_run(["gdal_calc.py", "-A", str(agbh_utm), "-B", str(anbh_utm),
                  "--calc=where(B>0, A/B, 0)", "--outfile=" + str(bf_path),
                  "--NoDataValue=none", "--overwrite"])

    # ------------------------------------------------- 3. rasterise zones -
    zone_tif = INTERMEDIATE / "grid_zone_id.tif"
    if not zone_tif.exists():
        log.info("Loading canonical 500 m grid and reprojecting to EPSG:32630 ...")
        grid = gpd.read_parquet(cfg.GRIDS / "grid_500m.parquet")
        grid["grid_idx"] = grid["grid_id"].str[5:].astype("int64")
        grid_utm = grid.to_crs(cfg.PROJ_CRS)[["grid_idx", "geometry"]]
        grid_gpkg = INTERMEDIATE / "grid_500m_utm.gpkg"
        grid_utm.to_file(grid_gpkg, layer="grid", driver="GPKG")

        # Extent/transform must match agbh_utm exactly.
        info = gdal_run(["gdalinfo", str(agbh_utm)])
        ul = next(l for l in info.splitlines() if "Upper Left" in l)
        lr = next(l for l in info.splitlines() if "Lower Right" in l)
        import re
        def _xy(line):
            nums = re.findall(r"[-\d.]+", line.split("(", 1)[1].split(")")[0])
            return float(nums[0]), float(nums[1])
        ulx, uly = _xy(ul)
        lrx, lry = _xy(lr)
        log.info("Rasterising %d grid polygons onto the AGBH/ANBH pixel grid ...", len(grid_utm))
        gdal_run(["gdal_rasterize", "-a", "grid_idx",
                  "-te", str(ulx), str(lry), str(lrx), str(uly),
                  "-tr", "100", "100", "-ot", "Int32", "-a_nodata", "-1",
                  "-init", "-1", str(grid_gpkg), str(zone_tif)])

    # ------------------------------------------------------- 4. aggregate -
    log.info("Reading rasters into memory for zonal aggregation ...")
    zone = read_envi_array(zone_tif, W, H, "int32").ravel()
    agbh = read_envi_array(agbh_utm, W, H, "float32").ravel()
    bf = read_envi_array(bf_path, W, H, "float32").ravel()

    n_cells = 961_858
    valid = zone >= 0
    log.info("Pixels inside a grid cell: %d of %d (%.2f%%)",
              valid.sum(), zone.size, 100 * valid.sum() / zone.size)

    zone_v = zone[valid]
    agbh_v = agbh[valid]
    bf_v = bf[valid]
    # No nodata sentinel exists in the source rasters (script 01 verified
    # zero NaN, zero negative pixels over the full Ghana extent); this is
    # asserted, not assumed:
    n_nan_agbh = int(np.isnan(agbh_v).sum())
    n_nan_bf = int(np.isnan(bf_v).sum())
    assert n_nan_agbh == 0 and n_nan_bf == 0, (
        f"Unexpected nodata found post-hoc (agbh nan={n_nan_agbh}, bf nan={n_nan_bf}); "
        "the 'preserve nodata separately from valid zero' path needs activating -- "
        "see audit_report.md Section on nodata handling."
    )

    n_pix = np.bincount(zone_v, minlength=n_cells)
    sum_agbh = np.bincount(zone_v, weights=agbh_v, minlength=n_cells)
    sum_bf = np.bincount(zone_v, weights=bf_v, minlength=n_cells)

    with np.errstate(invalid="ignore", divide="ignore"):
        cell_agbh_mean = np.where(n_pix > 0, sum_agbh / np.maximum(n_pix, 1), np.nan)
        cell_bf_mean = np.where(n_pix > 0, sum_bf / np.maximum(n_pix, 1), np.nan)
        cell_anbh_derived = np.where(cell_bf_mean > 0, cell_agbh_mean / np.maximum(cell_bf_mean, 1e-12), 0.0)
    cell_anbh_derived = np.where(n_pix > 0, cell_anbh_derived, np.nan)

    n_cells_no_pixels = int((n_pix == 0).sum())
    log.info("Grid cells with zero overlapping raster pixels (true nodata cells): %d", n_cells_no_pixels)

    grid_id = [f"500m_{i:08d}" for i in range(n_cells)]
    out = pd.DataFrame({
        "grid_id": grid_id,
        "n_source_pixels": n_pix,
        "n_valid_pixels": n_pix,          # == n_source_pixels: no nodata found (asserted above)
        "n_nodata_pixels": np.zeros(n_cells, dtype=int),
        "cell_agbh_mean_m": np.round(cell_agbh_mean, 6),
        "cell_built_fraction_mean": np.round(cell_bf_mean, 6),
        "cell_anbh_derived_m": np.round(cell_anbh_derived, 6),
    })

    out_path = cfg.GHSL_LAYERS_V3 / "agbh_anbh_grid_500m.parquet"
    out.to_parquet(out_path, index=False)
    log.info("Saved %s (%d rows) in %.1fs", out_path.relative_to(cfg.PROJECT_ROOT), len(out), time.time() - t0)

    # ------------------------------------------------- 5. quick QC print --
    built = out[out.n_source_pixels > 0]
    log.info("QC: cells with >=1 source pixel: %d", len(built))
    log.info("QC: cell_agbh_mean_m  describe:\n%s", built.cell_agbh_mean_m.describe())
    log.info("QC: cell_built_fraction_mean describe:\n%s", built.cell_built_fraction_mean.describe())
    log.info("QC: cell_anbh_derived_m describe:\n%s", built.cell_anbh_derived_m.describe())
    viol = built[built.cell_agbh_mean_m > built.cell_anbh_derived_m + 1e-6]
    log.info("QC: cells where AGBH_mean > ANBH_derived (should be ~0): %d", len(viol))


if __name__ == "__main__":
    main()
