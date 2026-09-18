"""
Phase 4 — Script 05: Process Temporal Building Data

Reads annual building-count and building-height rasters (2016–2023)
downloaded from GEE, samples them at the 500 m grid centroids, computes
temporal change indicators, and saves a grid-level temporal metrics table.

This script keeps all temporal analysis at the GRID CELL level.
The rasters are pre-aggregated to 500 m in GEE, so one raster pixel ≈
one 500 m grid cell.  Alignment is handled by reprojecting to match
the grid's centroid coordinates.

Outputs
-------
    02_processed_data/temporal_layers/temporal_grid_metrics.parquet
        Grid-level table with annual counts, heights, and change indicators.
        Join to the 500m grid via grid_id.

    02_processed_data/temporal_layers/annual_stacks/
        building_count_stack.tif   (8-band: one band per year 2016–2023)
        building_height_stack.tif  (8-band: one band per year 2016–2023)

    04_outputs/tables/temporal_summary_by_city.csv
        Annual means per urban region.

Variable catalogue
------------------
    building_count_{year}       estimated building count per 500 m cell
    mean_height_{year}          mean building height (m) per 500 m cell
    building_count_change       count(2023) - count(2016)
    building_growth_rate        (count(2023) - count(2016)) / count(2016) × 100
    height_change               mean_height(2023) - mean_height(2016)
    vertical_growth_index       normalised height growth minus normalised
                                count growth; positive = intensification,
                                negative = horizontal sprawl
    peak_growth_year            year with highest single-year count increase

Usage
-----
    python scripts/05_process_temporal_buildings.py
    python scripts/05_process_temporal_buildings.py --city "Accra-Tema"
"""

from __future__ import annotations
import argparse
import csv
import gc
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import rioxarray as rxr
import xarray as xr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=cfg.LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

YEARS     = list(range(2016, 2024))
FIRST_YR  = YEARS[0]
LAST_YR   = YEARS[-1]
RASTER_DIR = cfg.TEMPORAL_RAW / "annual"
STACK_DIR  = cfg.TEMPORAL_PROC / "annual_stacks"
OUT_METRICS = cfg.TEMPORAL_PROC / "temporal_grid_metrics.parquet"
OUT_SUMMARY = cfg.TABLES / "temporal_summary_by_city.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — raster checks
# ─────────────────────────────────────────────────────────────────────────────

def raster_path(band: str, year: int) -> Path:
    return RASTER_DIR / f"{band}_{year}.tif"


def check_rasters(years: list[int]) -> tuple[list[int], list[int]]:
    """Return (available_years, missing_years) for both band types."""
    available, missing = [], []
    for yr in years:
        count_ok  = raster_path("building_count",  yr).exists()
        height_ok = raster_path("building_height", yr).exists()
        if count_ok and height_ok:
            available.append(yr)
        else:
            missing.append(yr)
            if not count_ok:
                log.warning("Missing: building_count_%d.tif", yr)
            if not height_ok:
                log.warning("Missing: building_height_%d.tif", yr)
    return available, missing


# ─────────────────────────────────────────────────────────────────────────────
# Load reference grid
# ─────────────────────────────────────────────────────────────────────────────

def load_grid_500m() -> gpd.GeoDataFrame:
    gpkg = cfg.GRIDS / "ghana_grids.gpkg"
    if not gpkg.exists():
        raise FileNotFoundError(
            f"Grid not found: {gpkg}\n"
            "Run scripts/03_prepare_grids.py first."
        )
    log.info("Loading 500 m grid …")
    gdf = gpd.read_file(gpkg, layer="grid_500m", engine="pyogrio")
    log.info("  %d grid cells", len(gdf))
    return gdf


def load_urban_regions() -> gpd.GeoDataFrame:
    gpkg = cfg.CITY_REGIONS / "ghana_admin_levels.gpkg"
    if not gpkg.exists():
        return gpd.GeoDataFrame()
    import fiona
    layers = fiona.listlayers(str(gpkg))
    if "urban_regions" in layers:
        return gpd.read_file(gpkg, layer="urban_regions", engine="pyogrio")
    return gpd.GeoDataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Build multi-year raster stacks
# ─────────────────────────────────────────────────────────────────────────────

def build_stack(band: str, years: list[int]) -> Path:
    """
    Merge individual annual GeoTIFFs into a single multi-band GeoTIFF.
    Band order matches *years* list.  Returns path to the output stack.
    """
    STACK_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STACK_DIR / f"{band}_stack.tif"

    if out_path.exists():
        log.info("Stack already built: %s", out_path.name)
        return out_path

    log.info("Building %s stack (%d bands) …", band, len(years))

    # Read first raster to get profile
    with rasterio.open(raster_path(band, years[0])) as src:
        profile = src.profile.copy()

    profile.update(count=len(years), compress="lzw", dtype="float32")

    with rasterio.open(out_path, "w", **profile) as dst:
        for i, yr in enumerate(tqdm(years, desc=f"Stacking {band}"), start=1):
            p = raster_path(band, yr)
            with rasterio.open(p) as src:
                data = src.read(1).astype("float32")
                # Replace nodata with NaN
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                dst.write(data, i)
            dst.update_tags(i, year=str(yr))

    log.info("  Saved: %s", out_path.name)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Sample raster stack at grid centroids
# ─────────────────────────────────────────────────────────────────────────────

def sample_stack_at_centroids(
    stack_path: Path,
    grid: gpd.GeoDataFrame,
    years: list[int],
    col_prefix: str,
) -> pd.DataFrame:
    """
    Samples a multi-band raster stack at grid cell centroids.
    Returns a DataFrame indexed by grid_id with one column per year.
    """
    log.info("Sampling %s at %d grid centroids …", stack_path.name, len(grid))

    centroids = grid.geometry.centroid
    coords = [(pt.x, pt.y) for pt in centroids]

    with rasterio.open(stack_path) as src:
        # rasterio.sample returns an iterator of shape (n_bands,) per point
        values = list(src.sample(coords, masked=True))

    arr = np.array(values, dtype="float32")   # shape: (n_cells, n_years)
    arr[arr == src.nodata] = np.nan if src.nodata is None else arr[arr == src.nodata]

    cols = {f"{col_prefix}_{yr}": arr[:, i] for i, yr in enumerate(years)}
    df = pd.DataFrame(cols, index=grid["grid_id"].values)
    df.index.name = "grid_id"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Compute temporal change indicators
# ─────────────────────────────────────────────────────────────────────────────

def compute_temporal_indicators(df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """
    Adds temporal change columns to the grid-level DataFrame.

    Input *df* must have columns: building_count_{year} and mean_height_{year}
    for all years.
    """
    log.info("Computing temporal change indicators …")

    first, last = years[0], years[-1]
    cnt_first = df[f"building_count_{first}"].values.astype("float64")
    cnt_last  = df[f"building_count_{last}"].values.astype("float64")
    hgt_first = df[f"mean_height_{first}"].values.astype("float64")
    hgt_last  = df[f"mean_height_{last}"].values.astype("float64")

    eps = 1e-6   # avoid division by zero

    # Absolute change
    df["building_count_change"] = np.round(cnt_last - cnt_first, 2)
    df["height_change"]         = np.round(hgt_last - hgt_first, 2)

    # Growth rate (per cent)
    df["building_growth_rate"] = np.round(
        (cnt_last - cnt_first) / (cnt_first + eps) * 100, 2
    )

    # Vertical growth index
    # Positive → building height grew relatively faster than building count
    # (intensification / vertical densification)
    # Negative → count grew faster (horizontal sprawl)
    height_growth_rate = (hgt_last - hgt_first) / (hgt_first + eps)
    count_growth_rate  = (cnt_last - cnt_first) / (cnt_first + eps)
    df["vertical_growth_index"] = np.round(height_growth_rate - count_growth_rate, 4)

    # Peak growth year: year with greatest single-year count increase
    count_cols = [f"building_count_{yr}" for yr in years]
    count_arr  = df[count_cols].values   # shape: (n_cells, n_years)
    # Annual differences along year axis
    annual_diff = np.diff(count_arr, axis=1)  # shape: (n_cells, n_years-1)
    peak_idx = np.nanargmax(annual_diff, axis=1)
    df["peak_growth_year"] = np.array(years[:-1])[peak_idx]
    # Mark cells with no growth as NaN
    no_growth = np.nanmax(annual_diff, axis=1) <= 0
    df.loc[no_growth, "peak_growth_year"] = pd.NA

    # Growth class (qualitative)
    df["growth_class"] = classify_growth(
        df["building_growth_rate"].values,
        df["vertical_growth_index"].values,
    )

    return df


def classify_growth(growth_rate: np.ndarray, vgi: np.ndarray) -> pd.Series:
    """
    Simple rule-based growth class assigned to each grid cell.
    Used for labelling and visualisation; refined in Phase 11 clustering.
    """
    labels = np.full(len(growth_rate), "stable_or_no_data", dtype=object)

    high_growth  = growth_rate > 50
    mod_growth   = (growth_rate > 10) & (growth_rate <= 50)
    vertical     = (growth_rate <= 10) & (vgi > 0.1)
    rapid_spread = high_growth & (vgi < -0.1)
    rapid_vert   = high_growth & (vgi > 0.1)

    labels[rapid_spread] = "rapid_horizontal_expansion"
    labels[rapid_vert]   = "rapid_vertical_intensification"
    labels[high_growth & ~rapid_spread & ~rapid_vert] = "rapid_growth_mixed"
    labels[mod_growth]   = "moderate_growth"
    labels[vertical]     = "vertical_intensification"

    return pd.Series(labels, dtype="string")


# ─────────────────────────────────────────────────────────────────────────────
# City-level summary table
# ─────────────────────────────────────────────────────────────────────────────

def build_city_summary(
    metrics: pd.DataFrame,
    grid: gpd.GeoDataFrame,
    urban_regions: gpd.GeoDataFrame,
    years: list[int],
) -> pd.DataFrame:
    """
    Joins grid metrics to urban regions and computes mean annual counts
    and heights per urban region.
    """
    if urban_regions.empty:
        log.info("No urban region layer found — skipping city summary.")
        return pd.DataFrame()

    log.info("Building city-level temporal summary …")

    # Join metrics back to grid for spatial operations
    grid_m = grid.set_index("grid_id")[["geometry"]].join(metrics)

    # Spatial join: assign each grid cell to a city region
    centroids = grid_m.copy()
    centroids["geometry"] = grid_m.geometry.centroid
    joined = gpd.sjoin(
        centroids[["geometry"]],
        urban_regions[["name", "geometry"]],
        how="left",
        predicate="within",
    )
    joined_dedup = joined[~joined.index.duplicated(keep="first")]
    grid_m["city_region"] = joined_dedup["name"].reindex(grid_m.index).values

    rows = []
    for region_name, sub in grid_m.groupby("city_region", dropna=True):
        row = {"city_region": region_name, "n_cells": len(sub)}
        for yr in years:
            cnt_col = f"building_count_{yr}"
            hgt_col = f"mean_height_{yr}"
            if cnt_col in sub.columns:
                row[f"total_building_count_{yr}"] = round(sub[cnt_col].sum(), 0)
                row[f"mean_building_count_{yr}"]  = round(sub[cnt_col].mean(), 2)
            if hgt_col in sub.columns:
                row[f"mean_height_{yr}"] = round(sub[hgt_col].mean(skipna=True), 2)
        row["building_count_change"] = round(sub["building_count_change"].sum(), 0)
        row["mean_growth_rate"]      = round(sub["building_growth_rate"].mean(), 2)
        row["mean_vgi"]              = round(sub["vertical_growth_index"].mean(), 4)
        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# City-specific processing (optional — for memory-constrained runs)
# ─────────────────────────────────────────────────────────────────────────────

def get_city_bbox(city_name: str) -> tuple[float, float, float, float] | None:
    """Returns (xmin, ymin, xmax, ymax) for a named city region."""
    csv_path = cfg.CITY_REGIONS / "urban_region_lookup.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    match = df[df["name"] == city_name]
    if match.empty:
        log.warning("City '%s' not found in urban_region_lookup.csv", city_name)
        return None
    row = match.iloc[0]
    return (row["lon_min"], row["lat_min"], row["lon_max"], row["lat_max"])


def clip_rasters_to_city(city_name: str, years: list[int]) -> Path:
    """
    Clips annual rasters to a city bounding box and saves them in a city
    sub-directory.  Useful for exploratory work without loading all of Ghana.
    """
    bbox = get_city_bbox(city_name)
    if bbox is None:
        raise ValueError(f"No bounding box for city: {city_name}")

    city_dir = cfg.TEMPORAL_PROC / "city_clips" / city_name.replace(" ", "_")
    city_dir.mkdir(parents=True, exist_ok=True)

    log.info("Clipping rasters to %s bbox %s …", city_name, bbox)
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box as shapely_box
    city_geom = [shapely_box(*bbox).__geo_interface__]

    for yr in years:
        for band in ["building_count", "building_height"]:
            src_path = raster_path(band, yr)
            dst_path = city_dir / f"{band}_{yr}.tif"
            if dst_path.exists():
                continue
            if not src_path.exists():
                log.warning("  Missing %s — skip clip", src_path.name)
                continue
            with rasterio.open(src_path) as src:
                out_image, out_transform = rio_mask(src, city_geom, crop=True)
                profile = src.profile.copy()
                profile.update(
                    height=out_image.shape[1],
                    width=out_image.shape[2],
                    transform=out_transform,
                )
            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(out_image)

    return city_dir


# ─────────────────────────────────────────────────────────────────────────────
# Logging helper
# ─────────────────────────────────────────────────────────────────────────────

def append_processing_log(step: str, status: str, notes: str = "") -> None:
    log_path = cfg.PROC_LOG / "processing_log.csv"
    if not log_path.exists():
        return
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "script", "phase", "step", "status", "notes"]
        )
        writer.writerow({
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "script": "05_process_temporal_buildings.py",
            "phase": "4",
            "step": step,
            "status": status,
            "notes": notes,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global RASTER_DIR
    parser = argparse.ArgumentParser(
        description="Process temporal building rasters — Phase 4"
    )
    parser.add_argument(
        "--city",
        help=(
            "Process only a single city region (clips rasters first). "
            "If omitted, processes all of Ghana."
        ),
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=YEARS,
        help="Years to include (default: 2016–2023)",
    )
    args = parser.parse_args()

    log.info("=== Phase 4: Process Temporal Building Data ===")
    log.info("Years: %s", args.years)

    # ── Check rasters ─────────────────────────────────────────────────────────
    available, missing = check_rasters(args.years)
    if not available:
        log.error(
            "No temporal rasters found in %s\n"
            "Run scripts/01_download_data/download_temporal_buildings_gee.py first.",
            RASTER_DIR,
        )
        sys.exit(1)

    if missing:
        log.warning(
            "%d year(s) missing rasters: %s\n"
            "Continuing with available years: %s",
            len(missing), missing, available,
        )
    years = available

    # ── Optional: clip to city ────────────────────────────────────────────────
    if args.city:
        log.info("Clipping to city: %s", args.city)
        clip_rasters_to_city(args.city, years)
        # Override raster directory to clipped versions
        RASTER_DIR = cfg.TEMPORAL_PROC / "city_clips" / args.city.replace(" ", "_")

    # ── Build multi-year stacks ───────────────────────────────────────────────
    count_stack_path  = build_stack("building_count",  years)
    height_stack_path = build_stack("building_height", years)
    append_processing_log("Build raster stacks", "complete", f"years={years}")

    # ── Load 500 m grid ───────────────────────────────────────────────────────
    grid = load_grid_500m()
    if args.city:
        # Filter grid to city bbox
        bbox = get_city_bbox(args.city)
        if bbox:
            from shapely.geometry import box as shapely_box
            city_box = shapely_box(*bbox)
            grid = grid[grid.geometry.intersects(city_box)].copy()
            log.info("Grid filtered to %s: %d cells", args.city, len(grid))

    # ── Sample stacks at grid centroids ───────────────────────────────────────
    count_df  = sample_stack_at_centroids(count_stack_path,  grid, years, "building_count")
    height_df = sample_stack_at_centroids(height_stack_path, grid, years, "mean_height")
    append_processing_log("Sample rasters at grid centroids", "complete",
                          f"{len(count_df):,} cells")

    # ── Merge ─────────────────────────────────────────────────────────────────
    metrics = count_df.join(height_df)

    # ── Temporal indicators ───────────────────────────────────────────────────
    metrics = compute_temporal_indicators(metrics, years)
    append_processing_log("Compute temporal indicators", "complete", "")

    # ── Print summary ─────────────────────────────────────────────────────────
    total_growth = metrics["building_count_change"].sum()
    mean_growth_rate = metrics["building_growth_rate"].mean()
    high_growth_pct  = (metrics["building_growth_rate"] > 50).mean() * 100
    vert_pct         = (metrics["vertical_growth_index"] > 0.1).mean() * 100

    log.info(
        "Summary:\n"
        "  Total building count change (2016→%d): %.0f\n"
        "  Mean grid-cell growth rate:             %.1f%%\n"
        "  Cells with >50%% growth:                %.1f%% of cells\n"
        "  Cells with positive vertical growth:    %.1f%% of cells",
        LAST_YR, total_growth, mean_growth_rate, high_growth_pct, vert_pct,
    )

    # ── Save metrics ──────────────────────────────────────────────────────────
    cfg.TEMPORAL_PROC.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    metrics.reset_index().to_parquet(OUT_METRICS, engine="pyarrow", index=False)
    log.info(
        "Temporal metrics → %s  (%d cells, %d columns)",
        OUT_METRICS.relative_to(cfg.PROJECT_ROOT),
        len(metrics),
        len(metrics.columns),
    )

    # ── City-level summary ────────────────────────────────────────────────────
    urban_regions = load_urban_regions()
    city_df = build_city_summary(metrics, grid, urban_regions, years)
    if not city_df.empty:
        cfg.TABLES.mkdir(parents=True, exist_ok=True)
        city_df.to_csv(OUT_SUMMARY, index=False)
        log.info(
            "City summary → %s  (%d regions)",
            OUT_SUMMARY.relative_to(cfg.PROJECT_ROOT), len(city_df),
        )
        append_processing_log("City summary", "complete", f"{len(city_df)} regions")

    append_processing_log(
        "Phase 4 complete", "complete",
        f"metrics: {OUT_METRICS.name}"
    )

    log.info(
        "\nPhase 4 complete.\n"
        "  Temporal metrics:  %s\n"
        "  City summary:      %s\n"
        "  Raster stacks:     %s\n"
        "Next step: scripts/06_process_ghsl.py",
        OUT_METRICS.relative_to(cfg.PROJECT_ROOT),
        OUT_SUMMARY.relative_to(cfg.PROJECT_ROOT),
        STACK_DIR.relative_to(cfg.PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
