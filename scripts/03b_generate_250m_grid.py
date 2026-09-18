"""
Script 03b — Generate 250 m Model-Ready Grid

Produces grid_model_ready_250m.parquet for the grid-resolution sensitivity
test (script 16).  Uses an arithmetic centroid-projection approach — no
geopandas sjoin — so peak memory stays well under 4 GB.

Strategy
--------
1. Derive 250 m cell extents from the existing 500 m grid bounds (EPSG:32630).
2. Load OB-V3 buildings using their latitude/longitude columns (no geometry).
3. Project centroids to EPSG:32630 with pyproj, assign each to a 250 m cell
   using floor-division arithmetic.
4. Aggregate per cell:
     building_count, total_footprint_area, median_building_area,
     building_area_gini, built_coverage_ratio
5. Load temporal columns from the 500 m grid and disaggregate to 250 m cells
   (each 500 m cell → 4 child 250 m cells inherit the parent's temporal value).
6. Build 250 m cell geometries only for occupied cells (fast).
7. Save to 02_processed_data/covariates/grid_model_ready_250m.parquet.

Usage
-----
    python scripts/03b_generate_250m_grid.py
    python scripts/03b_generate_250m_grid.py --no-temporal
"""

from __future__ import annotations
import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import box

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

CELL_500  = 500.0
CELL_250  = 250.0
CELL_AREA = CELL_250 ** 2   # 62 500 m²
PROJ_EPSG = 32630

OUT_PATH  = cfg.COVARIATES / "grid_model_ready_250m.parquet"


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _gini(areas: np.ndarray) -> float:
    a = areas[np.isfinite(areas) & (areas > 0)]
    if len(a) < 2:
        return np.nan
    a = np.sort(a)
    n = len(a)
    return float((2 * (np.arange(1, n + 1) * a).sum() / (n * a.sum())) - (n + 1) / n)


def _get_500m_extent() -> tuple[float, float, float, float]:
    """Return (x_min, y_min, x_max, y_max) of the 500 m grid in EPSG:32630."""
    grid_500 = pd.read_parquet(
        cfg.COVARIATES / "grid_model_ready_500m.parquet",
        columns=["centroid_lon", "centroid_lat"],
    )
    tf = Transformer.from_crs(4326, PROJ_EPSG, always_xy=True)
    px, py = tf.transform(grid_500["centroid_lon"].values,
                          grid_500["centroid_lat"].values)
    x_min = px.min() - CELL_500 / 2
    y_min = py.min() - CELL_500 / 2
    x_max = px.max() + CELL_500 / 2
    y_max = py.max() + CELL_500 / 2
    log.info(
        "  500 m grid extent (EPSG:32630): x=[%.0f, %.0f]  y=[%.0f, %.0f]",
        x_min, x_max, y_min, y_max,
    )
    return x_min, y_min, x_max, y_max


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: Load OB-V3 and assign to 250 m cells
# ═════════════════════════════════════════════════════════════════════════════

def assign_buildings_to_250m(
    x_min: float, y_min: float,
    chunk_size: int = 2_000_000,
) -> pd.DataFrame:
    """
    Load OB-V3 buildings using latitude/longitude (no geometry),
    project to EPSG:32630, assign to 250 m cells via floor arithmetic.
    Returns DataFrame with columns [grid_id_250m, area_m2].
    """
    ob_path = cfg.BUILDINGS_CLEAN / "gob_v3_ghana_classified.parquet"
    log.info("Loading OB-V3 buildings (lat/lon + area_m2 only) …")
    ob = pd.read_parquet(ob_path, columns=["latitude", "longitude", "area_m2"])
    ob = ob.dropna(subset=["latitude", "longitude"])
    log.info("  %d buildings loaded.", len(ob))

    tf = Transformer.from_crs(4326, PROJ_EPSG, always_xy=True)
    n  = len(ob)
    all_keys = np.empty(n, dtype=np.int64)

    # Process in chunks to keep pyproj memory use low
    log.info("  Projecting centroids and assigning cells (chunks of %d) …", chunk_size)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        px, py = tf.transform(
            ob["longitude"].values[start:end],
            ob["latitude"].values[start:end],
        )
        col_idx = np.floor((px - x_min) / CELL_250).astype(np.int64)
        row_idx = np.floor((py - y_min) / CELL_250).astype(np.int64)
        # Encode (row, col) as a single int64 key
        # Max cols ≈ (x_range / 250) ≈ 1 700 → row * 2000 + col is safe
        all_keys[start:end] = row_idx * 2000 + col_idx
        if (start // chunk_size) % 5 == 0:
            log.info("  … %d / %d processed", end, n)

    ob["_key"] = all_keys
    log.info("  Cell assignment complete.")
    return ob[["_key", "area_m2"]]


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: Aggregate building metrics per 250 m cell
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_metrics(ob_keys: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-cell building metrics from (key, area_m2) pairs.
    """
    log.info("Aggregating building metrics per 250 m cell …")

    # Fast aggregation of count, sum, median
    agg = (
        ob_keys.groupby("_key")["area_m2"]
        .agg(
            building_count="count",
            total_footprint_area="sum",
            median_building_area="median",
        )
        .reset_index()
    )

    # Gini: apply per group (a bit slower but manageable)
    log.info("  Computing per-cell Gini (this may take a minute) …")
    gini_vals = (
        ob_keys.groupby("_key")["area_m2"]
        .apply(lambda s: _gini(s.values))
        .reset_index()
        .rename(columns={"area_m2": "building_area_gini"})
    )

    agg = agg.merge(gini_vals, on="_key", how="left")
    agg["built_coverage_ratio"] = (agg["total_footprint_area"] / CELL_AREA).clip(0, 1)

    log.info(
        "  Aggregation complete: %d occupied cells out of theoretical %.0f",
        len(agg), ((ob_keys["_key"].max() + 1)),
    )
    return agg


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: Add temporal columns from 500 m grid (parent-cell disaggregation)
# ═════════════════════════════════════════════════════════════════════════════

def add_temporal_columns(
    cells: pd.DataFrame,
    x_min: float,
    y_min: float,
) -> pd.DataFrame:
    """
    Each 250 m cell inherits temporal metrics from its parent 500 m cell.
    The parent is the 500 m cell whose south-west corner contains the 250 m
    cell's south-west corner.

    Columns added: building_growth_rate, height_change (and optionally others
    from the 500 m grid).
    """
    log.info("Loading 500 m grid for temporal column disaggregation …")

    temporal_cols = ["centroid_lon", "centroid_lat",
                     "building_growth_rate", "height_change",
                     "long_term_growth", "recent_growth_2010",
                     "urban_region_name"]
    avail = [c for c in temporal_cols
             if c in pd.read_parquet(
                 cfg.COVARIATES / "grid_model_ready_500m.parquet",
                 columns=["centroid_lon"],
             ).columns or c == "centroid_lon"]

    grid_500 = pd.read_parquet(
        cfg.COVARIATES / "grid_model_ready_500m.parquet",
        columns=temporal_cols,
    )

    tf = Transformer.from_crs(4326, PROJ_EPSG, always_xy=True)
    px, py = tf.transform(grid_500["centroid_lon"].values,
                          grid_500["centroid_lat"].values)

    # 500 m cell key uses the same encoding but cell size = 500 m
    # Each 500 m cell centre covers 4 250 m cells; find the 500 m cell
    # that contains each 250 m cell's south-west corner
    # SW corner of 250 m cell at (row, col): x_sw = x_min + col*250, y_sw = y_min + row*250
    # Parent 500 m cell index: col_500 = floor(col_250 * 250 / 500) = col_250 // 2
    #                          row_500 = row_250 // 2
    # Parent key: row_500 * 2000 + col_500  BUT using 500 m encoding:
    # Parent centre: x_min + 500*(col_500 + 0.5), y_min + 500*(row_500 + 0.5)
    # = x_min + col_500*500 + 250, etc.

    col_500 = np.floor((px - x_min) / CELL_500).astype(np.int64)
    row_500 = np.floor((py - y_min) / CELL_500).astype(np.int64)
    # 500 m parent key (same encoding, different stride → use col_500 * 2 for 250 m space)
    parent_key_250 = (row_500 * 2) * 2000 + (col_500 * 2)  # SW child of each 500 m cell

    value_cols = [c for c in temporal_cols
                  if c not in ("centroid_lon", "centroid_lat")]
    parent_df = grid_500[value_cols].copy()
    parent_df["_parent_base_key"] = parent_key_250

    # Each 250 m cell's key → parent: cell key = row_250 * 2000 + col_250
    # row_500 = row_250 // 2  →  row_250 in {2*row_500, 2*row_500+1}
    # So parent base key for a 250 m cell:
    # parent_base_key = (row_250 // 2 * 2) * 2000 + (col_250 // 2 * 2)
    cells["_row"] = cells["_key"] // 2000
    cells["_col"] = cells["_key"] % 2000
    cells["_parent_base_key"] = (cells["_row"] // 2 * 2) * 2000 + (cells["_col"] // 2 * 2)

    # Merge temporal values from parent
    cells = cells.merge(
        parent_df.drop_duplicates("_parent_base_key"),
        on="_parent_base_key",
        how="left",
    )
    cells = cells.drop(columns=["_row", "_col", "_parent_base_key"], errors="ignore")

    filled = sum(cells["building_growth_rate"].notna()) if "building_growth_rate" in cells.columns else 0
    log.info("  Temporal columns added: %d / %d cells with growth rate.", filled, len(cells))
    return cells


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: Build geometries for occupied cells only
# ═════════════════════════════════════════════════════════════════════════════

def build_geometries(
    cells: pd.DataFrame,
    x_min: float,
    y_min: float,
    chunk_size: int = 200_000,
) -> gpd.GeoDataFrame:
    """
    Create 250 m square polygons for each occupied cell.
    Processes in chunks to avoid creating all boxes at once.
    """
    log.info("Building cell geometries (%d cells, chunks of %d) …",
             len(cells), chunk_size)

    row_idx = (cells["_key"] // 2000).values
    col_idx = (cells["_key"] % 2000).values

    x_sw = x_min + col_idx * CELL_250
    y_sw = y_min + row_idx * CELL_250

    geoms = []
    n = len(x_sw)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_geoms = [
            box(x_sw[i], y_sw[i], x_sw[i] + CELL_250, y_sw[i] + CELL_250)
            for i in range(start, end)
        ]
        geoms.extend(chunk_geoms)
        if (start // chunk_size) % 5 == 0:
            log.info("  … %d / %d geometries built", end, n)

    gdf = gpd.GeoDataFrame(cells.copy(), geometry=geoms, crs=f"EPSG:{PROJ_EPSG}")

    # Add centroid lon/lat for downstream compatibility
    cents = gdf.geometry.centroid
    tf_back = Transformer.from_crs(PROJ_EPSG, 4326, always_xy=True)
    gdf["centroid_lon"], gdf["centroid_lat"] = tf_back.transform(
        cents.x.values, cents.y.values
    )

    # Add grid_id and grid_size_m
    gdf["grid_id"]    = "250m_" + gdf["_key"].astype(str).str.zfill(8)
    gdf["grid_size_m"] = 250

    gdf = gdf.drop(columns=["_key"], errors="ignore")

    log.info("  Geometry build complete.  GDF shape: %s", gdf.shape)
    return gdf


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Script 03b — Generate 250 m grid")
    p.add_argument("--no-temporal", action="store_true",
                   help="Skip temporal column disaggregation (building_growth_rate etc).")
    return p.parse_args()


def main():
    args = parse_args()

    log.info("=" * 70)
    log.info("Script 03b — Generate 250 m Model-Ready Grid")
    log.info("=" * 70)

    # ── Derive extents from 500 m grid ────────────────────────────────────────
    log.info("Deriving grid extent from 500 m grid …")
    x_min, y_min, x_max, y_max = _get_500m_extent()

    n_cols_est = int((x_max - x_min) / CELL_250)
    n_rows_est = int((y_max - y_min) / CELL_250)
    log.info("  Theoretical 250 m grid: %d cols × %d rows = %d cells",
             n_cols_est, n_rows_est, n_cols_est * n_rows_est)

    # ── Assign OB-V3 buildings to 250 m cells ────────────────────────────────
    ob_keys = assign_buildings_to_250m(x_min, y_min)

    # ── Aggregate building metrics ────────────────────────────────────────────
    cells = aggregate_metrics(ob_keys)
    del ob_keys  # free ~1-2 GB

    # ── Add temporal columns from 500 m parent ────────────────────────────────
    if not args.no_temporal:
        cells = add_temporal_columns(cells, x_min, y_min)

    # ── Build geometries ──────────────────────────────────────────────────────
    gdf = build_geometries(cells, x_min, y_min)
    del cells

    # ── Save ─────────────────────────────────────────────────────────────────
    log.info("Saving %d cells to %s …", len(gdf), OUT_PATH.name)
    gdf.to_parquet(OUT_PATH, index=False)
    size_mb = OUT_PATH.stat().st_size / 1e6
    log.info("  Saved → %s  (%.1f MB)", OUT_PATH.name, size_mb)

    # ── Quick summary ─────────────────────────────────────────────────────────
    built = gdf[gdf["building_count"] > 0]
    log.info("=" * 70)
    log.info("250 m grid summary:")
    log.info("  Total cells:      %d", len(gdf))
    log.info("  Built-up cells:   %d  (%.1f%%)", len(built), len(built)/len(gdf)*100)
    log.info("  Total buildings:  %d", int(gdf["building_count"].sum()))
    log.info("  Median bldg area: %.1f m²", float(built["median_building_area"].median()))
    log.info("  Median Gini:      %.4f", float(built["building_area_gini"].median()))
    if "building_growth_rate" in gdf.columns:
        log.info("  Growth rate cols: present")
    log.info("  Output:           %s", OUT_PATH)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
