"""
Phase 2 — Script 03: Prepare Analytical Grids
Creates regular fishnet grids at 250 m, 500 m, and 1 km over Ghana,
clipped to the national boundary.  The 500 m grid is the primary modelling unit.

Outputs (all layers in a single GeoPackage):
    02_processed_data/grids/ghana_grids.gpkg
        ├── grid_250m   (~2.1 M cells nationally; use for city-level only)
        ├── grid_500m   (~530 K cells; primary modelling unit)
        └── grid_1km    (~133 K cells; national sensitivity testing)

Each cell has:
    grid_id         globally unique string  (e.g. "500m_00123456")
    grid_size_m     cell edge length in metres
    centroid_lon    centroid longitude (EPSG:4326)
    centroid_lat    centroid latitude  (EPSG:4326)
    geometry        polygon in EPSG:4326

Run after 02_download_boundaries.py:
    python scripts/03_prepare_grids.py
"""

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

logging.basicConfig(
    level=cfg.LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

OUT_GPKG = cfg.GRIDS / "ghana_grids.gpkg"


# ─────────────────────────────────────────────────────────────────────────────
# Load national boundary
# ─────────────────────────────────────────────────────────────────────────────

def load_national_boundary() -> gpd.GeoDataFrame:
    gpkg = cfg.CITY_REGIONS / "ghana_admin_levels.gpkg"
    if not gpkg.exists():
        raise FileNotFoundError(
            f"Admin boundary not found: {gpkg}\n"
            "Run 02_download_boundaries.py first."
        )

    import fiona
    layers = fiona.listlayers(str(gpkg))
    # find the level-0 (national) layer
    national_layer = next((l for l in layers if "0" in l), layers[0])
    log.info("Loading national boundary from layer '%s'", national_layer)

    gdf = gpd.read_file(gpkg, layer=national_layer, engine="pyogrio")
    gdf = gdf.to_crs(cfg.GEO_CRS)
    log.info("National boundary: %d polygon(s), CRS %s", len(gdf), gdf.crs)
    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# Grid creation
# ─────────────────────────────────────────────────────────────────────────────

def make_grid(
    boundary_geo: gpd.GeoDataFrame,
    cell_size_m: int,
    label: str,
) -> gpd.GeoDataFrame:
    """
    Build a regular fishnet grid of *cell_size_m* metre squares covering
    *boundary_geo*, clipped to the boundary polygon.

    Strategy:
      1. Reproject boundary to projected CRS (metres).
      2. Create grid cells aligned to a round-number origin.
      3. Clip to boundary.
      4. Reproject result back to geographic CRS for storage.
    """
    log.info("Creating %s grid (cell = %d m) …", label, cell_size_m)

    boundary_proj = boundary_geo.to_crs(cfg.PROJ_CRS)
    union = boundary_proj.union_all()
    xmin, ymin, xmax, ymax = union.bounds

    # Align origin to a multiple of cell_size for reproducibility
    xmin_aligned = np.floor(xmin / cell_size_m) * cell_size_m
    ymin_aligned = np.floor(ymin / cell_size_m) * cell_size_m

    cols = np.arange(xmin_aligned, xmax + cell_size_m, cell_size_m)
    rows = np.arange(ymin_aligned, ymax + cell_size_m, cell_size_m)

    n_total = (len(cols) - 1) * (len(rows) - 1)
    log.info("  Raw grid extent: %d × %d = %d cells", len(cols) - 1, len(rows) - 1, n_total)

    cells = []
    for x in tqdm(cols[:-1], desc=f"Building {label} cells", unit="col"):
        for y in rows[:-1]:
            cells.append(box(x, y, x + cell_size_m, y + cell_size_m))

    grid_proj = gpd.GeoDataFrame(geometry=cells, crs=cfg.PROJ_CRS)

    # Clip to boundary (keeps only cells that intersect Ghana)
    log.info("  Clipping to national boundary …")
    grid_clipped = gpd.clip(grid_proj, boundary_proj)
    log.info("  After clip: %d cells", len(grid_clipped))

    # Reproject to geographic CRS for storage
    grid_geo = grid_clipped.to_crs(cfg.GEO_CRS)

    # Assign IDs and centroids
    grid_geo = grid_geo.reset_index(drop=True)
    grid_geo["grid_id"] = [f"{label}_{i:08d}" for i in range(len(grid_geo))]
    grid_geo["grid_size_m"] = cell_size_m

    centroids = grid_geo.geometry.centroid
    grid_geo["centroid_lon"] = centroids.x.round(6)
    grid_geo["centroid_lat"] = centroids.y.round(6)

    # Reorder columns
    grid_geo = grid_geo[["grid_id", "grid_size_m", "centroid_lon", "centroid_lat", "geometry"]]

    return grid_geo


# ─────────────────────────────────────────────────────────────────────────────
# Save and log
# ─────────────────────────────────────────────────────────────────────────────

def save_grid(gdf: gpd.GeoDataFrame, layer: str) -> None:
    gdf.to_file(OUT_GPKG, layer=layer, driver="GPKG", engine="pyogrio")
    size_mb = OUT_GPKG.stat().st_size / 1e6
    log.info("Saved layer '%s' → %s  (%.1f MB total)", layer, OUT_GPKG.name, size_mb)


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
            "script": "03_prepare_grids.py",
            "phase": "2",
            "step": step,
            "status": status,
            "notes": notes,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Phase 2: Prepare Analytical Grids ===")

    boundary = load_national_boundary()

    for label, cell_size_m in cfg.GRID_SIZES.items():
        # Skip 250m at national scale if running in quick mode — comment out
        # the next two lines to generate all three grids unconditionally.
        if cell_size_m == 250:
            log.info("Skipping 250m national grid (very large — generate per city region).")
            append_processing_log(f"Grid {label}", "skipped", "250m national grid deferred.")
            continue

        grid = make_grid(boundary, cell_size_m, label)
        save_grid(grid, f"grid_{label}")
        append_processing_log(
            f"Grid {label}", "complete",
            f"{len(grid):,} cells | output: {OUT_GPKG.name}"
        )

    # ── Per-city 250 m grids ─────────────────────────────────────────────────
    # Load urban-region bounding boxes and generate 250 m grids for each.
    log.info("Generating 250 m grids for individual urban regions …")
    ur_gpkg = cfg.CITY_REGIONS / "ghana_admin_levels.gpkg"
    ur_gdf_path = cfg.CITY_REGIONS / "urban_region_lookup.csv"

    if ur_gdf_path.exists():
        import pandas as pd
        from shapely.geometry import box as shapely_box

        ur_df = pd.read_csv(ur_gdf_path)
        ur_gdf = gpd.GeoDataFrame(
            ur_df,
            geometry=[
                shapely_box(row.lon_min, row.lat_min, row.lon_max, row.lat_max)
                for row in ur_df.itertuples()
            ],
            crs=cfg.GEO_CRS,
        )

        city_grids_gpkg = cfg.GRIDS / "city_grids_250m.gpkg"
        for _, region in ur_gdf.iterrows():
            name_safe = region["name"].replace(" ", "_").replace("-", "_").lower()
            region_gdf = gpd.GeoDataFrame(
                [region], geometry="geometry", crs=cfg.GEO_CRS
            )
            grid = make_grid(region_gdf, 250, "250m")
            grid.to_file(city_grids_gpkg, layer=name_safe, driver="GPKG", engine="pyogrio")
            log.info(
                "  %s: %d cells → layer '%s'",
                region["name"], len(grid), name_safe
            )

        append_processing_log(
            "City 250m grids", "complete",
            f"{len(ur_gdf)} regions → {city_grids_gpkg.name}"
        )
    else:
        log.warning(
            "Urban region lookup not found — run 02_download_boundaries.py first. "
            "250 m city grids skipped."
        )

    log.info(
        "Grids complete.\n"
        "  National:  %s  (layers: grid_500m, grid_1km)\n"
        "  Per-city:  %s  (layer per urban region, 250 m)",
        OUT_GPKG.relative_to(cfg.PROJECT_ROOT),
        (cfg.GRIDS / "city_grids_250m.gpkg").relative_to(cfg.PROJECT_ROOT),
    )
    log.info("Next step: run scripts/04_clean_building_footprints.py")


if __name__ == "__main__":
    main()
