"""
Phase 6 — Script 07: Population, Roads, POIs, Terrain, Night-Time Lights

Assembles all grid-level covariate layers into a single analysis-ready
table.  Each data source is processed in a self-contained module that
returns a DataFrame indexed by grid_id, then all modules are merged.

Modules
-------
  population    WorldPop 2020 → count and density per 500 m cell
  roads         OSM road network → density, distance to major road
  pois          OSM/Overture places → proximity metrics per category
  cbd           Predefined city-centre coords → distance per grid cell
  urban_edge    GHSL 2020 binary mask → distance to built-up boundary
  terrain       SRTM DEM → elevation and slope at grid centroids
  ntl           VIIRS DNBV 2020 → night-light intensity at centroids

Output
------
    02_processed_data/covariates/grid_covariates_500m.parquet
        One row per grid cell, joined on grid_id.

Usage
-----
    python scripts/07_add_population_roads_pois.py
    python scripts/07_add_population_roads_pois.py --modules population roads
    python scripts/07_add_population_roads_pois.py --skip-osm-download
"""

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
from rasterio.mask import mask as rio_mask
from scipy.spatial import KDTree
from shapely.geometry import box as shapely_box, Point
import osmnx as ox

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=cfg.LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Output ────────────────────────────────────────────────────────────────────
OUT_FILE = cfg.COVARIATES / "grid_covariates_500m.parquet"

# ── Source files ──────────────────────────────────────────────────────────────
WORLDPOP_FILE  = cfg.WORLDPOP_RAW      / "gha_ppp_2020.tif"
SRTM_FILE      = cfg.TERRAIN_RAW       / "srtm_dem_ghana.tif"
NTL_FILE       = cfg.NIGHT_LIGHTS_RAW  / "viirs_ntl_2020_ghana.tif"
OSM_PBF_FILE   = cfg.ROADS_RAW         / "ghana-latest.osm.pbf"
OSM_ROADS_GPKG = cfg.ROADS_RAW         / "ghana_roads.gpkg"
OSM_POIS_GPKG  = cfg.ROADS_RAW         / "ghana_pois.gpkg"

# ── Road classification ───────────────────────────────────────────────────────
# osmnx highway tag values, ordered by importance
MAJOR_ROAD_TYPES  = {"motorway", "trunk", "primary", "secondary",
                     "motorway_link", "trunk_link", "primary_link", "secondary_link"}
ALL_ROAD_TYPES    = MAJOR_ROAD_TYPES | {
    "tertiary", "tertiary_link", "unclassified", "residential",
    "living_street", "track", "service",
}

# ── Known CBD coordinates (lon, lat) EPSG:4326 ────────────────────────────────
CBD_COORDS = {
    "Accra":            (-0.187, 5.556),
    "Kumasi":           (-1.624, 6.688),
    "Tamale":           (-0.839, 9.401),
    "Sekondi-Takoradi": (-1.714, 4.934),
    "Cape Coast":       (-1.242, 5.105),
    "Koforidua":        (-0.261, 6.088),
    "Sunyani":          (-2.329, 7.340),
    "Ho":               ( 0.472, 6.601),
    "Wa":               (-2.503, 10.060),
    "Bolgatanga":       (-0.851, 10.785),
}

# ── POI categories and osmnx tags ────────────────────────────────────────────
POI_QUERIES = {
    "school":      {"amenity": "school"},
    "hospital":    {"amenity": ["hospital", "clinic", "health_centre"]},
    "market":      {"amenity": ["marketplace", "market"], "shop": "supermarket"},
    "university":  {"amenity": "university"},
    "church":      {"amenity": ["place_of_worship"]},
    "industrial":  {"landuse": "industrial"},
    "port":        {"industrial": "port", "harbour": True},
    "bus_terminal":{"amenity": ["bus_station", "bus_stop"]},
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_grid() -> gpd.GeoDataFrame:
    gpkg = cfg.GRIDS / "ghana_grids.gpkg"
    if not gpkg.exists():
        raise FileNotFoundError(f"Grid not found: {gpkg}")
    gdf = gpd.read_file(gpkg, layer="grid_500m", engine="pyogrio")
    log.info("Grid loaded: %d cells", len(gdf))
    return gdf


def grid_centroids_proj(grid: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Returns (xy_proj, xy_geo) arrays of centroid coordinates."""
    geo_cents = grid.geometry.centroid
    proj_cents = geo_cents.to_crs(cfg.PROJ_CRS)
    xy_proj = np.column_stack([proj_cents.x, proj_cents.y])
    xy_geo  = np.column_stack([geo_cents.x, geo_cents.y])
    return xy_proj, xy_geo


def sample_raster_at_points(
    raster_path: Path,
    xy_geo: np.ndarray,
    band: int = 1,
    nodata_fill: float = np.nan,
) -> np.ndarray:
    """
    Samples a raster at (lon, lat) coordinates.
    Returns 1-D array aligned with xy_geo.
    """
    if not raster_path.exists():
        log.warning("Raster not found, returning NaN: %s", raster_path.name)
        return np.full(len(xy_geo), nodata_fill, dtype="float32")

    with rasterio.open(raster_path) as src:
        coords = [(x, y) for x, y in xy_geo]
        vals = np.array(list(src.sample(coords, indexes=band)), dtype="float32").ravel()
        if src.nodata is not None:
            vals[vals == src.nodata] = nodata_fill
    return vals


def rasterise_and_sum(
    raster_path: Path,
    grid: gpd.GeoDataFrame,
    band: int = 1,
) -> np.ndarray:
    """
    Aggregates a raster to grid cells using the rasterisation approach:
    burn grid IDs into a temp raster, then groupby-sum.
    Returns 1-D array of sums aligned with grid index.
    """
    from rasterio import features as rio_features

    if not raster_path.exists():
        log.warning("Raster not found, returning zeros: %s", raster_path.name)
        return np.zeros(len(grid), dtype="float32")

    with rasterio.open(raster_path) as src:
        profile = src.profile
        H, W = src.shape
        transform = src.transform
        crs = src.crs

        grid_repr = grid.to_crs(crs) if str(grid.crs) != str(crs) else grid
        shapes = (
            (geom.__geo_interface__, idx)
            for idx, geom in enumerate(grid_repr.geometry)
        )
        cell_raster = rio_features.rasterize(
            shapes=shapes, out_shape=(H, W), transform=transform,
            fill=-1, dtype="int32",
        )

        data = src.read(band).astype("float32")
        if src.nodata is not None:
            data[data == src.nodata] = 0.0
        data = np.nan_to_num(data, nan=0.0)

    sums = np.zeros(len(grid), dtype="float64")
    valid = cell_raster >= 0
    np.add.at(sums, cell_raster[valid], data[valid])
    return sums.astype("float32")


def nearest_distance_m(
    query_xy: np.ndarray,
    target_xy: np.ndarray,
) -> np.ndarray:
    """
    Returns distance in metres from each query point to its nearest target
    point.  Both arrays must be in the same projected CRS (metres).
    """
    if len(target_xy) == 0:
        return np.full(len(query_xy), np.nan, dtype="float32")
    tree = KDTree(target_xy)
    dists, _ = tree.query(query_xy, workers=-1)
    return dists.astype("float32")


# ─────────────────────────────────────────────────────────────────────────────
# Module 1 — Population
# ─────────────────────────────────────────────────────────────────────────────

def process_population(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    log.info("--- Module: Population ---")
    ids = grid["grid_id"].values

    # Sum WorldPop pixels within each cell (100 m → 500 m)
    pop_sum = rasterise_and_sum(WORLDPOP_FILE, grid, band=1)

    # Cell area in km²: for 500 m grid = 0.25 km²
    cell_area_km2 = (500 ** 2) / 1e6

    df = pd.DataFrame({
        "population_count":   np.round(pop_sum, 1),
        "population_density": np.round(pop_sum / cell_area_km2, 1),
    }, index=ids)
    df.index.name = "grid_id"

    log.info(
        "  Total population: {:,.0f}  |  Mean cell density: {:.0f}/km²".format(
            pop_sum.sum(), (pop_sum / cell_area_km2).mean()
        )
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Module 2 — Roads
# ─────────────────────────────────────────────────────────────────────────────

def fetch_osm_roads(skip_download: bool = False) -> gpd.GeoDataFrame:
    """
    Returns a GeoDataFrame of OSM road lines for Ghana.
    Primary: read from pre-downloaded Geofabrik OSM PBF via fiona/pyogrio.
    Fallback: query via osmnx (slower but no file required).
    """
    if OSM_ROADS_GPKG.exists():
        log.info("  Loading roads from cached GeoPackage …")
        return gpd.read_file(OSM_ROADS_GPKG, engine="pyogrio")

    # Try reading OSM PBF directly with fiona (needs GDAL OSM driver)
    if OSM_PBF_FILE.exists() and not skip_download:
        log.info("  Reading roads from OSM PBF …")
        try:
            import fiona
            roads = gpd.read_file(
                str(OSM_PBF_FILE),
                layer="lines",
                engine="fiona",
                where="highway IS NOT NULL",
            )
            roads = roads[roads["highway"].notna()].copy()
            roads = roads.to_crs(cfg.GEO_CRS)
            roads.to_file(OSM_ROADS_GPKG, driver="GPKG", engine="pyogrio")
            log.info("  Roads extracted from PBF: %d features", len(roads))
            return roads
        except Exception as exc:
            log.warning("  PBF read failed (%s) — falling back to osmnx.", exc)

    # osmnx fallback: download by bbox
    log.info("  Downloading road network via osmnx (this may take a few minutes) …")
    try:
        ox.settings.timeout = 600
        ox.settings.max_query_area_size = 50e9
        north, south, east, west = 11.3, 4.3, 1.3, -3.5
        G = ox.graph_from_bbox(
            bbox=(north, south, east, west),
            network_type="all",
            retain_all=False,
        )
        roads, _ = ox.graph_to_gdfs(G)
        roads = roads.to_crs(cfg.GEO_CRS)
        roads.to_file(OSM_ROADS_GPKG, driver="GPKG", engine="pyogrio")
        log.info("  Roads downloaded via osmnx: %d edges", len(roads))
        return roads
    except Exception as exc:
        log.error("  osmnx download failed: %s", exc)
        return gpd.GeoDataFrame()


def process_roads(grid: gpd.GeoDataFrame, skip_osm_download: bool = False) -> pd.DataFrame:
    log.info("--- Module: Roads ---")
    ids = grid["grid_id"].values
    xy_proj, _ = grid_centroids_proj(grid)

    roads = fetch_osm_roads(skip_download=skip_osm_download)
    if roads.empty:
        log.warning("  No road data — returning NaN columns.")
        df = pd.DataFrame({
            "road_length_total_m":    np.nan,
            "road_density_m_per_km2": np.nan,
            "road_length_major_m":    np.nan,
            "distance_to_major_road_m": np.nan,
        }, index=ids)
        df.index.name = "grid_id"
        return df

    # ── Road length per grid cell ──────────────────────────────────────────
    log.info("  Computing road lengths per grid cell …")
    roads_proj = roads.to_crs(cfg.PROJ_CRS)
    grid_proj  = grid.to_crs(cfg.PROJ_CRS)

    # Spatial join: each road segment → grid cell
    road_join = gpd.sjoin(
        roads_proj[["highway", "geometry"]].reset_index(),
        grid_proj[["grid_id", "geometry"]],
        how="left",
        predicate="intersects",
    )
    road_join["seg_length_m"] = road_join.geometry.length

    total_by_cell = (
        road_join.groupby("grid_id")["seg_length_m"].sum().rename("road_length_total_m")
    )
    major_mask = road_join["highway"].apply(
        lambda h: bool(h and any(rt in str(h) for rt in MAJOR_ROAD_TYPES))
    )
    major_by_cell = (
        road_join[major_mask].groupby("grid_id")["seg_length_m"].sum()
        .rename("road_length_major_m")
    )

    cell_area_km2 = 0.25  # 500 m × 500 m
    road_density  = (total_by_cell / cell_area_km2).rename("road_density_m_per_km2")

    # ── Distance to major road ────────────────────────────────────────────
    log.info("  Computing distance to nearest major road …")
    major_roads = roads_proj[roads_proj["highway"].apply(
        lambda h: bool(h and any(rt in str(h) for rt in MAJOR_ROAD_TYPES))
    )]

    if not major_roads.empty:
        # Sample points along major roads for KDTree
        major_pts = []
        for geom in major_roads.geometry:
            if geom is not None and not geom.is_empty:
                for d in np.linspace(0, geom.length, max(2, int(geom.length / 100))):
                    pt = geom.interpolate(d)
                    major_pts.append([pt.x, pt.y])
        major_arr = np.array(major_pts, dtype="float64")
        dist_major = nearest_distance_m(xy_proj, major_arr)
    else:
        dist_major = np.full(len(grid), np.nan, dtype="float32")

    df = pd.DataFrame(index=ids)
    df.index.name = "grid_id"
    df = df.join(total_by_cell).join(road_density).join(major_by_cell)
    df["distance_to_major_road_m"] = dist_major
    df = df.fillna({"road_length_total_m": 0, "road_length_major_m": 0,
                    "road_density_m_per_km2": 0})
    df = df.round(1)

    log.info(
        "  Road density: mean={:.0f} m/km², max={:.0f} m/km²".format(
            df["road_density_m_per_km2"].mean(), df["road_density_m_per_km2"].max()
        )
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Module 3 — POI proximity
# ─────────────────────────────────────────────────────────────────────────────

def fetch_pois(category: str, tags: dict) -> gpd.GeoDataFrame:
    """Fetches POIs for one category from OSM via osmnx, with caching."""
    cache_layer = f"pois_{category}"

    if OSM_POIS_GPKG.exists():
        import fiona
        if cache_layer in fiona.listlayers(str(OSM_POIS_GPKG)):
            return gpd.read_file(OSM_POIS_GPKG, layer=cache_layer, engine="pyogrio")

    try:
        ox.settings.timeout = 300
        gdf = ox.features_from_bbox(
            bbox=(11.3, 4.3, 1.3, -3.5),
            tags=tags,
        )
        if gdf.empty:
            return gpd.GeoDataFrame()
        gdf = gdf.to_crs(cfg.GEO_CRS)
        # Use centroids for polygon features
        pts = gdf.copy()
        pts["geometry"] = pts.geometry.centroid
        pts = pts[pts.geometry.notna()].reset_index(drop=True)
        pts[["geometry"]].to_file(OSM_POIS_GPKG, layer=cache_layer, driver="GPKG", engine="pyogrio")
        return pts
    except Exception as exc:
        log.warning("  POI fetch failed for '%s': %s", category, exc)
        return gpd.GeoDataFrame()


def process_pois(grid: gpd.GeoDataFrame, skip_osm_download: bool = False) -> pd.DataFrame:
    log.info("--- Module: POI Proximity ---")
    ids = grid["grid_id"].values
    xy_proj, _ = grid_centroids_proj(grid)

    df = pd.DataFrame(index=ids)
    df.index.name = "grid_id"

    if skip_osm_download and not OSM_POIS_GPKG.exists():
        log.warning("  Skipping POI download — returning NaN columns.")
        for cat in POI_QUERIES:
            df[f"distance_to_{cat}_m"] = np.nan
        return df

    for category, tags in POI_QUERIES.items():
        log.info("  Fetching POIs: %s …", category)
        pois = fetch_pois(category, tags)
        col  = f"distance_to_{category}_m"
        if pois.empty:
            log.warning("    No POIs found for category '%s'", category)
            df[col] = np.nan
            continue
        pois_proj = pois.to_crs(cfg.PROJ_CRS)
        poi_xy = np.column_stack([pois_proj.geometry.x, pois_proj.geometry.y])
        df[col] = nearest_distance_m(xy_proj, poi_xy).round(0)
        log.info("    %d POIs  |  median dist: %.0f m", len(pois), df[col].median())

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Module 4 — CBD distances
# ─────────────────────────────────────────────────────────────────────────────

def process_cbd_distance(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    log.info("--- Module: CBD Distances ---")
    ids = grid["grid_id"].values
    xy_proj, _ = grid_centroids_proj(grid)

    # Project CBD coords
    cbd_gdf = gpd.GeoDataFrame(
        {"name": list(CBD_COORDS.keys())},
        geometry=[Point(lon, lat) for lon, lat in CBD_COORDS.values()],
        crs=cfg.GEO_CRS,
    ).to_crs(cfg.PROJ_CRS)
    cbd_xy = np.column_stack([cbd_gdf.geometry.x, cbd_gdf.geometry.y])

    # Distance to NEAREST CBD
    dist_nearest = nearest_distance_m(xy_proj, cbd_xy)

    # Also distance to each named CBD
    df = pd.DataFrame({"distance_to_nearest_cbd_m": dist_nearest.round(0)}, index=ids)
    df.index.name = "grid_id"

    for i, city in enumerate(cbd_gdf["name"]):
        col = f"distance_to_{city.lower().replace(' ', '_').replace('-', '_')}_cbd_m"
        dists = np.linalg.norm(xy_proj - cbd_xy[i], axis=1).astype("float32")
        df[col] = dists.round(0)

    # Name of nearest CBD
    tree = KDTree(cbd_xy)
    _, idx = tree.query(xy_proj)
    df["nearest_cbd"] = cbd_gdf["name"].iloc[idx].values

    log.info("  CBD distances computed for %d cities.", len(CBD_COORDS))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Module 5 — Distance to urban edge
# ─────────────────────────────────────────────────────────────────────────────

def process_urban_edge_distance(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Computes distance from each grid centroid to the nearest 'urban edge' —
    defined as built-up cells (built_coverage_2020 >= threshold) that are
    adjacent to non-built-up cells.  Uses the GHSL grid metrics if available,
    otherwise returns NaN.
    """
    log.info("--- Module: Urban Edge Distance ---")
    ids = grid["grid_id"].values
    xy_proj, _ = grid_centroids_proj(grid)

    ghsl_file = cfg.GHSL_PROC / "ghsl_grid_metrics.parquet"
    df_out = pd.DataFrame({"distance_to_urban_edge_m": np.nan}, index=ids)
    df_out.index.name = "grid_id"

    if not ghsl_file.exists():
        log.warning("  GHSL grid metrics not found — urban edge distance will be NaN.")
        log.warning("  Run scripts/06_process_ghsl.py first.")
        return df_out

    ghsl = pd.read_parquet(ghsl_file, columns=["grid_id", "built_coverage_2020"])
    BUILT_THRESHOLD = 0.05  # 5% built coverage counts as urban

    grid_m = grid.set_index("grid_id").join(ghsl.set_index("grid_id"))
    grid_m["is_urban"] = grid_m["built_coverage_2020"].fillna(0) >= BUILT_THRESHOLD

    # Edge cells: urban cells adjacent to non-urban cells
    # Simple approach: get centroids of urban cells, then find non-urban cells
    # nearest to an urban cell with a non-urban neighbour.
    # Approximate with all urban-cell centroids as the "edge" reference set.
    urban_mask = grid_m["is_urban"].values
    urban_proj = grid.to_crs(cfg.PROJ_CRS)
    urban_xy = np.column_stack([
        urban_proj.geometry.centroid.x.values[urban_mask],
        urban_proj.geometry.centroid.y.values[urban_mask],
    ])

    if len(urban_xy) == 0:
        return df_out

    dists = nearest_distance_m(xy_proj, urban_xy)
    # For cells that are already urban, distance is ~0; keep it
    df_out["distance_to_urban_edge_m"] = dists.round(0)
    log.info(
        "  Urban cells: %d  |  Non-urban cells: %d",
        urban_mask.sum(), (~urban_mask).sum(),
    )
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Module 6 — Terrain
# ─────────────────────────────────────────────────────────────────────────────

def process_terrain(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    log.info("--- Module: Terrain ---")
    ids   = grid["grid_id"].values
    _, xy_geo = grid_centroids_proj(grid)

    elevation = sample_raster_at_points(SRTM_FILE, xy_geo, band=1)

    # Compute slope from DEM using gradient (approximate degrees)
    if SRTM_FILE.exists():
        with rasterio.open(SRTM_FILE) as src:
            dem    = src.read(1).astype("float32")
            nodata = src.nodata
            res_x  = abs(src.transform.a)   # degrees per pixel
            res_y  = abs(src.transform.e)
            if nodata is not None:
                dem[dem == nodata] = np.nan

        # Convert degree-resolution to approximate metres at Ghana latitude (~7°N)
        lat_rad   = np.radians(7.0)
        m_per_deg_lon = 111_320 * np.cos(lat_rad)
        m_per_deg_lat = 110_574

        dy, dx = np.gradient(
            np.nan_to_num(dem),
            res_y * m_per_deg_lat,
            res_x * m_per_deg_lon,
        )
        slope_arr = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

        # Sample slope at centroid grid coordinates
        slope_raster_path = cfg.TERRAIN_RAW / "_slope_tmp.tif"
        profile = {
            "driver": "GTiff", "dtype": "float32", "count": 1,
            "width": slope_arr.shape[1], "height": slope_arr.shape[0],
            "nodata": -9999.0, "compress": "lzw",
        }
        with rasterio.open(SRTM_FILE) as src:
            profile["transform"] = src.transform
            profile["crs"]       = src.crs
        with rasterio.open(slope_raster_path, "w", **profile) as dst:
            dst.write(slope_arr[np.newaxis, :, :])

        slope = sample_raster_at_points(slope_raster_path, xy_geo, band=1)
        slope_raster_path.unlink(missing_ok=True)
    else:
        slope = np.full(len(ids), np.nan, dtype="float32")

    df = pd.DataFrame({
        "elevation_m": np.round(elevation, 1),
        "slope_deg":   np.round(slope,     2),
    }, index=ids)
    df.index.name = "grid_id"
    log.info(
        "  Elevation: mean={:.0f} m  |  Slope: mean={:.1f}°".format(
            np.nanmean(elevation), np.nanmean(slope)
        )
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Module 7 — Night-time lights
# ─────────────────────────────────────────────────────────────────────────────

def process_ntl(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    log.info("--- Module: Night-Time Lights ---")
    ids = grid["grid_id"].values
    _, xy_geo = grid_centroids_proj(grid)

    ntl = sample_raster_at_points(NTL_FILE, xy_geo, band=1, nodata_fill=0.0)
    ntl = np.where(ntl < 0, 0.0, ntl)   # clip negative outliers

    df = pd.DataFrame({
        "ntl_avg_rad_2020": np.round(ntl, 4),
    }, index=ids)
    df.index.name = "grid_id"
    log.info(
        "  NTL: mean={:.4f}  max={:.4f}  (nW/cm²/sr)".format(
            np.nanmean(ntl), np.nanmax(ntl)
        )
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Compile and save
# ─────────────────────────────────────────────────────────────────────────────

def compile_and_save(module_dfs: list[pd.DataFrame], grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log.info("Compiling covariate table …")
    base = grid.set_index("grid_id")[["centroid_lon", "centroid_lat", "geometry"]]
    merged = base.copy()
    for df in module_dfs:
        merged = merged.join(df, how="left")

    merged = merged.reset_index()
    cfg.COVARIATES.mkdir(parents=True, exist_ok=True)
    cols = [c for c in merged.columns if c != "geometry"] + ["geometry"]
    merged[cols].to_parquet(OUT_FILE, engine="pyarrow", index=False)
    size_mb = OUT_FILE.stat().st_size / 1e6
    log.info(
        "Covariate table → %s  (%d cells, %d columns, %.1f MB)",
        OUT_FILE.relative_to(cfg.PROJECT_ROOT),
        len(merged), len(merged.columns), size_mb,
    )
    return merged


def print_summary(gdf: gpd.GeoDataFrame) -> None:
    numeric = gdf.select_dtypes("number").columns.tolist()
    log.info("Covariate summary (means across grid cells):")
    for col in sorted(numeric):
        vals = gdf[col].dropna()
        if len(vals) > 0:
            log.info("  %-40s  mean=%10.2f  missing=%.1f%%",
                     col, vals.mean(), 100 * gdf[col].isna().mean())


# ─────────────────────────────────────────────────────────────────────────────
# Processing log
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
            "script": "07_add_population_roads_pois.py",
            "phase": "6",
            "step": step,
            "status": status,
            "notes": notes,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

ALL_MODULES = ["population", "roads", "pois", "cbd", "urban_edge", "terrain", "ntl"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble covariate table — Phase 6")
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=ALL_MODULES,
        default=ALL_MODULES,
        help="Modules to run (default: all)",
    )
    parser.add_argument(
        "--skip-osm-download",
        action="store_true",
        help="Do not attempt to download OSM data via osmnx (use cached files only)",
    )
    args = parser.parse_args()

    log.info("=== Phase 6: Population, Roads, POIs, Terrain, Night-Time Lights ===")
    log.info("Modules: %s", args.modules)

    grid = load_grid()
    module_dfs: list[pd.DataFrame] = []

    def run(name: str, fn, *fn_args):
        if name not in args.modules:
            return
        try:
            df = fn(*fn_args)
            module_dfs.append(df)
            append_processing_log(f"Module: {name}", "complete",
                                  f"{len(df.columns)} columns")
            gc.collect()
        except Exception as exc:
            log.error("Module '%s' failed: %s", name, exc)
            append_processing_log(f"Module: {name}", "failed", str(exc))

    run("population",  process_population,        grid)
    run("roads",       process_roads,             grid, args.skip_osm_download)
    run("pois",        process_pois,              grid, args.skip_osm_download)
    run("cbd",         process_cbd_distance,       grid)
    run("urban_edge",  process_urban_edge_distance, grid)
    run("terrain",     process_terrain,            grid)
    run("ntl",         process_ntl,               grid)

    if not module_dfs:
        log.error("No modules produced output — nothing to save.")
        sys.exit(1)

    result = compile_and_save(module_dfs, grid)
    print_summary(result)
    append_processing_log("Phase 6 complete", "complete",
                          f"{len(result.columns)} total columns")

    log.info(
        "\nPhase 6 complete.\n"
        "  Covariate table: %s\n"
        "Next step: scripts/08_calculate_building_metrics.py",
        OUT_FILE.relative_to(cfg.PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
