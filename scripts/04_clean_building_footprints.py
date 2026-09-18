"""
Phase 3 — Script 04: Clean Building Footprints

Reads raw Open Buildings V3 tiles and Overture Maps buildings, applies
a full cleaning and geometry-metrics pipeline, and saves analysis-ready
GeoParquet files.

Processing steps
----------------
For each source (OB V3, Overture):
  1. Load raw tiles / parquet file
  2. Clip to Ghana national boundary
  3. Validate and repair geometries
  4. Filter by minimum confidence (OB V3 only)
  5. Remove exact-duplicate geometries
  6. Calculate building-level metrics (area, perimeter, shape, volume)
  7. Assign provisional size class
  8. Flag suspect geometries (too small, non-polygon)
  9. Spatial join: assign grid_id_500m, grid_id_1km, district_id
 10. Save as GeoParquet

The large-building flag is SET for monitoring only.  Large buildings are
NEVER removed (they may represent warehouses, churches, institutions, malls).

Outputs
-------
    02_processed_data/buildings_cleaned/gob_v3_ghana.parquet
    02_processed_data/buildings_cleaned/overture_ghana.parquet
    04_outputs/tables/building_qc_report.csv

Usage
-----
    python scripts/04_clean_building_footprints.py [--source {gob,overture,both}]
"""

import argparse
import csv
import gc
import logging
import sys
import warnings
from datetime import datetime
from math import pi, sqrt, log1p
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.errors import ShapelyDeprecationWarning as ShapelyWarning
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

warnings.filterwarnings("ignore", category=ShapelyWarning)

logging.basicConfig(
    level=cfg.LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# OB V3 confidence threshold: keep buildings with confidence >= this value.
# 0.65 is a widely-used threshold in the literature; lower = more buildings
# but higher omission/commission.
MIN_CONFIDENCE = 0.65

# Area thresholds for QC flags (sq m, measured in projected CRS)
FLAG_TINY_M2     = cfg.MIN_BUILDING_AREA_M2  # likely noise or partial footprint
FLAG_LARGE_M2    = 50_000                    # >5 ha — flagged for review, NOT removed

# Tile processing batch size (rows read from each CSV.gz at once)
TILE_CHUNKSIZE   = 200_000

# ── OB V3 column schema ───────────────────────────────────────────────────────
# Expected columns in Open Buildings V3 CSV tiles.
# Update these if the schema changes between releases.
OBV3_GEOMETRY_COL  = "geometry"   # WKT polygon string
OBV3_AREA_COL      = "area_in_meters"   # V3 uses American spelling
OBV3_CONF_COL      = "confidence"
OBV3_LAT_COL       = "latitude"
OBV3_LON_COL       = "longitude"

# ── Overture column schema ────────────────────────────────────────────────────
OVT_GEOM_COL   = "geometry"        # standard GeoDataFrame geometry column
OVT_HEIGHT_COL = "height"
OVT_FLOORS_COL = "num_floors"
OVT_CLASS_COL  = "class"


# ─────────────────────────────────────────────────────────────────────────────
# Load reference layers (boundary + grids)
# ─────────────────────────────────────────────────────────────────────────────

def load_boundary() -> gpd.GeoDataFrame:
    gpkg = cfg.CITY_REGIONS / "ghana_admin_levels.gpkg"
    if not gpkg.exists():
        raise FileNotFoundError(f"Admin boundary not found: {gpkg}")
    import fiona
    layers = fiona.listlayers(str(gpkg))
    nat_layer = next((l for l in layers if "0" in l), layers[0])
    gdf = gpd.read_file(gpkg, layer=nat_layer, engine="pyogrio")
    return gdf.to_crs(cfg.GEO_CRS)


def load_grid(size_label: str) -> gpd.GeoDataFrame:
    gpkg = cfg.GRIDS / "ghana_grids.gpkg"
    if not gpkg.exists():
        log.warning("Grid file not found; spatial join step will be skipped.")
        return gpd.GeoDataFrame()
    return gpd.read_file(gpkg, layer=f"grid_{size_label}", engine="pyogrio")


def load_districts() -> gpd.GeoDataFrame:
    gpkg = cfg.CITY_REGIONS / "ghana_admin_levels.gpkg"
    if not gpkg.exists():
        return gpd.GeoDataFrame()
    import fiona
    layers = fiona.listlayers(str(gpkg))
    dist_layer = next((l for l in layers if "2" in l), None)
    if dist_layer is None:
        return gpd.GeoDataFrame()
    gdf = gpd.read_file(gpkg, layer=dist_layer, engine="pyogrio")
    return gdf.to_crs(cfg.GEO_CRS)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry utilities
# ─────────────────────────────────────────────────────────────────────────────

def repair_geometry(geom):
    """Return a valid geometry or None if it cannot be repaired."""
    if geom is None:
        return None
    if geom.is_valid:
        return geom
    try:
        fixed = make_valid(geom)
        # make_valid can return GeometryCollection; keep largest polygon
        if fixed.geom_type == "GeometryCollection":
            polys = [g for g in fixed.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
            if not polys:
                return None
            fixed = max(polys, key=lambda g: g.area)
        return fixed
    except Exception:
        return None


def to_polygon_only(geom):
    """Keep only the largest polygon part of a MultiPolygon."""
    if geom is None:
        return None
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type == "Polygon":
        return geom
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Building metrics (calculated in projected CRS for accuracy)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics_batch(gdf_proj: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compute area, perimeter, and shape metrics for a projected GeoDataFrame.
    Returns a DataFrame with metric columns aligned to gdf_proj's index.
    """
    geoms = gdf_proj.geometry

    area   = geoms.area.values
    perim  = geoms.length.values

    # Avoid division by zero for degenerate geometries
    safe_perim = np.where(perim > 0, perim, np.nan)
    safe_area  = np.where(area > 0,  area,  np.nan)

    compactness  = 4 * pi * safe_area / (safe_perim ** 2)
    shape_index  = safe_perim / (2 * np.sqrt(pi * safe_area))
    log_area     = np.log1p(area)

    # Elongation: ratio of bounding box long side to short side
    bboxes = geoms.bounds   # DataFrame: minx, miny, maxx, maxy
    bb_w   = (bboxes["maxx"] - bboxes["minx"]).values
    bb_h   = (bboxes["maxy"] - bboxes["miny"]).values
    long_s = np.maximum(bb_w, bb_h)
    short_s = np.minimum(bb_w, bb_h)
    elongation = np.where(short_s > 0, long_s / short_s, np.nan)

    return pd.DataFrame(
        {
            "area_m2":      np.round(area,  2),
            "perimeter_m":  np.round(perim, 2),
            "compactness":  np.round(compactness,  4),
            "shape_index":  np.round(shape_index,  4),
            "elongation":   np.round(elongation,   4),
            "log_area_m2":  np.round(log_area,     4),
        },
        index=gdf_proj.index,
    )


def assign_size_class(area_m2: pd.Series) -> pd.Series:
    """Apply provisional size classes from config.SIZE_CLASS_THRESHOLDS_M2."""
    classes = pd.Series("unknown", index=area_m2.index, dtype="object")
    for cls, (lo, hi) in cfg.SIZE_CLASS_THRESHOLDS_M2.items():
        mask = (area_m2 >= lo) & (area_m2 < hi)
        classes[mask] = cls
    return classes


# ─────────────────────────────────────────────────────────────────────────────
# Spatial joins
# ─────────────────────────────────────────────────────────────────────────────

def spatial_join_grid(
    buildings: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    id_col: str,
    out_col: str,
) -> gpd.GeoDataFrame:
    if grid.empty:
        buildings[out_col] = pd.NA
        return buildings

    log.info("  Spatial join: assigning %s …", out_col)
    centroids = buildings.copy()
    centroids["geometry"] = buildings.geometry.centroid

    joined = gpd.sjoin(
        centroids[["geometry"]],
        grid[[id_col, "geometry"]],
        how="left",
        predicate="within",
    )
    buildings[out_col] = joined[id_col].values
    return buildings


def spatial_join_districts(
    buildings: gpd.GeoDataFrame,
    districts: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if districts.empty:
        buildings["district_id"]   = pd.NA
        buildings["district_name"] = pd.NA
        return buildings

    log.info("  Spatial join: assigning district …")
    id_col   = next((c for c in ["gid_2", "GID_2"] if c in districts.columns), None)
    name_col = next((c for c in ["name_2", "NAME_2"] if c in districts.columns), None)

    if id_col is None:
        buildings["district_id"]   = pd.NA
        buildings["district_name"] = pd.NA
        return buildings

    centroids = buildings.copy()
    centroids["geometry"] = buildings.geometry.centroid

    join_cols = [c for c in [id_col, name_col, "geometry"] if c is not None]
    joined = gpd.sjoin(
        centroids[["geometry"]],
        districts[join_cols],
        how="left",
        predicate="within",
    )
    buildings["district_id"]   = joined[id_col].values
    buildings["district_name"] = joined[name_col].values if name_col else pd.NA
    return buildings


# ─────────────────────────────────────────────────────────────────────────────
# QC report
# ─────────────────────────────────────────────────────────────────────────────

class QCStats:
    def __init__(self, source: str):
        self.source          = source
        self.n_raw           = 0
        self.n_outside_ghana = 0
        self.n_invalid_geom  = 0
        self.n_repaired      = 0
        self.n_irrecoverable = 0
        self.n_below_conf    = 0
        self.n_duplicates    = 0
        self.n_tiny          = 0
        self.n_large_flagged = 0
        self.n_final         = 0
        self.area_stats: dict = {}

    def to_dict(self) -> dict:
        return {
            "source":           self.source,
            "timestamp":        datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "n_raw":            self.n_raw,
            "n_outside_ghana":  self.n_outside_ghana,
            "n_invalid_geom":   self.n_invalid_geom,
            "n_repaired":       self.n_repaired,
            "n_irrecoverable":  self.n_irrecoverable,
            "n_below_conf":     self.n_below_conf,
            "n_duplicates":     self.n_duplicates,
            "n_tiny_flagged":   self.n_tiny,
            "n_large_flagged":  self.n_large_flagged,
            "n_final":          self.n_final,
            "pct_retained":     (
                round(100 * self.n_final / self.n_raw, 2) if self.n_raw else 0
            ),
            **self.area_stats,
        }


def write_qc_report(stats_list: list[QCStats]) -> None:
    out = cfg.TABLES / "building_qc_report.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [s.to_dict() for s in stats_list]
    if rows:
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    log.info("QC report → %s", out.relative_to(cfg.PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Source 1 — Google Open Buildings V3
# ─────────────────────────────────────────────────────────────────────────────

def load_obv3_tiles(boundary_geom) -> gpd.GeoDataFrame:
    """
    Reads all downloaded CSV.gz tiles, filters to Ghana, and returns a
    single GeoDataFrame.  Processes in chunks to manage memory.
    """
    tile_dir = cfg.BUILDINGS_V3_RAW / "tiles"
    tiles = sorted(tile_dir.glob("*_buildings.csv.gz"))
    if not tiles:
        raise FileNotFoundError(
            f"No OB V3 tiles found in {tile_dir}\n"
            "Run scripts/01_download_data/download_open_buildings_v3.py first."
        )

    log.info("Loading %d OB V3 tiles …", len(tiles))
    parts: list[gpd.GeoDataFrame] = []

    for tile_path in tqdm(tiles, desc="Reading OB V3 tiles"):
        for chunk in pd.read_csv(
            tile_path,
            compression="gzip",
            chunksize=TILE_CHUNKSIZE,
            usecols=[OBV3_LAT_COL, OBV3_LON_COL, OBV3_AREA_COL,
                     OBV3_CONF_COL, OBV3_GEOMETRY_COL],
        ):
            # Parse WKT geometries
            try:
                gdf = gpd.GeoDataFrame(
                    chunk.drop(columns=[OBV3_GEOMETRY_COL]),
                    geometry=gpd.GeoSeries.from_wkt(chunk[OBV3_GEOMETRY_COL]),
                    crs=cfg.GEO_CRS,
                )
            except Exception as exc:
                log.warning("  Chunk parse error in %s: %s", tile_path.name, exc)
                continue

            # Rough bbox filter before the more expensive within-boundary check
            gdf = gdf[gdf.geometry.notna()]
            xmin, ymin, xmax, ymax = boundary_geom.bounds
            gdf = gdf[
                (gdf.geometry.centroid.x.between(xmin, xmax)) &
                (gdf.geometry.centroid.y.between(ymin, ymax))
            ]
            if not gdf.empty:
                parts.append(gdf)

        gc.collect()

    if not parts:
        raise RuntimeError("No buildings loaded from OB V3 tiles — check tile contents.")

    combined = pd.concat(parts, ignore_index=True)
    log.info("OB V3 raw (bbox filter): %d buildings", len(combined))
    return combined


def clean_obv3(
    boundary: gpd.GeoDataFrame,
    grid_500: gpd.GeoDataFrame,
    grid_1km: gpd.GeoDataFrame,
    districts: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, QCStats]:

    stats = QCStats("google_open_buildings_v3")
    boundary_geom = boundary.union_all()

    # ── Load ─────────────────────────────────────────────────────────────────
    gdf = load_obv3_tiles(boundary_geom)
    stats.n_raw = len(gdf)

    # ── Rename to project schema ──────────────────────────────────────────────
    gdf = gdf.rename(columns={
        OBV3_CONF_COL: "confidence",
        OBV3_AREA_COL: "area_source_m2",   # source-reported area; we recalculate
    })
    gdf["source"] = "gob_v3"

    # ── Confidence filter ─────────────────────────────────────────────────────
    if "confidence" in gdf.columns:
        n_before = len(gdf)
        gdf = gdf[gdf["confidence"] >= MIN_CONFIDENCE]
        stats.n_below_conf = n_before - len(gdf)
        log.info("Confidence filter (>= %.2f): removed %d buildings", MIN_CONFIDENCE, stats.n_below_conf)

    # ── Clip to Ghana boundary ────────────────────────────────────────────────
    # Use pre-computed lat/lon columns (OB V3) to avoid recomputing centroids
    # from WKT (which takes 30–60 min for 16 M geometries).
    # Simplify the boundary to ~500 m tolerance before the point-in-polygon
    # test — Ghana's GADM boundary has tens of thousands of vertices, making
    # contains_xy very slow on 16 M points without simplification.
    log.info("Clipping to Ghana boundary …")
    from shapely import contains_xy, simplify
    boundary_simple = simplify(boundary_geom, tolerance=0.005, preserve_topology=True)
    log.info("  Boundary simplified: %d → %d coords",
             len(boundary_geom.geoms[0].exterior.coords) if boundary_geom.geom_type == "MultiPolygon"
             else len(boundary_geom.exterior.coords),
             len(boundary_simple.geoms[0].exterior.coords) if boundary_simple.geom_type == "MultiPolygon"
             else len(boundary_simple.exterior.coords))
    if OBV3_LAT_COL in gdf.columns and OBV3_LON_COL in gdf.columns:
        cx = gdf[OBV3_LON_COL].values
        cy = gdf[OBV3_LAT_COL].values
    else:
        cx = gdf.geometry.centroid.x.values
        cy = gdf.geometry.centroid.y.values
    inside = contains_xy(boundary_simple, cx, cy)
    stats.n_outside_ghana = int((~inside).sum())
    gdf = gdf[inside].copy()
    log.info("After clip: %d buildings", len(gdf))

    # ── Geometry validation and repair ────────────────────────────────────────
    log.info("Validating and repairing geometries …")
    invalid_mask = ~gdf.geometry.is_valid
    stats.n_invalid_geom = invalid_mask.sum()

    if stats.n_invalid_geom > 0:
        log.info("  Repairing %d invalid geometries …", stats.n_invalid_geom)
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].apply(repair_geometry)
        stats.n_repaired = invalid_mask.sum()

    # Remove unrecoverable
    null_mask = gdf.geometry.isna() | gdf.geometry.apply(
        lambda g: g is None or g.is_empty
    )
    stats.n_irrecoverable = null_mask.sum()
    gdf = gdf[~null_mask]

    # Keep only Polygon / MultiPolygon
    non_poly = ~gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    gdf = gdf[~non_poly]
    log.info("After geometry repair: %d buildings", len(gdf))

    # ── Deduplicate ───────────────────────────────────────────────────────────
    log.info("Removing duplicate geometries …")
    wkb = gdf.geometry.apply(lambda g: g.wkb if g is not None else None)
    dup_mask = wkb.duplicated()
    stats.n_duplicates = dup_mask.sum()
    gdf = gdf[~dup_mask]
    log.info("After dedup: %d buildings (removed %d)", len(gdf), stats.n_duplicates)

    # ── Compute metrics ───────────────────────────────────────────────────────
    log.info("Computing building metrics (reprojecting to %s) …", cfg.PROJ_CRS)
    gdf_proj = gdf.to_crs(cfg.PROJ_CRS)
    metrics  = compute_metrics_batch(gdf_proj)
    for col in metrics.columns:
        gdf[col] = metrics[col].values
    del gdf_proj; gc.collect()

    # ── QC flags ──────────────────────────────────────────────────────────────
    gdf["flag_tiny"]  = gdf["area_m2"] < FLAG_TINY_M2
    gdf["flag_large"] = gdf["area_m2"] >= FLAG_LARGE_M2   # flagged, NOT removed
    stats.n_tiny         = gdf["flag_tiny"].sum()
    stats.n_large_flagged = gdf["flag_large"].sum()

    log.info(
        "QC flags: %d tiny (<%.0f m²), %d large (>=%d m²) — large buildings retained.",
        stats.n_tiny, FLAG_TINY_M2, stats.n_large_flagged, FLAG_LARGE_M2,
    )

    # ── Size class ────────────────────────────────────────────────────────────
    gdf["size_class"] = assign_size_class(gdf["area_m2"])

    # ── Assign unique building IDs ────────────────────────────────────────────
    gdf = gdf.reset_index(drop=True)
    gdf["building_id"] = ["gob_" + str(i).zfill(8) for i in range(len(gdf))]

    # ── Spatial joins ─────────────────────────────────────────────────────────
    gdf = spatial_join_grid(gdf, grid_500, "grid_id", "grid_id_500m")
    gdf = spatial_join_grid(gdf, grid_1km, "grid_id", "grid_id_1km")
    gdf = spatial_join_districts(gdf, districts)

    # ── Area summary stats ────────────────────────────────────────────────────
    stats.area_stats = {
        "area_min_m2":    round(float(gdf["area_m2"].min()),    2),
        "area_median_m2": round(float(gdf["area_m2"].median()), 2),
        "area_mean_m2":   round(float(gdf["area_m2"].mean()),   2),
        "area_p90_m2":    round(float(gdf["area_m2"].quantile(0.90)), 2),
        "area_max_m2":    round(float(gdf["area_m2"].max()),    2),
    }
    stats.n_final = len(gdf)

    return gdf, stats


# ─────────────────────────────────────────────────────────────────────────────
# Source 2 — Overture Maps
# ─────────────────────────────────────────────────────────────────────────────

def load_overture_raw() -> gpd.GeoDataFrame:
    p = cfg.OVERTURE_RAW / "buildings_ghana.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"Overture file not found: {p}\n"
            "Run scripts/01_download_data/download_overture_buildings.py first."
        )

    log.info("Loading Overture buildings from %s …", p.name)
    try:
        gdf = gpd.read_parquet(p)
        # Verify geometry is present and usable
        if not isinstance(gdf, gpd.GeoDataFrame) or gdf.geometry is None:
            raise ValueError("No geometry after read_parquet")
    except Exception:
        # Fallback: geometry stored as WKB bytes in a column named 'geometry'
        # (written by PyArrow ParquetWriter without GeoParquet metadata)
        import pyarrow.parquet as pq
        from shapely import wkb as shapely_wkb
        table = pq.read_table(p)
        df = table.to_pandas()
        geom_col = next(
            (c for c in ("geometry", "geometry_wkb") if c in df.columns), None
        )
        if geom_col is None:
            raise RuntimeError(
                f"No geometry column found in Overture Parquet. "
                f"Columns: {list(df.columns)}"
            )
        geoms = df[geom_col].apply(
            lambda b: shapely_wkb.loads(bytes(b)) if b is not None else None
        )
        df = df.drop(columns=[geom_col])
        gdf = gpd.GeoDataFrame(df, geometry=geoms, crs=cfg.GEO_CRS)

    if gdf.crs is None:
        gdf = gdf.set_crs(cfg.GEO_CRS)
    else:
        gdf = gdf.to_crs(cfg.GEO_CRS)

    log.info("Overture raw: %d buildings, CRS %s", len(gdf), gdf.crs)
    return gdf


def clean_overture(
    boundary: gpd.GeoDataFrame,
    grid_500: gpd.GeoDataFrame,
    grid_1km: gpd.GeoDataFrame,
    districts: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, QCStats]:

    stats = QCStats("overture_maps")
    boundary_geom = boundary.union_all()

    gdf = load_overture_raw()
    stats.n_raw = len(gdf)
    gdf["source"] = "overture"

    # ── Clip ─────────────────────────────────────────────────────────────────
    log.info("Clipping Overture buildings to Ghana …")
    from shapely import contains_xy, simplify
    boundary_simple = simplify(boundary_geom, tolerance=0.005, preserve_topology=True)
    cx = gdf.geometry.centroid.x.values
    cy = gdf.geometry.centroid.y.values
    inside = contains_xy(boundary_simple, cx, cy)
    stats.n_outside_ghana = int((~inside).sum())
    gdf = gdf[inside].copy()
    log.info("After clip: %d buildings", len(gdf))

    # ── Geometry validation ───────────────────────────────────────────────────
    log.info("Validating Overture geometries …")
    invalid_mask = ~gdf.geometry.is_valid
    stats.n_invalid_geom = invalid_mask.sum()
    if stats.n_invalid_geom:
        gdf.loc[invalid_mask, "geometry"] = (
            gdf.loc[invalid_mask, "geometry"].apply(repair_geometry)
        )
        stats.n_repaired = stats.n_invalid_geom

    null_mask = gdf.geometry.isna() | gdf.geometry.apply(
        lambda g: g is None or g.is_empty
    )
    stats.n_irrecoverable = null_mask.sum()
    gdf = gdf[~null_mask]

    # ── Deduplicate ───────────────────────────────────────────────────────────
    wkb = gdf.geometry.apply(lambda g: g.wkb if g else None)
    dup_mask = wkb.duplicated()
    stats.n_duplicates = dup_mask.sum()
    gdf = gdf[~dup_mask]

    # ── Metrics ───────────────────────────────────────────────────────────────
    log.info("Computing Overture metrics …")
    gdf_proj = gdf.to_crs(cfg.PROJ_CRS)
    metrics  = compute_metrics_batch(gdf_proj)
    for col in metrics.columns:
        gdf[col] = metrics[col].values
    del gdf_proj; gc.collect()

    # ── Standardise key attribute columns ────────────────────────────────────
    # Overture height is in metres (may be null for most buildings)
    if OVT_HEIGHT_COL in gdf.columns:
        gdf["height_m"] = pd.to_numeric(gdf[OVT_HEIGHT_COL], errors="coerce")
    else:
        gdf["height_m"] = np.nan

    if OVT_FLOORS_COL in gdf.columns:
        gdf["num_floors"] = pd.to_numeric(gdf[OVT_FLOORS_COL], errors="coerce")
    else:
        gdf["num_floors"] = np.nan

    if OVT_CLASS_COL in gdf.columns:
        gdf["building_class"] = gdf[OVT_CLASS_COL].astype("string")
    else:
        gdf["building_class"] = pd.NA

    # Rough volume estimate where height is known (area × height)
    gdf["volume_m3"] = np.where(
        gdf["height_m"].notna() & (gdf["height_m"] > 0),
        (gdf["area_m2"] * gdf["height_m"]).round(1),
        np.nan,
    )

    # ── QC flags ──────────────────────────────────────────────────────────────
    gdf["flag_tiny"]  = gdf["area_m2"] < FLAG_TINY_M2
    gdf["flag_large"] = gdf["area_m2"] >= FLAG_LARGE_M2
    stats.n_tiny          = gdf["flag_tiny"].sum()
    stats.n_large_flagged = gdf["flag_large"].sum()

    # ── Size class ────────────────────────────────────────────────────────────
    gdf["size_class"] = assign_size_class(gdf["area_m2"])

    # ── Building IDs ─────────────────────────────────────────────────────────
    gdf = gdf.reset_index(drop=True)
    if "id" not in gdf.columns:
        gdf["building_id"] = ["ovt_" + str(i).zfill(8) for i in range(len(gdf))]
    else:
        gdf["building_id"] = gdf["id"].astype(str)

    # ── Spatial joins ─────────────────────────────────────────────────────────
    gdf = spatial_join_grid(gdf, grid_500, "grid_id", "grid_id_500m")
    gdf = spatial_join_grid(gdf, grid_1km, "grid_id", "grid_id_1km")
    gdf = spatial_join_districts(gdf, districts)

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats.area_stats = {
        "area_min_m2":    round(float(gdf["area_m2"].min()),    2),
        "area_median_m2": round(float(gdf["area_m2"].median()), 2),
        "area_mean_m2":   round(float(gdf["area_m2"].mean()),   2),
        "area_p90_m2":    round(float(gdf["area_m2"].quantile(0.90)), 2),
        "area_max_m2":    round(float(gdf["area_m2"].max()),    2),
    }
    stats.n_final = len(gdf)

    return gdf, stats


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_buildings(gdf: gpd.GeoDataFrame, fname: str) -> Path:
    out = cfg.BUILDINGS_CLEAN / fname
    out.parent.mkdir(parents=True, exist_ok=True)

    # Keep geometry column last for GeoParquet compatibility
    cols = [c for c in gdf.columns if c != "geometry"] + ["geometry"]
    gdf[cols].to_parquet(out, engine="pyarrow", index=False)

    size_mb = out.stat().st_size / 1e6
    log.info("Saved → %s  (%.1f MB, %d buildings)", out.name, size_mb, len(gdf))
    return out


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
            "script": "04_clean_building_footprints.py",
            "phase": "3",
            "step": step,
            "status": status,
            "notes": notes,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Clean building footprints — Phase 3")
    parser.add_argument(
        "--source",
        choices=["gob", "overture", "both"],
        default="both",
        help="Which source to process (default: both)",
    )
    args = parser.parse_args()

    log.info("=== Phase 3: Clean Building Footprints ===")

    # Load shared reference layers
    log.info("Loading reference layers …")
    boundary  = load_boundary()
    grid_500  = load_grid("500m")
    grid_1km  = load_grid("1km")
    districts = load_districts()

    all_stats: list[QCStats] = []

    # ── Google Open Buildings V3 ──────────────────────────────────────────────
    if args.source in ("gob", "both"):
        log.info("--- Processing Google Open Buildings V3 ---")
        try:
            gdf_gob, stats_gob = clean_obv3(boundary, grid_500, grid_1km, districts)
            save_buildings(gdf_gob, "gob_v3_ghana.parquet")
            all_stats.append(stats_gob)
            append_processing_log(
                "Clean OB V3", "complete",
                f"{stats_gob.n_final:,} buildings retained "
                f"({stats_gob.to_dict()['pct_retained']}%)"
            )
            del gdf_gob; gc.collect()
        except FileNotFoundError as e:
            log.error("%s\nSkipping OB V3.", e)

    # ── Overture Maps ─────────────────────────────────────────────────────────
    if args.source in ("overture", "both"):
        log.info("--- Processing Overture Maps Buildings ---")
        try:
            gdf_ovt, stats_ovt = clean_overture(boundary, grid_500, grid_1km, districts)
            save_buildings(gdf_ovt, "overture_ghana.parquet")
            all_stats.append(stats_ovt)
            append_processing_log(
                "Clean Overture", "complete",
                f"{stats_ovt.n_final:,} buildings retained "
                f"({stats_ovt.to_dict()['pct_retained']}%)"
            )
            del gdf_ovt; gc.collect()
        except FileNotFoundError as e:
            log.error("%s\nSkipping Overture.", e)

    # ── QC report ─────────────────────────────────────────────────────────────
    if all_stats:
        write_qc_report(all_stats)
        for s in all_stats:
            d = s.to_dict()
            log.info(
                "%-30s  raw=%s  final=%s  retained=%.1f%%  "
                "median_area=%.0f m²",
                d["source"], f"{d['n_raw']:,}", f"{d['n_final']:,}",
                d["pct_retained"], d.get("area_median_m2", 0),
            )

    log.info(
        "Phase 3 complete.\n"
        "  Cleaned files: %s\n"
        "  QC report:     %s\n"
        "Next step: scripts/05_process_temporal_buildings.py",
        cfg.BUILDINGS_CLEAN.relative_to(cfg.PROJECT_ROOT),
        (cfg.TABLES / "building_qc_report.csv").relative_to(cfg.PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
