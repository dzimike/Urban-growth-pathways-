"""
Phase 1 — Script 01: Project Setup
Verifies the folder structure exists, writes the initial data inventory
template, and produces a CRS decision note in 05_documentation/.

Run once at project initialisation:
    python scripts/01_setup_project.py
"""

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# ── Allow running from the project root without installing the package ────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=cfg.LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Folder structure
# ─────────────────────────────────────────────────────────────────────────────

ALL_DIRS = [
    cfg.BOUNDARIES_RAW, cfg.BUILDINGS_V3_RAW, cfg.TEMPORAL_RAW,
    cfg.GHSL_RAW, cfg.OVERTURE_RAW, cfg.CENSUS_RAW, cfg.WORLDPOP_RAW,
    cfg.ROADS_RAW, cfg.NIGHT_LIGHTS_RAW, cfg.TERRAIN_RAW,
    cfg.BUILDINGS_CLEAN, cfg.GRIDS, cfg.TEMPORAL_PROC, cfg.GHSL_PROC,
    cfg.COVARIATES, cfg.CITY_REGIONS, cfg.VALIDATION_DIR,
    cfg.ANALYSIS / "descriptive_statistics",
    cfg.ANALYSIS / "spatial_autocorrelation",
    cfg.ANALYSIS / "clustering",
    cfg.ANALYSIS / "regression_models",
    cfg.ANALYSIS / "machine_learning",
    cfg.ANALYSIS / "sensitivity_tests",
    cfg.ANALYSIS / "validation",
    cfg.MAPS, cfg.FIGURES, cfg.TABLES, cfg.DASHBOARD,
    cfg.OUTPUTS / "policy_briefs", cfg.OUTPUTS / "manuscript",
    cfg.DATA_DICT, cfg.PROC_LOG, cfg.METADATA,
    cfg.DOCS / "reproducibility_notes", cfg.DOCS / "codebook",
]


def ensure_dirs() -> None:
    created = 0
    for d in ALL_DIRS:
        if not d.exists():
            d.mkdir(parents=True)
            log.info("Created  %s", d.relative_to(cfg.PROJECT_ROOT))
            created += 1
    log.info("Folder check complete — %d new directories created.", created)


# ─────────────────────────────────────────────────────────────────────────────
# Data inventory template
# ─────────────────────────────────────────────────────────────────────────────

INVENTORY_PATH = cfg.DOCS / "metadata" / "data_inventory.csv"

INVENTORY_FIELDS = [
    "dataset_name", "short_id", "source", "url_or_path",
    "format", "crs", "coverage", "temporal_range",
    "licence", "download_date", "file_path_raw",
    "file_path_processed", "notes", "status",
]

INITIAL_DATASETS = [
    {
        "dataset_name": "GADM Ghana Administrative Boundaries",
        "short_id": "gadm_gha",
        "source": "GADM v4.1",
        "url_or_path": cfg.GADM_GHANA_URL,
        "format": "GeoPackage",
        "crs": "EPSG:4326",
        "coverage": "National",
        "temporal_range": "2023",
        "licence": "Academic / non-commercial",
        "download_date": "",
        "file_path_raw": str(cfg.BOUNDARIES_RAW / "gadm41_GHA.gpkg"),
        "file_path_processed": str(cfg.CITY_REGIONS / "ghana_admin_levels.gpkg"),
        "notes": "Levels 0–3: national, region, district, sub-district",
        "status": "pending",
    },
    {
        "dataset_name": "Google Open Buildings V3 Polygons",
        "short_id": "gob_v3",
        "source": "Google / open-buildings",
        "url_or_path": "https://sites.research.google/open-buildings/",
        "format": "GeoParquet (tiled)",
        "crs": "EPSG:4326",
        "coverage": "National (tiled)",
        "temporal_range": "~2023",
        "licence": "CC BY 4.0",
        "download_date": "",
        "file_path_raw": str(cfg.BUILDINGS_V3_RAW),
        "file_path_processed": str(cfg.BUILDINGS_CLEAN / "gob_v3_ghana.parquet"),
        "notes": "Download tiles covering Ghana bounding box",
        "status": "pending",
    },
    {
        "dataset_name": "Google Open Buildings 2.5D Temporal",
        "short_id": "gob_temporal",
        "source": "Google Earth Engine",
        "url_or_path": "projects/sat-io/open-datasets/OPEN-BUILDINGS-TEMPORAL",
        "format": "Cloud Optimized GeoTIFF (raster)",
        "crs": "EPSG:4326",
        "coverage": "National",
        "temporal_range": "2016–2023",
        "licence": "CC BY 4.0",
        "download_date": "",
        "file_path_raw": str(cfg.TEMPORAL_RAW),
        "file_path_processed": str(cfg.TEMPORAL_PROC),
        "notes": "Annual layers: building presence, fractional count, height",
        "status": "pending",
    },
    {
        "dataset_name": "GHSL Built-up Surface (GHS-BUILT-S)",
        "short_id": "ghsl_built_s",
        "source": "European Commission JRC",
        "url_or_path": "https://human-settlement.emergency.copernicus.eu/",
        "format": "GeoTIFF",
        "crs": "ESRI:54009",
        "coverage": "National",
        "temporal_range": "1975, 1990, 2000, 2010, 2015, 2020",
        "licence": "CC BY 4.0",
        "download_date": "",
        "file_path_raw": str(cfg.GHSL_RAW),
        "file_path_processed": str(cfg.GHSL_PROC),
        "notes": "R2023A release, 100 m resolution",
        "status": "pending",
    },
    {
        "dataset_name": "Overture Maps Buildings",
        "short_id": "overture_bldg",
        "source": "Overture Maps Foundation",
        "url_or_path": "https://overturemaps.org/",
        "format": "GeoParquet",
        "crs": "EPSG:4326",
        "coverage": "National",
        "temporal_range": "2024",
        "licence": "ODbL / CDLA Permissive 2.0",
        "download_date": "",
        "file_path_raw": str(cfg.OVERTURE_RAW / "buildings_ghana.parquet"),
        "file_path_processed": str(cfg.BUILDINGS_CLEAN / "overture_ghana.parquet"),
        "notes": "Use overturemaps CLI: overturemaps download --bbox=...",
        "status": "pending",
    },
    {
        "dataset_name": "OpenStreetMap Roads and POIs",
        "short_id": "osm_roads_poi",
        "source": "OpenStreetMap / Geofabrik",
        "url_or_path": "https://download.geofabrik.de/africa/ghana.html",
        "format": "PBF / GeoPackage",
        "crs": "EPSG:4326",
        "coverage": "National",
        "temporal_range": "Current",
        "licence": "ODbL",
        "download_date": "",
        "file_path_raw": str(cfg.ROADS_RAW / "ghana-latest.osm.pbf"),
        "file_path_processed": str(cfg.COVARIATES / "osm_roads.gpkg"),
        "notes": "Also download via osmnx for targeted queries",
        "status": "pending",
    },
    {
        "dataset_name": "WorldPop Ghana 100m Population",
        "short_id": "worldpop_gha",
        "source": "WorldPop / University of Southampton",
        "url_or_path": "https://www.worldpop.org/geodata/country/GH/",
        "format": "GeoTIFF",
        "crs": "EPSG:4326",
        "coverage": "National",
        "temporal_range": "2020",
        "licence": "CC BY 4.0",
        "download_date": "",
        "file_path_raw": str(cfg.WORLDPOP_RAW / "gha_ppp_2020_100m.tif"),
        "file_path_processed": str(cfg.COVARIATES / "worldpop_2020_grid.parquet"),
        "notes": "UN-adjusted constrained individual year",
        "status": "pending",
    },
]


def write_inventory() -> None:
    if INVENTORY_PATH.exists():
        log.info("Data inventory already exists — skipping.")
        return
    with open(INVENTORY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        writer.writerows(INITIAL_DATASETS)
    log.info("Data inventory written → %s", INVENTORY_PATH.relative_to(cfg.PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# CRS decision note
# ─────────────────────────────────────────────────────────────────────────────

CRS_NOTE_PATH = cfg.DOCS / "reproducibility_notes" / "crs_decisions.json"

CRS_NOTE = {
    "created": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "project": "Building-Size Regimes and Urban Growth Pathways in Ghana",
    "decisions": [
        {
            "name": "Geographic CRS",
            "epsg": "EPSG:4326",
            "use": "Data storage, tile download, interchange",
            "rationale": (
                "WGS 84 is the native CRS for all source datasets "
                "(Google Open Buildings, GHSL, OSM). All raw data is retained "
                "in geographic coordinates to avoid double reprojection."
            ),
        },
        {
            "name": "Projected CRS — primary",
            "epsg": "EPSG:32630",
            "use": "Area (m²), perimeter (m), distance calculations for individual buildings",
            "rationale": (
                "WGS 84 / UTM Zone 30N. Ghana spans roughly 3°W–1°E; "
                "UTM 30N (0°–6°W) covers the majority of the country including "
                "Accra, Kumasi, and Takoradi. The easternmost settlements "
                "(Ho, parts of Volta Region) lie in UTM 31N but the area "
                "distortion at national scale is within acceptable bounds for "
                "comparative analysis. City-level studies in the east should "
                "re-project to EPSG:32631 locally."
            ),
        },
        {
            "name": "Equal-area CRS — national aggregates",
            "epsg": "ESRI:102022",
            "use": "National built-up area totals, GHSL aggregation",
            "rationale": (
                "Africa Albers Equal Area Conic preserves area for "
                "continent-wide and national statistics, avoiding the "
                "systematic bias of UTM at the country scale."
            ),
        },
    ],
    "grid_note": (
        "Analytical grids are generated in EPSG:32630 and stored with "
        "centroid coordinates in EPSG:4326 for web mapping compatibility."
    ),
}


def write_crs_note() -> None:
    if CRS_NOTE_PATH.exists():
        log.info("CRS note already exists — skipping.")
        return
    with open(CRS_NOTE_PATH, "w", encoding="utf-8") as f:
        json.dump(CRS_NOTE, f, indent=2)
    log.info("CRS note written → %s", CRS_NOTE_PATH.relative_to(cfg.PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Processing log initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_processing_log() -> None:
    log_path = cfg.PROC_LOG / "processing_log.csv"
    if log_path.exists():
        return
    fields = ["timestamp", "script", "phase", "step", "status", "notes"]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "script": "01_setup_project.py",
            "phase": "1",
            "step": "Project initialisation",
            "status": "complete",
            "notes": "Folder structure, inventory, CRS note created.",
        })
    log.info("Processing log initialised → %s", log_path.relative_to(cfg.PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Phase 1: Project Setup ===")
    ensure_dirs()
    write_inventory()
    write_crs_note()
    init_processing_log()
    log.info("Setup complete. Next step: run scripts/02_download_boundaries.py")


if __name__ == "__main__":
    main()
