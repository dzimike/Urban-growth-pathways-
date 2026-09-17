"""
Central configuration for the Ghana Building-Size Regimes project.
All scripts import from here — no hard-coded paths or constants elsewhere.
"""

from pathlib import Path

# ── Roots ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATA  = PROJECT_ROOT / "01_raw_data"
PROCESSED = PROJECT_ROOT / "02_processed_data"
ANALYSIS  = PROJECT_ROOT / "03_analysis"
OUTPUTS   = PROJECT_ROOT / "04_outputs"
DOCS      = PROJECT_ROOT / "05_documentation"
SCRIPTS   = PROJECT_ROOT / "scripts"

# ── Sub-directories (raw) ────────────────────────────────────────────────────
BOUNDARIES_RAW    = RAW_DATA / "boundaries"
BUILDINGS_V3_RAW  = RAW_DATA / "open_buildings_v3"
TEMPORAL_RAW      = RAW_DATA / "open_buildings_temporal"
GHSL_RAW          = RAW_DATA / "ghsl"
OVERTURE_RAW      = RAW_DATA / "overture"
CENSUS_RAW        = RAW_DATA / "census"
WORLDPOP_RAW      = RAW_DATA / "worldpop"
ROADS_RAW         = RAW_DATA / "roads_poi_osm"
NIGHT_LIGHTS_RAW  = RAW_DATA / "night_lights"
TERRAIN_RAW       = RAW_DATA / "terrain"

# ── Sub-directories (processed) ──────────────────────────────────────────────
BUILDINGS_CLEAN = PROCESSED / "buildings_cleaned"
GRIDS           = PROCESSED / "grids"
TEMPORAL_PROC   = PROCESSED / "temporal_layers"
GHSL_PROC       = PROCESSED / "ghsl_layers"
COVARIATES      = PROCESSED / "covariates"
CITY_REGIONS    = PROCESSED / "city_regions"
VALIDATION_DIR  = PROCESSED / "validation_samples"

# ── Output sub-directories ───────────────────────────────────────────────────
MAPS         = OUTPUTS / "maps"
FIGURES      = OUTPUTS / "figures"
TABLES       = OUTPUTS / "tables"
DASHBOARD    = OUTPUTS / "dashboard"

# ── Documentation ────────────────────────────────────────────────────────────
DATA_DICT   = DOCS / "data_dictionary"
PROC_LOG    = DOCS / "processing_log"
METADATA    = DOCS / "metadata"

# ── Coordinate Reference Systems ─────────────────────────────────────────────
# Storage / interchange CRS
GEO_CRS = "EPSG:4326"          # WGS 84 geographic

# Primary projected CRS for area and distance calculations.
# UTM Zone 30N covers the bulk of Ghana (west of 0° meridian).
# The easternmost strip (east of 0°) falls in Zone 31N but the error
# for national-level work is acceptable; re-project per city if needed.
PROJ_CRS = "EPSG:32630"        # WGS 84 / UTM Zone 30N

# Equal-area CRS for national aggregate area statistics.
EQUAL_AREA_CRS = "ESRI:102022" # Africa Albers Equal Area Conic

# ── Analytical grids (metres) ────────────────────────────────────────────────
GRID_SIZES = {
    "250m": 250,
    "500m": 500,    # primary modelling unit
    "1km":  1000,
}
PRIMARY_GRID = "500m"

# ── Study regions ────────────────────────────────────────────────────────────
URBAN_REGIONS = [
    "Accra-Tema",
    "Kumasi",
    "Sekondi-Takoradi",
    "Tamale",
    "Cape Coast",
    "Koforidua",
    "Sunyani",
    "Ho",
    "Wa",
    "Bolgatanga",
]

CORRIDORS = [
    "Accra-Kasoa",
    "Accra-Prampram",
    "Accra-Dodowa",
    "Kumasi-Ejisu",
    "Tamale-Savelugu",
]

# ── Building-size classes (sq m) — provisional ───────────────────────────────
# These are starting thresholds for Phase 3 QC flagging only.
# Data-driven thresholds (percentile-based) are computed in Phase 7.
SIZE_CLASS_THRESHOLDS_M2 = {
    "very_small": (0,    20),
    "small":      (20,   60),
    "medium":     (60,   200),
    "large":      (200,  1_000),
    "very_large": (1_000, float("inf")),
}

# Buildings below this area (sq m) are flagged as suspect during QC.
MIN_BUILDING_AREA_M2 = 5.0

# ── Data source URLs ─────────────────────────────────────────────────────────
# GADM 4.1 — Ghana GeoPackage (all admin levels in one file)
GADM_GHANA_URL = (
    "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_GHA.gpkg"
)

# Google Open Buildings V3 index (GeoJSON catalogue; polygons fetched per tile)
OPEN_BUILDINGS_V3_CATALOGUE = (
    "https://openbuildings-public-dot-prod-external-assets"
    ".storage.googleapis.com/public/catalogue.csv"
)

# ── GHSL epochs available ────────────────────────────────────────────────────
GHSL_EPOCHS = [1975, 1990, 2000, 2010, 2015, 2020]

# ── GHSL building-height assets (added for Paper 2 v3 revision) ─────────────
# R2023A release. AGBH = Average Gross Building Height (volumetric density,
# coverage x height). ANBH = Average Net Building Height (height over building
# pixels only). Official identity: AGBH = ANBH x built_fraction at the pixel
# level (JRC GHSL data package documentation).
# NOTE: GHSL_BUILT_H_AGBH_ASSET does not exist on Earth Engine (confirmed;
# see scripts/revision_v3/audit_report.md Section 1) -- kept here only as a
# documented, checked-and-rejected identifier. Both AGBH and ANBH are
# acquired directly from JRC's distribution server instead
# (scripts/revision_v3/01_acquire_ghsl_height.py), not via Earth Engine.
GHSL_BUILT_H_AGBH_ASSET = "JRC/GHSL/P2023A/GHS_BUILT_H_AGBH"  # does not exist on EE
GHSL_BUILT_H_ANBH_ASSET = "JRC/GHSL/P2023A/GHS_BUILT_H_ANBH"  # not used; JRC direct download used instead
GHSL_HEIGHT_EPOCH = 2018          # the only epoch AGBH/ANBH R2023A provides
GHSL_HEIGHT_NATIVE_SCALE_M = 100  # native resolution, both assets

# Built-fraction thresholds for the valid-height sample sensitivity sweep
# (Paper 2 v3, Req. 4). ">0" is encoded as a small epsilon above zero so
# floating-point-exact zero cells are excluded consistently across schemes.
BUILT_FRACTION_THRESHOLDS = [1e-9, 0.005, 0.01, 0.02, 0.05]

# ── Paper 2 v3 revision paths (additive; scripts/revision/ v2 is untouched) ──
REVISION_V3      = SCRIPTS / "revision_v3"
GHSL_LAYERS_V3   = PROCESSED / "ghsl_layers_v3"
PAPER2_V3_OUT    = OUTPUTS / "paper2_v3"
PAPER2_V3_FIG    = PAPER2_V3_OUT / "figures"
PAPER2_V3_TABLES = PAPER2_V3_OUT / "tables"

# ── GDAL tooling (added for Paper 2 v3 revision) ─────────────────────────────
# rasterio and standalone osgeo Python bindings are not installable in this
# environment (no system GDAL, no working gdal-config; the project's own
# declared environment.yml conda env, ghana_buildings, does not exist on
# this machine -- see audit_report.md Section 6). All raster processing in
# the v3 pipeline shells out to QGIS's bundled GDAL command-line tools
# instead. These three constants are the SINGLE place that binds to this
# machine's QGIS install; every v3 script imports them from here rather
# than hard-coding its own copy.
QGIS_GDAL_BIN_DIR = "/Applications/QGIS.app/Contents/MacOS"
QGIS_GDAL_ENV = {
    "DYLD_FALLBACK_LIBRARY_PATH": "/Applications/QGIS.app/Contents/Frameworks",
    "PROJ_LIB": "/Applications/QGIS.app/Contents/Resources/qgis/proj",
    "PROJ_DATA": "/Applications/QGIS.app/Contents/Resources/qgis/proj",
}

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
