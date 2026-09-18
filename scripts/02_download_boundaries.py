"""
Phase 2 — Script 02: Download Ghana Administrative Boundaries
Downloads GADM 4.1 GeoPackage for Ghana (all admin levels in one file),
extracts levels 0–2, reprojects to the project CRS, and saves clean
GeoPackage layers to 02_processed_data/city_regions/.

Also writes the urban-region bounding boxes and corridor lookup tables
defined in config.py.

Run after 01_setup_project.py:
    python scripts/02_download_boundaries.py
"""

import csv
import hashlib
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
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


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    """Stream-download *url* to *dest*, showing a progress bar."""
    if dest.exists():
        log.info("Already downloaded: %s", dest.name)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s → %s", url, dest.name)

    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    log.info("Saved: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Download GADM
# ─────────────────────────────────────────────────────────────────────────────

GADM_DEST = cfg.BOUNDARIES_RAW / "gadm41_GHA.gpkg"


def download_gadm() -> Path:
    return download_file(cfg.GADM_GHANA_URL, GADM_DEST)


# ─────────────────────────────────────────────────────────────────────────────
# Process admin levels
# ─────────────────────────────────────────────────────────────────────────────

OUT_GPKG = cfg.CITY_REGIONS / "ghana_admin_levels.gpkg"

LEVEL_NAMES = {
    "ADM_0": "national",
    "ADM_1": "regions",        # 16 regions of Ghana
    "ADM_2": "districts",      # ~261 districts
}

# Column renames to project-standard names
RENAME = {
    "GID_0": "gid_0", "NAME_0": "name_0",
    "GID_1": "gid_1", "NAME_1": "name_1",
    "GID_2": "gid_2", "NAME_2": "name_2",
    "GID_3": "gid_3", "NAME_3": "name_3",
}


def process_admin_levels(gpkg_path: Path) -> dict[str, gpd.GeoDataFrame]:
    """Read GADM layers, reproject, clean columns, return dict of GDFs."""
    import fiona
    available_layers = fiona.listlayers(str(gpkg_path))
    log.info("GADM layers in file: %s", available_layers)

    gdfs = {}
    for layer in available_layers:
        gdf = gpd.read_file(gpkg_path, layer=layer, engine="pyogrio")
        gdf = gdf.rename(columns={k: v for k, v in RENAME.items() if k in gdf.columns})
        # Reproject to project geographic CRS (already 4326 from GADM, but make explicit)
        gdf = gdf.to_crs(cfg.GEO_CRS)
        level_key = layer.lower().replace("gadm41_gha_", "level")
        gdfs[level_key] = gdf
        log.info("  %s → %d features, CRS %s", layer, len(gdf), gdf.crs)

    return gdfs


def save_admin_layers(gdfs: dict[str, gpd.GeoDataFrame]) -> None:
    for layer_name, gdf in gdfs.items():
        gdf.to_file(OUT_GPKG, layer=layer_name, driver="GPKG", engine="pyogrio")
        log.info("Saved layer '%s' → %s", layer_name, OUT_GPKG.name)


# ─────────────────────────────────────────────────────────────────────────────
# Urban-region bounding boxes
# ─────────────────────────────────────────────────────────────────────────────
# Hand-coded approximate bounding boxes (lon_min, lat_min, lon_max, lat_max)
# in EPSG:4326.  These are study-area envelopes, not precise boundaries.
# They will be refined once we have the district layer.

URBAN_REGION_BBOX = {
    "Accra-Tema":         (-0.45,  5.45,  0.10,  5.75),
    "Kumasi":             (-1.75,  6.55, -1.45,  6.80),
    "Sekondi-Takoradi":   (-1.85,  4.85, -1.55,  5.05),
    "Tamale":             (-0.95,  9.30, -0.65,  9.55),
    "Cape Coast":         (-1.35,  5.05, -1.10,  5.20),
    "Koforidua":          (-0.30,  6.05,  0.00,  6.25),
    "Sunyani":            (-2.45,  7.25, -2.25,  7.45),
    "Ho":                 ( 0.40,  6.50,  0.65,  6.75),
    "Wa":                 (-2.60, 10.00, -2.40, 10.15),
    "Bolgatanga":         (-0.95, 10.70, -0.70, 10.90),
    # Corridors
    "Accra-Kasoa":        (-0.45,  5.45,  0.00,  5.75),
    "Accra-Prampram":     (-0.30,  5.50,  0.30,  5.75),
    "Accra-Dodowa":       (-0.30,  5.60,  0.10,  6.00),
    "Kumasi-Ejisu":       (-1.75,  6.55, -1.20,  6.80),
    "Tamale-Savelugu":    (-0.95,  9.30, -0.65,  9.85),
}


def build_urban_region_layer() -> gpd.GeoDataFrame:
    rows = []
    for name, (xmin, ymin, xmax, ymax) in URBAN_REGION_BBOX.items():
        region_type = "corridor" if name in cfg.CORRIDORS else "urban_region"
        rows.append({
            "name": name,
            "type": region_type,
            "lon_min": xmin, "lat_min": ymin,
            "lon_max": xmax, "lat_max": ymax,
            "geometry": box(xmin, ymin, xmax, ymax),
        })
    gdf = gpd.GeoDataFrame(rows, crs=cfg.GEO_CRS)
    return gdf


def save_urban_regions(gdf: gpd.GeoDataFrame) -> None:
    gdf.to_file(OUT_GPKG, layer="urban_regions", driver="GPKG", engine="pyogrio")
    log.info("Saved layer 'urban_regions' → %s  (%d features)", OUT_GPKG.name, len(gdf))

    # Also save as CSV lookup table
    csv_path = cfg.CITY_REGIONS / "urban_region_lookup.csv"
    gdf.drop(columns="geometry").to_csv(csv_path, index=False)
    log.info("Urban region lookup → %s", csv_path.name)


# ─────────────────────────────────────────────────────────────────────────────
# Update data inventory
# ─────────────────────────────────────────────────────────────────────────────

def update_inventory(gpkg_path: Path) -> None:
    inv_path = cfg.DOCS / "metadata" / "data_inventory.csv"
    if not inv_path.exists():
        log.warning("Inventory file not found — run 01_setup_project.py first.")
        return

    rows = []
    with open(inv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row["short_id"] == "gadm_gha":
                row["download_date"] = datetime.utcnow().date().isoformat()
                row["status"] = "downloaded"
                row["notes"] += f" | MD5: {md5(gpkg_path)}"
            rows.append(row)

    with open(inv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Data inventory updated.")


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
            "script": "02_download_boundaries.py",
            "phase": "2",
            "step": step,
            "status": status,
            "notes": notes,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Phase 2: Download and Process Administrative Boundaries ===")

    # 1. Download
    gpkg_path = download_gadm()
    append_processing_log("Download GADM", "complete", f"Path: {gpkg_path}")

    # 2. Process admin levels
    log.info("Processing GADM admin levels …")
    gdfs = process_admin_levels(gpkg_path)
    save_admin_layers(gdfs)
    append_processing_log(
        "Process admin levels", "complete",
        f"Layers: {list(gdfs.keys())}"
    )

    # 3. Build urban-region bounding-box layer
    log.info("Building urban-region layer …")
    ur_gdf = build_urban_region_layer()
    save_urban_regions(ur_gdf)
    append_processing_log("Build urban regions", "complete", f"{len(ur_gdf)} features")

    # 4. Update inventory
    update_inventory(gpkg_path)

    log.info("Boundaries complete. Next step: run scripts/03_prepare_grids.py")


if __name__ == "__main__":
    main()
