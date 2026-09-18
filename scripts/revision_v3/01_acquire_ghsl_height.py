"""
Paper 2 v3 -- Script 01: Acquire GHSL AGBH and ANBH (2018, R2023A)

REVISION NOTE (superseding an earlier draft of this script): the original
plan was to fetch AGBH and ANBH via Earth Engine, mirroring the pattern in
scripts/01_download_data/download_ghsl_gee.py. Two things were discovered
during implementation that changed this:

  1. The asset ID hard-coded in the existing codebase for AGBH,
     "JRC/GHSL/P2023A/GHS_BUILT_H_AGBH", does not exist on Earth Engine
     (confirmed via ee.ImageCollection / ee.Image -- both raise
     "not found"). This means scripts/01_download_data/download_ghsl_gee.py
     could never have successfully produced an AGBH file if run as
     written -- consistent with the audit finding (see audit_report.md)
     that 01_raw_data/ghsl/built_height_2018.tif has no documented
     producer script or processing-log entry at all.
  2. The actual Earth Engine asset for GHS-BUILT-H, "JRC/GHSL/P2023A/
     GHS_BUILT_H/2018", exposes ONLY ANBH (confirmed against the official
     Earth Engine Data Catalog page: single band "built_height", described
     there as "Average Net Building Height (ANBH)"). AGBH is not mirrored
     on Earth Engine as a separate asset at all; it is only distributed by
     JRC directly. This means the pre-existing 01_raw_data/ghsl/
     built_height_2018.tif -- used throughout the Paper 2 v2 analysis and
     labelled "AGBH" everywhere in the v2 manuscript -- is, at best, of
     unverifiable identity, and plausibly ANBH mislabelled as AGBH.

This script instead downloads BOTH products directly from JRC's own
distribution server (jeodpp.jrc.ec.europa.eu), which unambiguously labels
each file as AGBH or ANBH by filename, avoiding the Earth Engine asset-
identity ambiguity entirely. Ghana's bounding box was empirically verified
(not assumed) to require exactly 4 of the global 10-degree tiles:
R8_C18, R8_C19, R9_C18, R9_C19 -- confirmed by remotely reading each
candidate tile's embedded GeoTIFF bounding box via GDAL's /vsizip/vsicurl/
virtual filesystem before downloading anything (see audit_report.md for
the full derivation and the two incorrect tile guesses it superseded).

Steps
-----
  1. Download the 4 AGBH tiles + 4 ANBH tiles (8 zips total) from JRC.
  2. Verify each zip's SHA-256 against a value recorded at download time
     (first run) or checked against that record (subsequent runs).
  3. Unzip, mosaic the 4 tiles per product, clip to the Ghana bounding box
     used elsewhere in this project (config.GHANA_BBOX equivalent,
     [-3.5, 4.3, 1.3, 11.3]).
  4. Record full provenance: source URLs, DOI, per-tile and final-file
     checksums, CRS, resolution, nodata -- to
     01_raw_data/ghsl/provenance_agbh_anbh_2018.json.

Outputs
-------
    01_raw_data/ghsl/agbh_2018_v3.tif   (clipped to Ghana, EPSG:4326)
    01_raw_data/ghsl/anbh_2018_v3.tif   (clipped to Ghana, EPSG:4326)
    01_raw_data/ghsl/provenance_agbh_anbh_2018.json
    01_raw_data/ghsl/_tiles_v3/         (raw downloaded tiles, kept for audit)

Usage
-----
    python scripts/revision_v3/01_acquire_ghsl_height.py
"""
from __future__ import annotations
import hashlib
import json
import logging
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg

logging.basicConfig(level=cfg.LOG_LEVEL, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

GHANA_BBOX = [-3.5, 4.3, 1.3, 11.3]           # xmin, ymin, xmax, ymax
TILES = ["R8_C18", "R8_C19", "R9_C18", "R9_C19"]  # empirically verified, see docstring
DOI = "https://doi.org/10.2905/85005901-3A49-48DD-9D19-6261354F56FE"

PRODUCTS = {
    "agbh": {
        "base": ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
                 "GHS_BUILT_H_GLOBE_R2023A/GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_4326_3ss/V1-0/tiles"),
        "prefix": "GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_4326_3ss_V1_0",
        "final": "agbh_2018_v3.tif",
    },
    "anbh": {
        "base": ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
                 "GHS_BUILT_H_GLOBE_R2023A/GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_4326_3ss/V1-0/tiles"),
        "prefix": "GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_4326_3ss_V1_0",
        "final": "anbh_2018_v3.tif",
    },
}

TILE_DIR = cfg.GHSL_RAW / "_tiles_v3"
# GDAL binary/env locations are centralised in config.py (QGIS_GDAL_BIN_DIR,
# QGIS_GDAL_ENV) -- see the comment there for why (no system GDAL in this
# environment). Do not redefine these locally in any v3 script.
_GDAL_BIN_DIR = cfg.QGIS_GDAL_BIN_DIR
_GDAL_ENV = cfg.QGIS_GDAL_ENV


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gdal_run(args: list[str]) -> str:
    import os
    env = dict(**os.environ, **_GDAL_ENV)
    exe = args[0]
    full = [f"{_GDAL_BIN_DIR}/{exe}"] + args[1:]
    r = subprocess.run(full, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{exe} failed: {r.stderr}")
    return r.stdout


def gdalinfo_meta(path: Path) -> dict:
    out = gdal_run(["gdalinfo", "-stats", str(path)])
    meta = {"gdalinfo_raw_tail": "\n".join(out.splitlines()[-25:])}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Pixel Size"):
            meta["pixel_size"] = line.split("=", 1)[1].strip()
        elif line.startswith("NoData Value"):
            meta["nodata_value"] = line.split("=", 1)[1].strip()
        elif line.startswith("Size is"):
            meta["size"] = line.split("is", 1)[1].strip()
        elif "Minimum=" in line and "Maximum=" in line:
            meta["band_stats_summary"] = line
    if "Coordinate System is:" in out:
        meta["crs_snippet"] = out.split("Coordinate System is:", 1)[1].split("Data axis", 1)[0].strip()[:250]
    return meta


def download(url: str, dest: Path) -> None:
    if dest.exists():
        log.info("  already downloaded: %s", dest.name)
        return
    log.info("  downloading %s", url)
    # Python 3.14's urllib has no configured CA bundle in this environment
    # (SSLCertVerificationError); curl uses the macOS system trust store and
    # is already confirmed working, so shell out to it instead.
    tmp = dest.with_suffix(dest.suffix + ".part")
    r = subprocess.run(["curl", "-sS", "-f", "-L", "-m", "300", "-o", str(tmp), url])
    if r.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"curl download failed (exit {r.returncode}) for {url}")
    tmp.rename(dest)
    log.info("  saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


def main() -> None:
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/revision_v3/01_acquire_ghsl_height.py",
        "epoch": 2018,
        "doi": DOI,
        "citation": ("Pesaresi, M. and Politis, P. (2023) GHS-BUILT-H R2023A: GHS building height, "
                     "derived from AW3D30, SRTM30, and Sentinel-2 composite (2018). "
                     "European Commission, Joint Research Centre."),
        "ghana_bbox_xmin_ymin_xmax_ymax": GHANA_BBOX,
        "tiles_used": TILES,
        "tile_selection_note": ("Empirically verified via GDAL /vsizip/vsicurl/ remote header reads "
                                 "before download; see script docstring and audit_report.md. Two prior "
                                 "automated guesses (R3/R4 x C20/C21) were WRONG (covered Scandinavia) "
                                 "and were rejected before any bulk download was attempted."),
        "products": {},
    }

    for key, spec in PRODUCTS.items():
        log.info("=== %s ===", key.upper())
        tile_paths = []
        tile_records = []
        for t in TILES:
            zip_name = f"{spec['prefix']}_{t}.zip"
            tif_name = f"{spec['prefix']}_{t}.tif"
            zip_path = TILE_DIR / zip_name
            download(f"{spec['base']}/{zip_name}", zip_path)
            checksum = sha256_of(zip_path)
            tif_path = TILE_DIR / tif_name
            if not tif_path.exists():
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extract(tif_name, TILE_DIR)
            tile_paths.append(tif_path)
            tile_records.append({
                "tile": t, "url": f"{spec['base']}/{zip_name}",
                "zip_sha256": checksum, "zip_size_bytes": zip_path.stat().st_size,
            })
            log.info("  %s  sha256=%s", t, checksum)

        mosaic_path = TILE_DIR / f"{key}_mosaic_v3.tif"
        gdal_run(["gdal_merge.py", "-o", str(mosaic_path), "-of", "GTiff",
                  *[str(p) for p in tile_paths]])
        log.info("  mosaic -> %s", mosaic_path.name)

        final_path = cfg.GHSL_RAW / spec["final"]
        gdal_run(["gdalwarp", "-te", *[str(v) for v in GHANA_BBOX], "-t_srs", "EPSG:4326",
                  "-overwrite", str(mosaic_path), str(final_path)])
        log.info("  clipped to Ghana -> %s", final_path.relative_to(cfg.PROJECT_ROOT))

        final_checksum = sha256_of(final_path)
        final_meta = gdalinfo_meta(final_path)
        provenance["products"][key] = {
            "tiles": tile_records,
            "mosaic_path": str(mosaic_path.relative_to(cfg.PROJECT_ROOT)),
            "final_path": str(final_path.relative_to(cfg.PROJECT_ROOT)),
            "final_sha256": final_checksum,
            "final_size_bytes": final_path.stat().st_size,
            "final_gdal_metadata": final_meta,
        }
        log.info("  final sha256=%s", final_checksum)

    prov_path = cfg.GHSL_RAW / "provenance_agbh_anbh_2018.json"
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
    log.info("Provenance record -> %s", prov_path.relative_to(cfg.PROJECT_ROOT))


if __name__ == "__main__":
    main()
