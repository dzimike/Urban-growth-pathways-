"""
Phase 8 (Grid Level) — Script 09: Aggregate Buildings to Grid and Assemble Modelling Dataset

Loads the classified building footprints (output of script 08), aggregates
them to the 500 m analytical grid, and left-joins the resulting feature table
with the temporal, GHSL, and covariate layers produced by earlier phases.
The final output is the modelling-ready dataset for all subsequent analysis.

Aggregated building metrics (per grid cell)
-------------------------------------------
    building_count              number of buildings with centroid in cell
    total_footprint_area        sum of footprint area (m²)
    mean/median_building_area   central tendency (m²)
    building_area_std           spread (m²)
    building_area_min/max       range (m²)
    p10_area_m2 / p90_area_m2   tail percentiles (m²)
    coefficient_of_variation    std / mean
    p90_p10_ratio               tail spread ratio
    building_area_gini          Gini coefficient of area inequality
    building_area_theil         Theil-T index of area inequality
    building_area_entropy       Shannon entropy of log-area distribution
    share_very_small … share_very_large   size-class shares (0–1)
    built_coverage_ratio        total_footprint_area / cell_area_m2
    mean_height / median_height mean and median building height (m)
    total_estimated_volume      sum of estimated volumes (m³)

Joined layers
-------------
    temporal_grid_metrics.parquet   Phase 4 — annual building count + height
    ghsl_grid_metrics.parquet       Phase 5 — historical built-up surface
    grid_covariates_500m.parquet    Phase 6/7 — population, roads, POIs, terrain

Outputs
-------
    02_processed_data/covariates/grid_model_ready_500m.parquet        GeoParquet
    02_processed_data/covariates/grid_model_ready_500m_flat.parquet   tabular
    04_outputs/tables/grid_level_summary.csv                          column stats
    05_documentation/metadata/model_column_manifest.csv               data dictionary

Usage
-----
    python scripts/09_aggregate_to_grid.py
    python scripts/09_aggregate_to_grid.py --no-temporal
    python scripts/09_aggregate_to_grid.py --no-ghsl
    python scripts/09_aggregate_to_grid.py --no-covariates
"""

from __future__ import annotations
import argparse
import csv
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=cfg.LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Input paths ───────────────────────────────────────────────────────────────
BUILDINGS_FILE  = cfg.BUILDINGS_CLEAN / "gob_v3_ghana_classified.parquet"
GRID_500M_FILE  = cfg.GRIDS / "grid_500m.parquet"
TEMPORAL_FILE   = cfg.TEMPORAL_PROC / "temporal_grid_metrics.parquet"
GHSL_FILE       = cfg.GHSL_PROC / "ghsl_grid_metrics.parquet"
COVARIATES_FILE = cfg.COVARIATES / "grid_covariates_500m.parquet"

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_GEO      = cfg.COVARIATES / "grid_model_ready_500m.parquet"
OUT_FLAT     = cfg.COVARIATES / "grid_model_ready_500m_flat.parquet"
OUT_SUMMARY  = cfg.TABLES / "grid_level_summary.csv"
OUT_MANIFEST = cfg.METADATA / "model_column_manifest.csv"

SIZE_CLASSES = ["very_small", "small", "medium", "large", "very_large"]

# ── Column descriptions for the manifest ─────────────────────────────────────
_DESCRIPTIONS = {
    "grid_id":                   "Unique grid cell identifier (500 m grid)",
    "cell_area_m2":              "Actual cell area (m²); boundary cells are smaller than 250 000 m²",
    "centroid_lon":              "Grid cell centroid longitude (WGS 84)",
    "centroid_lat":              "Grid cell centroid latitude (WGS 84)",
    "building_count":            "Number of buildings whose centroid falls in this cell",
    "total_footprint_area":      "Sum of building footprint areas (m²)",
    "mean_building_area":        "Mean building footprint area (m²)",
    "median_building_area":      "Median building footprint area (m²)",
    "building_area_std":         "Standard deviation of building footprint area (m²)",
    "building_area_min":         "Minimum building footprint area in cell (m²)",
    "building_area_max":         "Maximum building footprint area in cell (m²)",
    "p10_area_m2":               "10th percentile of building footprint area (m²)",
    "p90_area_m2":               "90th percentile of building footprint area (m²)",
    "coefficient_of_variation":  "CV of building footprint area (std / mean); dimensionless",
    "p90_p10_ratio":             "P90 / P10 ratio; measures spread in the size distribution",
    "building_area_gini":        "Gini coefficient of building footprint area inequality (0–1)",
    "building_area_theil":       "Theil-T index of building footprint area inequality (≥0)",
    "building_area_entropy":     "Shannon entropy of log-area histogram; higher = more diverse sizes",
    "share_very_small":          "Fraction of buildings in very_small size class",
    "share_small":               "Fraction of buildings in small size class",
    "share_medium":              "Fraction of buildings in medium size class",
    "share_large":               "Fraction of buildings in large size class",
    "share_very_large":          "Fraction of buildings in very_large size class",
    "built_coverage_ratio":      "Total footprint area / cell area; ranges 0–1",
    "mean_height":               "Mean building height (m); from 2.5D temporal where available",
    "median_height":             "Median building height (m)",
    "total_estimated_volume":    "Sum of estimated building volumes (m³); area × height",
    "urban_region_name":         "Modal urban region among buildings in this cell",
}


# ─────────────────────────────────────────────────────────────────────────────
# Load helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_buildings() -> pd.DataFrame:
    if not BUILDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Classified buildings not found: {BUILDINGS_FILE}\n"
            "Run scripts/08_calculate_building_metrics.py first."
        )
    log.info("Loading classified buildings …")
    desired = [
        "building_id", "area_m2", "height_m", "volume_m3",
        "size_class", "grid_id_500m", "grid_id_1km", "urban_region_name",
    ]
    try:
        df = pd.read_parquet(BUILDINGS_FILE, columns=desired, engine="pyarrow")
    except Exception:
        df = pd.read_parquet(BUILDINGS_FILE, engine="pyarrow")
        df = df.drop(columns=["geometry"], errors="ignore")

    if "grid_id_500m" not in df.columns and "grid_id" in df.columns:
        df = df.rename(columns={"grid_id": "grid_id_500m"})

    df = df[df["grid_id_500m"].notna()].copy()
    log.info("  %d buildings with valid grid assignment.", len(df))
    return df


def load_grid() -> gpd.GeoDataFrame:
    if not GRID_500M_FILE.exists():
        raise FileNotFoundError(
            f"500 m grid not found: {GRID_500M_FILE}\n"
            "Run scripts/03_prepare_grids.py first."
        )
    log.info("Loading 500 m grid …")
    grid = gpd.read_parquet(GRID_500M_FILE)
    if "grid_id" not in grid.columns:
        for alt in ["grid_id_500m", "id"]:
            if alt in grid.columns:
                grid = grid.rename(columns={alt: "grid_id"})
                break
    grid_proj = grid.to_crs(cfg.PROJ_CRS)
    grid["cell_area_m2"] = grid_proj.geometry.area.round(1).values
    log.info("  %d grid cells loaded.", len(grid))
    return grid


def _detect_key(df: pd.DataFrame) -> str | None:
    for candidate in ("grid_id", "grid_id_500m"):
        if candidate in df.columns:
            return candidate
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Building aggregation — basic statistics
# ─────────────────────────────────────────────────────────────────────────────

def _prep_area(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["grid_id_500m", "area_m2"]]
        .rename(columns={"grid_id_500m": "grid_id"})
        .pipe(lambda d: d[d["area_m2"].notna() & (d["area_m2"] > 0)])
        .copy()
    )


def aggregate_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Count, sums, mean, median, std, range, p10/p90, CV, p90/p10 per cell."""
    log.info("  Basic statistics …")
    area = _prep_area(df)

    agg = (
        area.groupby("grid_id")["area_m2"]
        .agg(
            building_count="count",
            total_footprint_area="sum",
            mean_building_area="mean",
            median_building_area="median",
            building_area_std="std",
            building_area_min="min",
            building_area_max="max",
        )
        .reset_index()
    )

    pct = (
        area.groupby("grid_id")["area_m2"]
        .quantile([0.10, 0.90])
        .unstack(level=-1)
        .rename(columns={0.10: "p10_area_m2", 0.90: "p90_area_m2"})
        .reset_index()
    )
    agg = agg.merge(pct, on="grid_id", how="left")

    agg["coefficient_of_variation"] = np.where(
        agg["mean_building_area"] > 0,
        agg["building_area_std"] / agg["mean_building_area"],
        np.nan,
    )
    agg["p90_p10_ratio"] = np.where(
        agg["p10_area_m2"] > 0,
        agg["p90_area_m2"] / agg["p10_area_m2"],
        np.nan,
    )
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Building aggregation — inequality metrics
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_inequality(df: pd.DataFrame) -> pd.DataFrame:
    """Gini (vectorized), Theil-T (vectorized), Shannon entropy (apply) per cell."""
    log.info("  Inequality metrics (Gini, Theil, entropy) …")
    area = _prep_area(df)

    # ── Gini: rank-sum formula, fully vectorized ──────────────────────────────
    # Sort ascending within each group so rank corresponds to ordered position.
    area_sorted = area.sort_values(["grid_id", "area_m2"])
    area_sorted["_rank"] = area_sorted.groupby("grid_id").cumcount() + 1
    area_sorted["_rank_x"] = area_sorted["_rank"] * area_sorted["area_m2"]

    ineq = area_sorted.groupby("grid_id").agg(
        _n=("area_m2", "count"),
        _sum_x=("area_m2", "sum"),
        _rank_sum=("_rank_x", "sum"),
    )
    ineq["building_area_gini"] = np.where(
        (ineq["_n"] >= 2) & (ineq["_sum_x"] > 0),
        (2 * ineq["_rank_sum"] / (ineq["_n"] * ineq["_sum_x"]))
        - (ineq["_n"] + 1) / ineq["_n"],
        np.nan,
    ).clip(0, 1)

    # ── Theil-T: vectorized via group mean ────────────────────────────────────
    area["_mu"] = area.groupby("grid_id")["area_m2"].transform("mean")
    area["_theil"] = np.where(
        area["_mu"] > 0,
        (area["area_m2"] / area["_mu"]) * np.log(area["area_m2"] / area["_mu"]),
        np.nan,
    )
    theil = (
        area.groupby("grid_id")["_theil"]
        .mean()
        .rename("building_area_theil")
        .clip(lower=0)
    )
    ineq = ineq.join(theil)

    # ── Shannon entropy: apply (per-cell histogram on log-area) ───────────────
    def _cell_entropy(vals: pd.Series) -> float:
        x = vals.values
        if len(x) < 2:
            return np.nan
        log_x = np.log(x)
        hist, _ = np.histogram(log_x, bins=min(20, len(x)))
        hist = hist[hist > 0]
        p = hist / hist.sum()
        return float(-np.sum(p * np.log2(p)))

    entropy = (
        area.groupby("grid_id")["area_m2"]
        .apply(_cell_entropy)
        .rename("building_area_entropy")
    )
    ineq = ineq.join(entropy)

    return (
        ineq[["building_area_gini", "building_area_theil", "building_area_entropy"]]
        .reset_index()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Building aggregation — size-class shares
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_size_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Share of buildings in each size class per cell."""
    log.info("  Size-class shares …")
    sub = (
        df[["grid_id_500m", "size_class"]]
        .rename(columns={"grid_id_500m": "grid_id"})
        .pipe(lambda d: d[d["size_class"].notna()])
        .copy()
    )
    sub["_one"] = 1

    pivot = (
        sub.pivot_table(
            index="grid_id",
            columns="size_class",
            values="_one",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )
    # Ensure every class column is present
    for cls in SIZE_CLASSES:
        if cls not in pivot.columns:
            pivot[cls] = 0

    total = pivot[SIZE_CLASSES].sum(axis=1)
    for cls in SIZE_CLASSES:
        pivot[f"share_{cls}"] = np.where(total > 0, pivot[cls] / total, np.nan)

    return pivot[["grid_id"] + [f"share_{c}" for c in SIZE_CLASSES]]


# ─────────────────────────────────────────────────────────────────────────────
# Building aggregation — height, volume, urban region
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_height_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/median height and total estimated volume per cell."""
    log.info("  Height and volume …")
    sub = df[["grid_id_500m", "height_m", "volume_m3"]].copy()
    sub = sub.rename(columns={"grid_id_500m": "grid_id"})

    parts = []
    if "height_m" in sub.columns:
        h = sub[sub["height_m"].notna() & (sub["height_m"] > 0)]
        if len(h):
            parts.append(
                h.groupby("grid_id")["height_m"]
                .agg(mean_height="mean", median_height="median")
                .reset_index()
            )
    if "volume_m3" in sub.columns:
        v = sub[sub["volume_m3"].notna() & (sub["volume_m3"] > 0)]
        if len(v):
            parts.append(
                v.groupby("grid_id")["volume_m3"]
                .sum()
                .rename("total_estimated_volume")
                .reset_index()
            )

    if not parts:
        return pd.DataFrame(
            columns=["grid_id", "mean_height", "median_height", "total_estimated_volume"]
        )
    result = parts[0]
    for p in parts[1:]:
        result = result.merge(p, on="grid_id", how="outer")
    return result


def aggregate_city_assignment(df: pd.DataFrame) -> pd.DataFrame:
    """Modal urban_region_name per cell (from buildings)."""
    if "urban_region_name" not in df.columns:
        return pd.DataFrame(columns=["grid_id", "urban_region_name"])
    sub = (
        df[["grid_id_500m", "urban_region_name"]]
        .rename(columns={"grid_id_500m": "grid_id"})
        .pipe(lambda d: d[d["urban_region_name"].notna()])
    )
    if sub.empty:
        return pd.DataFrame(columns=["grid_id", "urban_region_name"])

    mode_city = (
        sub.groupby("grid_id")["urban_region_name"]
        .agg(lambda x: x.mode().iloc[0])
        .rename("urban_region_name")
        .reset_index()
    )
    return mode_city


# ─────────────────────────────────────────────────────────────────────────────
# Assemble grid-level building feature table
# ─────────────────────────────────────────────────────────────────────────────

def assemble_building_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all building aggregations and merge results into one table."""
    log.info("Aggregating buildings to 500 m grid …")
    tables = [
        aggregate_basic(df),
        aggregate_inequality(df),
        aggregate_size_shares(df),
        aggregate_height_volume(df),
        aggregate_city_assignment(df),
    ]
    result = tables[0]
    for t in tables[1:]:
        if t is not None and not t.empty:
            result = result.merge(t, on="grid_id", how="outer")
    log.info("  Building features: %d cells × %d columns", len(result), result.shape[1])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Build final GeoDataFrame
# ─────────────────────────────────────────────────────────────────────────────

def attach_to_grid(
    grid: gpd.GeoDataFrame,
    building_features: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Left-join building aggregation onto the full grid (all cells preserved)."""
    log.info("Attaching building features to grid …")
    gdf = grid.merge(building_features, on="grid_id", how="left")

    gdf["built_coverage_ratio"] = np.where(
        gdf["cell_area_m2"] > 0,
        (gdf["total_footprint_area"].fillna(0) / gdf["cell_area_m2"]).clip(0, 1),
        np.nan,
    )
    # Zero-building cells: mark building_count as 0, not NaN
    gdf["building_count"] = gdf["building_count"].fillna(0).astype("int32")
    log.info(
        "  Grid cells with ≥1 building: %d / %d",
        int((gdf["building_count"] > 0).sum()),
        len(gdf),
    )
    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# Join upstream tables
# ─────────────────────────────────────────────────────────────────────────────

def _safe_join(
    gdf: gpd.GeoDataFrame,
    path: Path,
    label: str,
) -> gpd.GeoDataFrame:
    """Load a parquet table and left-join it onto gdf on grid_id."""
    if not path.exists():
        log.warning("%s file not found — skipping: %s", label, path.name)
        return gdf
    log.info("Joining %s …", label)
    incoming = pd.read_parquet(path, engine="pyarrow")
    key = _detect_key(incoming)
    if key is None:
        log.warning("  Cannot identify grid_id column in %s — skipping.", label)
        return gdf
    incoming = incoming.drop(
        columns=[c for c in incoming.columns if c.lower() == "geometry"],
        errors="ignore",
    )
    if key != "grid_id":
        incoming = incoming.rename(columns={key: "grid_id"})
    # Drop columns that already exist in gdf to avoid _x/_y suffixes
    existing = set(gdf.columns) - {"grid_id"}
    incoming = incoming.drop(
        columns=[c for c in incoming.columns if c in existing],
        errors="ignore",
    )
    n_cols_before = len(gdf.columns)
    gdf = gdf.merge(incoming, on="grid_id", how="left")
    log.info(
        "  %s: %d new columns added.", label, len(gdf.columns) - n_cols_before
    )
    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# Summary table and column manifest
# ─────────────────────────────────────────────────────────────────────────────

def write_summary_table(gdf: gpd.GeoDataFrame) -> None:
    """Write per-column descriptive statistics for the final dataset."""
    cfg.TABLES.mkdir(parents=True, exist_ok=True)
    numeric = gdf.select_dtypes(include=["number"])
    rows = []
    for col in numeric.columns:
        v = numeric[col].dropna()
        if len(v) == 0:
            continue
        rows.append(
            {
                "column":    col,
                "n_valid":   int(len(v)),
                "n_missing": int(numeric[col].isna().sum()),
                "mean":      round(float(v.mean()), 4),
                "median":    round(float(v.median()), 4),
                "std":       round(float(v.std()), 4),
                "min":       round(float(v.min()), 4),
                "p25":       round(float(v.quantile(0.25)), 4),
                "p75":       round(float(v.quantile(0.75)), 4),
                "max":       round(float(v.max()), 4),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_SUMMARY, index=False)
    log.info("Grid summary → %s  (%d numeric columns)", OUT_SUMMARY.name, len(rows))


def _infer_source(col: str) -> str:
    building_cols = {
        "building_count", "total_footprint_area", "mean_building_area",
        "median_building_area", "building_area_std", "building_area_min",
        "building_area_max", "p10_area_m2", "p90_area_m2",
        "coefficient_of_variation", "p90_p10_ratio",
        "building_area_gini", "building_area_theil", "building_area_entropy",
        "built_coverage_ratio", "mean_height", "median_height",
        "total_estimated_volume", "urban_region_name",
    }
    if col in building_cols or col.startswith("share_"):
        return "09_aggregate_to_grid"
    if col in {"grid_id", "cell_area_m2", "centroid_lon", "centroid_lat", "geometry"}:
        return "03_prepare_grids"
    if any(k in col for k in ("count_20", "height_20", "growth_rate", "vertical", "peak_growth")):
        return "05_process_temporal_buildings"
    if any(k in col for k in ("built_surface", "ghsl", "age_class", "expansion_phase", "builtup")):
        return "06_process_ghsl"
    if any(k in col for k in ("population", "road", "distance", "poi", "night", "slope", "elevation")):
        return "07_add_population_roads_pois"
    return ""


def write_column_manifest(gdf: gpd.GeoDataFrame) -> None:
    """Write a data dictionary CSV describing every column."""
    cfg.METADATA.mkdir(parents=True, exist_ok=True)
    rows = []
    for col in gdf.columns:
        n_valid   = int(gdf[col].notna().sum())
        n_missing = int(gdf[col].isna().sum())
        rows.append(
            {
                "column":      col,
                "dtype":       str(gdf[col].dtype),
                "n_valid":     n_valid,
                "n_missing":   n_missing,
                "description": _DESCRIPTIONS.get(col, ""),
                "source":      _infer_source(col),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_MANIFEST, index=False)
    log.info(
        "Column manifest → %s  (%d columns)", OUT_MANIFEST.name, len(rows)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Processing log
# ─────────────────────────────────────────────────────────────────────────────

def append_log(step: str, status: str, notes: str = "") -> None:
    log_path = cfg.PROC_LOG / "processing_log.csv"
    if not log_path.exists():
        return
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "script", "phase", "step", "status", "notes"],
        )
        writer.writerow(
            {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "script":    "09_aggregate_to_grid.py",
                "phase":     "8",
                "step":      step,
                "status":    status,
                "notes":     notes,
            }
        )


# ─────────────────────────────────────────────────────────────────────────────
# Validation log — print counts for key modelling columns
# ─────────────────────────────────────────────────────────────────────────────

def log_completeness(gdf: gpd.GeoDataFrame) -> None:
    n = len(gdf)
    key_cols = [
        "building_count", "total_footprint_area", "building_area_gini",
        "share_very_small", "share_very_large", "built_coverage_ratio",
        "mean_height", "total_estimated_volume",
    ]
    log.info("Data completeness (n=%d grid cells):", n)
    for col in key_cols:
        if col in gdf.columns:
            n_valid = int(gdf[col].notna().sum())
            log.info(
                "  %-35s  %7d valid  (%4.1f%%)",
                col, n_valid, 100 * n_valid / n,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate buildings to grid and assemble modelling dataset — Phase 8"
    )
    parser.add_argument(
        "--no-temporal",   action="store_true",
        help="Skip joining temporal building metrics (Phase 4 output)",
    )
    parser.add_argument(
        "--no-ghsl",       action="store_true",
        help="Skip joining GHSL built-up metrics (Phase 5 output)",
    )
    parser.add_argument(
        "--no-covariates", action="store_true",
        help="Skip joining population/road/terrain covariates (Phase 6/7 output)",
    )
    args = parser.parse_args()

    log.info("=== Phase 8 (Grid Level): Aggregate Buildings to Grid ===")

    # ── Load ──────────────────────────────────────────────────────────────────
    buildings = load_buildings()
    grid      = load_grid()

    # ── Aggregate buildings ───────────────────────────────────────────────────
    building_features = assemble_building_features(buildings)
    append_log("Aggregate buildings", "complete",
               f"{int((building_features['building_count'].notna()).sum())} cells with buildings")

    # ── Attach to grid ────────────────────────────────────────────────────────
    gdf = attach_to_grid(grid, building_features)

    # ── Join upstream tables ──────────────────────────────────────────────────
    if not args.no_temporal:
        gdf = _safe_join(gdf, TEMPORAL_FILE, "temporal building metrics")
        append_log("Join temporal", "complete" if TEMPORAL_FILE.exists() else "skipped — file missing")

    if not args.no_ghsl:
        gdf = _safe_join(gdf, GHSL_FILE, "GHSL built-up metrics")
        append_log("Join GHSL", "complete" if GHSL_FILE.exists() else "skipped — file missing")

    if not args.no_covariates:
        gdf = _safe_join(gdf, COVARIATES_FILE, "covariates")
        append_log("Join covariates", "complete" if COVARIATES_FILE.exists() else "skipped — file missing")

    # ── Ensure storage CRS ────────────────────────────────────────────────────
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(cfg.GEO_CRS)

    # ── Completeness report ───────────────────────────────────────────────────
    log_completeness(gdf)
    log.info(
        "Final dataset: %d grid cells × %d columns", len(gdf), len(gdf.columns)
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    cfg.COVARIATES.mkdir(parents=True, exist_ok=True)

    log.info("Saving GeoParquet (with geometry) …")
    gdf.to_parquet(OUT_GEO, engine="pyarrow", index=False)
    log.info("  → %s  (%.1f MB)", OUT_GEO.name, OUT_GEO.stat().st_size / 1e6)

    log.info("Saving flat Parquet (no geometry) …")
    flat = gdf.drop(columns=["geometry"], errors="ignore")
    flat.to_parquet(OUT_FLAT, engine="pyarrow", index=False)
    log.info("  → %s  (%.1f MB)", OUT_FLAT.name, OUT_FLAT.stat().st_size / 1e6)

    append_log("Save outputs", "complete",
               f"{len(gdf)} cells, {len(gdf.columns)} columns")

    # ── Documentation ─────────────────────────────────────────────────────────
    write_summary_table(gdf)
    write_column_manifest(gdf)

    append_log("Phase 8 grid-level complete", "complete",
               f"output: {OUT_GEO.name}")

    log.info(
        "\nPhase 8 (grid level) complete.\n"
        "  Modelling dataset:  %s\n"
        "  Flat parquet:       %s\n"
        "  Column manifest:    %s\n"
        "  Grid summary:       %s\n"
        "Next: scripts/10_descriptive_analysis.py",
        OUT_GEO.relative_to(cfg.PROJECT_ROOT),
        OUT_FLAT.relative_to(cfg.PROJECT_ROOT),
        OUT_MANIFEST.relative_to(cfg.PROJECT_ROOT),
        OUT_SUMMARY.relative_to(cfg.PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
