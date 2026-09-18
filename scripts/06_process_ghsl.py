"""
Phase 5 — Script 06: Process GHSL Historical Built-Up Surface

Reads GHS-BUILT-S 100 m rasters for epochs 1975–2020, clips them to
Ghana, derives per-pixel built-up age and urban expansion phase, then
aggregates all layers to the 500 m analytical grid.

Processing steps
----------------
1.  Load 100 m rasters for each epoch.
2.  Clip to Ghana national boundary.
3.  Compute built-up coverage ratio per pixel (built_surface_m2 / 10 000).
4.  Build a binary presence stack (built = coverage > threshold).
5.  Derive BUILT-UP AGE CLASS:
        For each 100 m pixel, find the earliest epoch where built = True.
        Pixels never classified as built → 'not_built'.
6.  Derive URBAN EXPANSION PHASE at the 500 m grid level based on the
        dominant age class and the trajectory of built-up coverage.
7.  Aggregate all 100 m layers to 500 m grid cells:
        - built_surface_m2_{epoch}   sum of built surface within cell (m²)
        - built_coverage_{epoch}     built_surface / cell_area_m2
        - dominant_age_class         age class of the majority of 100 m pixels
        - age_class_distribution     share of each age class within cell
        - long_term_growth_1975_2020 coverage(2020) - coverage(1975)
        - recent_growth_2010_2020    coverage(2020) - coverage(2010)
        - urban_expansion_phase      typology string
8.  Save grid-level metrics as GeoParquet.
9.  Save city-level GHSL summary table.

NOTE: Large buildings are retained throughout — they contribute to built
coverage.  Only aggregate metrics change with building size.

Outputs
-------
    02_processed_data/ghsl_layers/ghsl_grid_metrics.parquet
    02_processed_data/ghsl_layers/builtup_age_class_100m.tif
    04_outputs/tables/ghsl_summary_by_city.csv

Usage
-----
    python scripts/06_process_ghsl.py
    python scripts/06_process_ghsl.py --epochs 1975 2000 2020
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
from rasterio import features as rio_features
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject
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

# ── Constants ─────────────────────────────────────────────────────────────────
PIXEL_AREA_M2  = 10_000.0       # 100 m × 100 m
CELL_AREA_M2   = 250_000.0      # 500 m × 500 m  (used for coverage ratio)
BUILT_THRESHOLD = 0.01          # min fraction (1%) of pixel to count as built
NODATA_VAL      = -9999.0

OUT_DIR       = cfg.GHSL_PROC
OUT_METRICS   = OUT_DIR / "ghsl_grid_metrics.parquet"
OUT_AGE_RAST  = OUT_DIR / "builtup_age_class_100m.tif"
OUT_SUMMARY   = cfg.TABLES / "ghsl_summary_by_city.csv"

# Age class integer codes (stored in raster)
AGE_CODE = {
    "pre_1975":   1,
    "1975_1990":  2,
    "1990_2000":  3,
    "2000_2010":  4,
    "2010_2020":  5,
    "not_built":  0,
}
AGE_CODE_INV = {v: k for k, v in AGE_CODE.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Load reference data
# ─────────────────────────────────────────────────────────────────────────────

def load_boundary_geom():
    """Returns the Ghana national boundary as a list of GeoJSON-like dicts."""
    gpkg = cfg.CITY_REGIONS / "ghana_admin_levels.gpkg"
    if not gpkg.exists():
        raise FileNotFoundError(f"Admin boundary not found: {gpkg}")
    import fiona
    layers = fiona.listlayers(str(gpkg))
    nat_layer = next((l for l in layers if "0" in l), layers[0])
    gdf = gpd.read_file(gpkg, layer=nat_layer, engine="pyogrio").to_crs(cfg.GEO_CRS)
    return [geom.__geo_interface__ for geom in gdf.geometry]


def load_grid_500m() -> gpd.GeoDataFrame:
    gpkg = cfg.GRIDS / "ghana_grids.gpkg"
    if not gpkg.exists():
        raise FileNotFoundError(f"Grid not found: {gpkg}")
    return gpd.read_file(gpkg, layer="grid_500m", engine="pyogrio")


def load_urban_regions() -> gpd.GeoDataFrame:
    try:
        import fiona
        gpkg = cfg.CITY_REGIONS / "ghana_admin_levels.gpkg"
        layers = fiona.listlayers(str(gpkg))
        if "urban_regions" in layers:
            return gpd.read_file(gpkg, layer="urban_regions", engine="pyogrio")
    except Exception:
        pass
    return gpd.GeoDataFrame()


def raster_path(prefix: str, epoch: int) -> Path:
    return cfg.GHSL_RAW / f"{prefix}_{epoch}.tif"


def available_epochs(requested: list[int]) -> list[int]:
    avail = [e for e in requested if raster_path("built_surface", e).exists()]
    missing = [e for e in requested if e not in avail]
    if missing:
        log.warning("Missing rasters for epochs: %s — skipping.", missing)
    if not avail:
        raise FileNotFoundError(
            f"No GHSL rasters found in {cfg.GHSL_RAW}\n"
            "Run scripts/01_download_data/download_ghsl_gee.py first."
        )
    log.info("Available epochs: %s", avail)
    return avail


# ─────────────────────────────────────────────────────────────────────────────
# Clip a raster to Ghana and return as numpy array + profile
# ─────────────────────────────────────────────────────────────────────────────

def clip_to_ghana(
    src_path: Path,
    ghana_geoms: list,
) -> tuple[np.ndarray, dict]:
    """
    Returns (data_array, rasterio_profile) after clipping *src_path* to
    the Ghana boundary.  data_array shape: (height, width), dtype float32.
    Nodata pixels are set to NaN.
    """
    with rasterio.open(src_path) as src:
        out_image, out_transform = rio_mask(src, ghana_geoms, crop=True, nodata=NODATA_VAL)
        profile = src.profile.copy()
        profile.update(
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
            dtype="float32",
            nodata=np.nan,
            count=1,
        )
        data = out_image[0].astype("float32")
        data[data == NODATA_VAL] = np.nan
        data[data < 0] = np.nan
    return data, profile


# ─────────────────────────────────────────────────────────────────────────────
# Built-up coverage stack
# ─────────────────────────────────────────────────────────────────────────────

def build_coverage_stack(
    epochs: list[int],
    ghana_geoms: list,
) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """
    Loads each epoch's built_surface raster, clips to Ghana, and returns:
        coverage_stack: ndarray shape (n_epochs, H, W) — fraction [0, 1]
        raw_stacks:     list of ndarray, one per epoch (m² built surface)
        profile:        rasterio profile from the first raster

    Built-up coverage = built_surface_m2 / PIXEL_AREA_M2.
    """
    log.info("Building coverage stack for epochs %s …", epochs)
    coverage_layers: list[np.ndarray] = []
    raw_layers: list[np.ndarray] = []
    profile_ref: dict = {}

    for epoch in tqdm(epochs, desc="Loading GHSL epochs"):
        data, profile = clip_to_ghana(raster_path("built_surface", epoch), ghana_geoms)
        raw_layers.append(data)
        coverage = np.where(np.isfinite(data), data / PIXEL_AREA_M2, np.nan)
        coverage = np.clip(coverage, 0, 1)
        coverage_layers.append(coverage)
        if not profile_ref:
            profile_ref = profile

    stack = np.stack(coverage_layers, axis=0)   # (n_epochs, H, W)
    log.info("Coverage stack shape: %s", stack.shape)
    return stack, raw_layers, profile_ref


# ─────────────────────────────────────────────────────────────────────────────
# Built-up age class raster
# ─────────────────────────────────────────────────────────────────────────────

EPOCH_WINDOWS = [
    ("pre_1975",  None,  1975),
    ("1975_1990", 1975,  1990),
    ("1990_2000", 1990,  2000),
    ("2000_2010", 2000,  2010),
    ("2010_2020", 2010,  2020),
]


def compute_age_class(
    coverage_stack: np.ndarray,
    epochs: list[int],
) -> np.ndarray:
    """
    Returns an integer array (H × W) with AGE_CODE values.

    Logic:
      - If built in 1975 (or first available epoch):  pre_1975
      - Otherwise, assign to the window where it first appears.
      - If not built in any epoch (coverage < threshold): not_built
    """
    log.info("Computing built-up age class …")
    H, W = coverage_stack.shape[1], coverage_stack.shape[2]
    age = np.full((H, W), AGE_CODE["not_built"], dtype="int8")

    # Work backwards through epoch windows; later assignments overwrite earlier
    # so that the final result reflects the *first* epoch of appearance
    for window_label, yr_start, yr_end in reversed(EPOCH_WINDOWS):
        # Find indices in epochs list that fall within (yr_start, yr_end]
        if yr_start is None:
            # pre-1975: built in the earliest available epoch
            idx_list = [i for i, e in enumerate(epochs) if e <= yr_end]
        else:
            idx_list = [i for i, e in enumerate(epochs) if yr_start < e <= yr_end]

        if not idx_list:
            continue

        # A pixel enters this window if:
        #   - it was NOT built in any prior epoch
        #   - it IS built in at least one epoch in this window
        window_stack = coverage_stack[idx_list]   # (k, H, W)
        was_built_in_window = np.any(window_stack >= BUILT_THRESHOLD, axis=0)

        # "Not built before this window" = not built in any earlier epoch
        prior_idx = [i for i, e in enumerate(epochs) if yr_start is not None and e <= yr_start]
        if prior_idx:
            prior_stack = coverage_stack[prior_idx]
            not_built_before = ~np.any(prior_stack >= BUILT_THRESHOLD, axis=0)
        else:
            not_built_before = np.ones((H, W), dtype=bool)

        assign = was_built_in_window & not_built_before
        age[assign] = AGE_CODE[window_label]

    log.info("Age class distribution:")
    for code, label in AGE_CODE_INV.items():
        n = (age == code).sum()
        log.info("  %-15s  %s pixels (%.1f%%)", label, f"{n:,}", 100 * n / age.size)

    return age


def save_age_raster(age: np.ndarray, profile: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = profile.copy()
    p.update(dtype="int8", count=1, nodata=-1, compress="lzw")
    with rasterio.open(OUT_AGE_RAST, "w", **p) as dst:
        dst.write(age[np.newaxis, :, :])
    log.info("Built-up age raster → %s", OUT_AGE_RAST.name)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate to 500 m grid via rasterisation
# ─────────────────────────────────────────────────────────────────────────────

def rasterise_grid(
    grid: gpd.GeoDataFrame,
    profile: dict,
) -> np.ndarray:
    """
    Burns grid cell integer indices into a raster matching *profile*.
    Returns an int32 array (H × W); -1 = outside any cell.
    """
    log.info("Rasterising 500 m grid to 100 m resolution …")
    H, W = profile["height"], profile["width"]
    transform = profile["transform"]
    crs = profile.get("crs", cfg.GEO_CRS)

    grid_repr = grid.to_crs(crs) if grid.crs and str(grid.crs) != str(crs) else grid
    shapes = (
        (geom.__geo_interface__, idx)
        for idx, geom in enumerate(grid_repr.geometry)
    )
    cell_raster = rio_features.rasterize(
        shapes=shapes,
        out_shape=(H, W),
        transform=transform,
        fill=-1,
        dtype="int32",
    )
    n_covered = (cell_raster >= 0).sum()
    log.info(
        "  Grid rasterised: %d/%d pixels assigned to a cell (%.1f%%)",
        n_covered, H * W, 100 * n_covered / (H * W),
    )
    return cell_raster


def aggregate_coverage(
    coverage_stack: np.ndarray,
    raw_stack: list[np.ndarray],
    age_arr: np.ndarray,
    cell_raster: np.ndarray,
    grid: gpd.GeoDataFrame,
    epochs: list[int],
) -> pd.DataFrame:
    """
    Aggregates 100 m raster data to 500 m grid cells.
    Returns a DataFrame indexed by grid_id.
    """
    log.info("Aggregating to 500 m grid (%d cells) …", len(grid))

    n_cells   = len(grid)
    cell_ids  = grid["grid_id"].values
    flat_cells = cell_raster.ravel()
    valid_mask = flat_cells >= 0

    # ── Built surface and coverage per epoch ─────────────────────────────────
    rows: dict[str, np.ndarray] = {}
    for i, epoch in enumerate(tqdm(epochs, desc="Aggregating epochs")):
        flat_bs = raw_stack[i].ravel()    # m² per 100m pixel
        flat_cv = coverage_stack[i].ravel()

        # Sum built surface per cell
        bs_sum = np.zeros(n_cells, dtype="float32")
        cv_sum = np.zeros(n_cells, dtype="float32")
        px_cnt = np.zeros(n_cells, dtype="int32")

        valid_idx = np.where(valid_mask & np.isfinite(flat_bs))[0]
        np.add.at(bs_sum, flat_cells[valid_idx], flat_bs[valid_idx])
        np.add.at(cv_sum, flat_cells[valid_idx], flat_cv[valid_idx])
        np.add.at(px_cnt, flat_cells[valid_idx], 1)

        rows[f"built_surface_m2_{epoch}"] = np.round(bs_sum, 1)
        # Coverage = sum(coverage) / n_100m_pixels_in_cell (each cell has 25)
        with np.errstate(invalid="ignore", divide="ignore"):
            rows[f"built_coverage_{epoch}"] = np.where(
                px_cnt > 0, np.round(cv_sum / px_cnt, 4), np.nan
            )

    # ── Age class distribution per cell ──────────────────────────────────────
    log.info("  Aggregating age classes …")
    flat_age = age_arr.ravel()
    age_counts = {
        label: np.zeros(n_cells, dtype="int32")
        for label in AGE_CODE
    }
    valid_age_idx = np.where(valid_mask)[0]
    for idx in valid_age_idx:
        code = flat_age[idx]
        label = AGE_CODE_INV.get(int(code), "not_built")
        age_counts[label][flat_cells[idx]] += 1

    total_px = np.maximum(sum(age_counts.values()), 1)

    # Dominant age class
    age_arr_2d = np.stack(list(age_counts.values()), axis=1)
    dominant_idx = np.argmax(age_arr_2d, axis=1)
    age_labels = list(age_counts.keys())
    rows["dominant_age_class"] = np.array(age_labels)[dominant_idx]

    for label, counts in age_counts.items():
        rows[f"share_{label}"] = np.round(counts / total_px, 4)

    df = pd.DataFrame(rows, index=cell_ids)
    df.index.name = "grid_id"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Temporal change metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_change_metrics(df: pd.DataFrame, epochs: list[int]) -> pd.DataFrame:
    log.info("Computing change metrics …")

    first, last = epochs[0], epochs[-1]
    cv_first = df[f"built_coverage_{first}"].fillna(0).values
    cv_last  = df[f"built_coverage_{last}"].fillna(0).values

    df["long_term_growth"] = np.round(cv_last - cv_first, 4)

    # Recent growth: 2010 → last available
    yr_2010 = 2010
    if yr_2010 in epochs:
        cv_2010 = df[f"built_coverage_{yr_2010}"].fillna(0).values
        df["recent_growth_2010"] = np.round(cv_last - cv_2010, 4)
    else:
        df["recent_growth_2010"] = np.nan

    # Long-term growth rate (per cent)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["long_term_growth_rate_pct"] = np.where(
            cv_first > 0,
            np.round((cv_last - cv_first) / cv_first * 100, 2),
            np.nan,
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Urban expansion phase
# ─────────────────────────────────────────────────────────────────────────────

def assign_expansion_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns an urban expansion phase to each grid cell based on the
    dominant age class and recent growth trajectory.

    Phases
    ------
    established_urban   : dominant_age_class in {pre_1975, 1975_1990}
                          AND recent_growth < 0.05  (stable core)
    consolidating_urban : dominant_age_class in {1990_2000, 2000_2010}
    active_expansion    : dominant_age_class == 2010_2020
                          OR recent_growth >= 0.10
    emerging_frontier   : share_not_built > 0.5 but recent_growth > 0
    undeveloped         : share_not_built > 0.9
    """
    log.info("Assigning urban expansion phase …")

    dom = df["dominant_age_class"].fillna("not_built")
    share_nb   = df.get("share_not_built", pd.Series(0.0, index=df.index)).fillna(0)
    rec_growth = df.get("recent_growth_2010", pd.Series(0.0, index=df.index)).fillna(0)

    phase = pd.Series("undeveloped", index=df.index, dtype="string")

    phase.loc[share_nb < 0.9] = "sparsely_built"
    phase.loc[dom.isin(["1990_2000", "2000_2010"])] = "consolidating_urban"
    phase.loc[dom.isin(["pre_1975", "1975_1990"])] = "established_urban"
    phase.loc[dom == "2010_2020"] = "active_expansion"
    phase.loc[(share_nb > 0.5) & (rec_growth > 0.01)] = "emerging_frontier"
    phase.loc[share_nb > 0.9] = "undeveloped"

    df["urban_expansion_phase"] = phase

    log.info("Expansion phase distribution:")
    for ph, n in phase.value_counts().items():
        log.info("  %-25s  %s cells (%.1f%%)", ph, f"{n:,}", 100 * n / len(df))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# City-level summary
# ─────────────────────────────────────────────────────────────────────────────

def build_city_summary(
    metrics: pd.DataFrame,
    grid: gpd.GeoDataFrame,
    urban_regions: gpd.GeoDataFrame,
    epochs: list[int],
) -> pd.DataFrame:
    if urban_regions.empty:
        return pd.DataFrame()

    log.info("Building city-level GHSL summary …")
    grid_m = grid.set_index("grid_id")[["geometry"]].join(metrics)

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
    for region, sub in grid_m.groupby("city_region", dropna=True):
        row = {"city_region": region, "n_cells": len(sub)}
        for epoch in epochs:
            col = f"built_coverage_{epoch}"
            if col in sub:
                row[f"mean_built_coverage_{epoch}"] = round(sub[col].mean(skipna=True), 4)
                row[f"total_built_surface_km2_{epoch}"] = round(
                    sub[f"built_surface_m2_{epoch}"].sum() / 1e6, 2
                )
        if "long_term_growth" in sub:
            row["mean_long_term_growth"]  = round(sub["long_term_growth"].mean(), 4)
        if "recent_growth_2010" in sub:
            row["mean_recent_growth"] = round(sub["recent_growth_2010"].mean(), 4)
        rows.append(row)

    return pd.DataFrame(rows)


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
            "script": "06_process_ghsl.py",
            "phase": "5",
            "step": step,
            "status": status,
            "notes": notes,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Process GHSL built-up surface — Phase 5")
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=cfg.GHSL_EPOCHS,
        help="Epochs to process (default: 1975 1990 2000 2010 2015 2020)",
    )
    args = parser.parse_args()

    log.info("=== Phase 5: Process GHSL Historical Built-Up Surface ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Check which epochs are actually available ────────────────────────────
    epochs = available_epochs(args.epochs)

    # ── Load reference layers ────────────────────────────────────────────────
    ghana_geoms   = load_boundary_geom()
    grid          = load_grid_500m()
    urban_regions = load_urban_regions()

    # ── Build coverage stack ─────────────────────────────────────────────────
    coverage_stack, raw_stack, profile = build_coverage_stack(epochs, ghana_geoms)
    append_processing_log("Build coverage stack", "complete",
                          f"epochs={epochs}, shape={coverage_stack.shape}")

    # ── Built-up age class at 100 m ──────────────────────────────────────────
    age_arr = compute_age_class(coverage_stack, epochs)
    save_age_raster(age_arr, profile)
    append_processing_log("Compute age class", "complete", f"raster: {OUT_AGE_RAST.name}")

    # ── Rasterise the 500 m grid ─────────────────────────────────────────────
    cell_raster = rasterise_grid(grid, profile)
    append_processing_log("Rasterise grid", "complete", f"{(cell_raster >= 0).sum():,} pixels assigned")

    # ── Aggregate to 500 m ───────────────────────────────────────────────────
    metrics = aggregate_coverage(coverage_stack, raw_stack, age_arr, cell_raster, grid, epochs)
    del coverage_stack, raw_stack, age_arr, cell_raster; gc.collect()
    append_processing_log("Aggregate to 500m grid", "complete", f"{len(metrics):,} cells")

    # ── Change metrics ───────────────────────────────────────────────────────
    metrics = compute_change_metrics(metrics, epochs)

    # ── Urban expansion phase ────────────────────────────────────────────────
    metrics = assign_expansion_phase(metrics)
    append_processing_log("Assign expansion phase", "complete", "")

    # ── Print highlights ─────────────────────────────────────────────────────
    if f"built_coverage_{epochs[-1]}" in metrics:
        mean_cov_2020 = metrics[f"built_coverage_{epochs[-1]}"].mean(skipna=True)
        total_built_km2 = metrics[f"built_surface_m2_{epochs[-1]}"].sum() / 1e6
        mean_ltg = metrics["long_term_growth"].mean(skipna=True)
        log.info(
            "Ghana-wide summary (%d):\n"
            "  Mean built coverage:          %.3f\n"
            "  Total built surface:          %.0f km²\n"
            "  Mean long-term growth (%d→%d): %.3f",
            epochs[-1], mean_cov_2020, total_built_km2,
            epochs[0], epochs[-1], mean_ltg,
        )

    # ── Save metrics ─────────────────────────────────────────────────────────
    metrics.reset_index().to_parquet(OUT_METRICS, engine="pyarrow", index=False)
    log.info(
        "GHSL metrics → %s  (%d cells, %d columns)",
        OUT_METRICS.relative_to(cfg.PROJECT_ROOT), len(metrics), len(metrics.columns),
    )

    # ── City summary ─────────────────────────────────────────────────────────
    city_df = build_city_summary(metrics, grid, urban_regions, epochs)
    if not city_df.empty:
        cfg.TABLES.mkdir(parents=True, exist_ok=True)
        city_df.to_csv(OUT_SUMMARY, index=False)
        log.info(
            "City summary → %s  (%d regions)",
            OUT_SUMMARY.relative_to(cfg.PROJECT_ROOT), len(city_df),
        )

    append_processing_log("Phase 5 complete", "complete",
                          f"metrics: {OUT_METRICS.name}")

    log.info(
        "\nPhase 5 complete.\n"
        "  GHSL grid metrics:  %s\n"
        "  Age-class raster:   %s\n"
        "  City summary:       %s\n"
        "Next step: scripts/07_add_population_roads_pois.py",
        OUT_METRICS.relative_to(cfg.PROJECT_ROOT),
        OUT_AGE_RAST.relative_to(cfg.PROJECT_ROOT),
        OUT_SUMMARY.relative_to(cfg.PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
