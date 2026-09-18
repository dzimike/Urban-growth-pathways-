"""
Phases 7 & 8 — Script 08: Building-Size Classification and Building-Level Metrics

Loads the cleaned Google Open Buildings V3 layer, derives data-driven and
functional size-class thresholds, assigns every building a size class,
computes enriched building-level metrics, and produces the distribution
figures required for Phase 9 (descriptive analysis).

Steps
-----
1.  Load cleaned buildings (gob_v3_ghana.parquet).
2.  Compute the Ghana-wide area distribution and log-transform it.
3.  Derive DATA-DRIVEN thresholds using quintiles (20th, 40th, 60th, 80th pct).
4.  Derive FUNCTIONAL thresholds via visual inspection of the distribution,
    anchored to expected urban morphology:
        very_small  < 25 m²     kiosks, small informal structures
        small       25–65 m²    incremental residential
        medium      65–200 m²   compound / standard house
        large       200–1000 m² institutional, commercial, religious
        very_large  ≥ 1000 m²   warehouse, factory, mall, logistics
5.  Save both threshold sets to 05_documentation/metadata/size_thresholds.json.
6.  Re-assign size_class on the building layer using the DATA-DRIVEN thresholds
    (functional classes stored separately; either can be used for analysis).
7.  Compute enriched building-level metrics not already in the cleaned file:
        volume_m3          area_m2 × height_m  (where height available)
        nearest_poi_type   from OSM POI layer   (placeholder if POIs missing)
        urban_region_id    spatial join to urban-region layer
8.  Save enriched buildings as GeoParquet.
9.  Produce and save distribution figures (histogram, Lorenz curve, boxplots).
10. Save Ghana-wide and city-level building-size summary tables.

Outputs
-------
    02_processed_data/buildings_cleaned/gob_v3_ghana_classified.parquet
    05_documentation/metadata/size_thresholds.json
    04_outputs/figures/building_size_distribution.png
    04_outputs/figures/lorenz_curve.png
    04_outputs/figures/size_class_comparison.png
    04_outputs/tables/building_size_summary_national.csv
    04_outputs/tables/building_size_summary_by_city.csv

Usage
-----
    python scripts/08_calculate_building_metrics.py
    python scripts/08_calculate_building_metrics.py --no-plots
"""

import argparse
import csv
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from math import pi

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

IN_FILE  = cfg.BUILDINGS_CLEAN / "gob_v3_ghana.parquet"
OUT_FILE = cfg.BUILDINGS_CLEAN / "gob_v3_ghana_classified.parquet"
THRESH_FILE = cfg.METADATA / "size_thresholds.json"

FIG_DIR = cfg.FIGURES
TAB_DIR = cfg.TABLES

# Minimum buildings in a city to include in plots
MIN_CITY_BUILDINGS = 5_000

# Functional thresholds (sq m) — starting values; override after inspecting
# the data-driven percentiles once real data is available.
FUNCTIONAL_THRESHOLDS = {
    "very_small": (0,      25),
    "small":      (25,     65),
    "medium":     (65,    200),
    "large":      (200,  1_000),
    "very_large": (1_000, float("inf")),
}


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_buildings() -> gpd.GeoDataFrame:
    if not IN_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned buildings not found: {IN_FILE}\n"
            "Run scripts/04_clean_building_footprints.py first."
        )
    log.info("Loading buildings from %s …", IN_FILE.name)
    gdf = gpd.read_parquet(IN_FILE)
    log.info("  %d buildings loaded, columns: %s", len(gdf), list(gdf.columns))
    return gdf


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


def load_pois() -> gpd.GeoDataFrame:
    """Load cached POIs if available; returns empty GDF otherwise."""
    if cfg.ROADS_RAW.joinpath("ghana_pois.gpkg").exists():
        try:
            import fiona
            layers = fiona.listlayers(str(cfg.ROADS_RAW / "ghana_pois.gpkg"))
            if layers:
                parts = [
                    gpd.read_file(cfg.ROADS_RAW / "ghana_pois.gpkg",
                                  layer=l, engine="pyogrio")
                    .assign(poi_type=l.replace("pois_", ""))
                    for l in layers
                ]
                pois = pd.concat(parts, ignore_index=True)
                pois = gpd.GeoDataFrame(pois, geometry="geometry", crs=cfg.GEO_CRS)
                log.info("POIs loaded: %d features, %d types", len(pois),
                         pois["poi_type"].nunique())
                return pois
        except Exception as exc:
            log.warning("Could not load POIs: %s", exc)
    return gpd.GeoDataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Inequality and distribution metrics (scalar, used for national summary)
# ─────────────────────────────────────────────────────────────────────────────

def gini(x: np.ndarray) -> float:
    x = np.sort(x[np.isfinite(x) & (x > 0)])
    if len(x) < 2:
        return np.nan
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((2 * np.dot(idx, x) / (n * x.sum())) - (n + 1) / n)


def theil_t(x: np.ndarray) -> float:
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 2:
        return np.nan
    mu = x.mean()
    return float(np.mean((x / mu) * np.log(x / mu)))


def shannon_entropy(x: np.ndarray, n_bins: int = 30) -> float:
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 2:
        return np.nan
    log_x = np.log(x)
    hist, _ = np.histogram(log_x, bins=n_bins)
    hist = hist[hist > 0]
    p = hist / hist.sum()
    return float(-np.sum(p * np.log2(p)))


def lorenz_points(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(x[np.isfinite(x) & (x >= 0)])
    n = len(x)
    cumshare_pop  = np.linspace(0, 1, n)
    cumshare_area = np.cumsum(x) / x.sum()
    return cumshare_pop, cumshare_area


# ─────────────────────────────────────────────────────────────────────────────
# Size-class thresholds
# ─────────────────────────────────────────────────────────────────────────────

def compute_percentile_thresholds(area: np.ndarray) -> dict:
    """Compute quintile-based thresholds on the raw area distribution."""
    area_clean = area[np.isfinite(area) & (area > 0)]
    pcts = np.percentile(area_clean, [20, 40, 60, 80])
    return {
        "very_small": (0,          float(pcts[0])),
        "small":      (float(pcts[0]), float(pcts[1])),
        "medium":     (float(pcts[1]), float(pcts[2])),
        "large":      (float(pcts[2]), float(pcts[3])),
        "very_large": (float(pcts[3]), float("inf")),
    }


def assign_size_class(area: np.ndarray, thresholds: dict) -> np.ndarray:
    classes = np.full(len(area), "unknown", dtype=object)
    for label, (lo, hi) in thresholds.items():
        mask = (area >= lo) & (area < hi)
        classes[mask] = label
    return classes.astype(object)


def save_thresholds(pct_thresh: dict, func_thresh: dict) -> None:
    cfg.METADATA.mkdir(parents=True, exist_ok=True)
    payload = {
        "computed":  datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "note": (
            "data_driven thresholds are quintile-based on Ghana-wide area distribution. "
            "functional thresholds are anchored to expected morphological classes and "
            "should be compared and validated against satellite imagery. "
            "Script 09 uses data_driven by default; change --thresholds flag to override."
        ),
        "data_driven": {k: list(v) for k, v in pct_thresh.items()},
        "functional":  {k: list(v) for k, v in func_thresh.items()},
    }
    with open(THRESH_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Thresholds saved → %s", THRESH_FILE.relative_to(cfg.PROJECT_ROOT))


def log_threshold_table(label: str, thresholds: dict, area: np.ndarray) -> None:
    area_clean = area[np.isfinite(area) & (area > 0)]
    log.info("%s thresholds:", label)
    for cls, (lo, hi) in thresholds.items():
        n = ((area_clean >= lo) & (area_clean < hi)).sum()
        pct = 100 * n / len(area_clean)
        hi_str = f"{hi:.0f}" if hi < float("inf") else "∞"
        log.info("  %-12s  %6.0f – %-8s  %8d bldgs  (%.1f%%)", cls, lo, hi_str, n, pct)


# ─────────────────────────────────────────────────────────────────────────────
# Enriched metrics
# ─────────────────────────────────────────────────────────────────────────────

def add_volume(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Estimate volume where height is available (from Overture join or temporal)."""
    if "height_m" not in gdf.columns:
        gdf["height_m"] = np.nan
    if "volume_m3" not in gdf.columns:
        gdf["volume_m3"] = np.where(
            gdf["height_m"].notna() & (gdf["height_m"] > 0),
            (gdf["area_m2"] * gdf["height_m"]).round(1),
            np.nan,
        )
    return gdf


def add_urban_region(
    gdf: gpd.GeoDataFrame,
    urban_regions: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if urban_regions.empty:
        gdf["urban_region_id"]   = pd.NA
        gdf["urban_region_name"] = pd.NA
        return gdf
    log.info("Assigning urban region IDs …")
    cents = gdf.copy()
    cents["geometry"] = gdf.geometry.centroid
    joined = gpd.sjoin(
        cents[["building_id", "geometry"]],
        urban_regions[["name", "geometry"]].reset_index().rename(
            columns={"index": "ur_idx"}
        ),
        how="left",
        predicate="within",
    )
    joined_dedup = joined[~joined.index.duplicated(keep="first")]
    gdf["urban_region_name"] = joined_dedup["name"].reindex(gdf.index).values
    gdf["urban_region_id"] = joined_dedup["ur_idx"].reindex(gdf.index).values
    return gdf


def add_nearest_poi_type(
    gdf: gpd.GeoDataFrame,
    pois: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if pois.empty:
        gdf["nearest_poi_type"]     = pd.NA
        gdf["nearest_poi_dist_m"]   = np.nan
        return gdf
    log.info("Assigning nearest POI type …")
    from scipy.spatial import KDTree
    gdf_proj  = gdf.to_crs(cfg.PROJ_CRS)
    pois_proj = pois.to_crs(cfg.PROJ_CRS)
    bldg_xy   = np.column_stack([gdf_proj.geometry.centroid.x,
                                  gdf_proj.geometry.centroid.y])
    poi_xy    = np.column_stack([pois_proj.geometry.centroid.x,
                                  pois_proj.geometry.centroid.y])
    tree = KDTree(poi_xy)
    dists, idx = tree.query(bldg_xy, workers=-1)
    gdf["nearest_poi_type"]   = pois["poi_type"].iloc[idx].values
    gdf["nearest_poi_dist_m"] = dists.round(0).astype("float32")
    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(
    gdf: gpd.GeoDataFrame,
    pct_thresh: dict,
    func_thresh: dict,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        log.warning("matplotlib not available — skipping figures.")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    area = gdf["area_m2"].values
    area_pos = area[np.isfinite(area) & (area > 0)]

    CLASS_COLORS = {
        "very_small": "#4575b4",
        "small":      "#91bfdb",
        "medium":     "#fee090",
        "large":      "#fc8d59",
        "very_large": "#d73027",
    }

    # ── Figure 1: Log-scale area histogram ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    log_area = np.log10(area_pos)
    axes[0].hist(log_area, bins=120, color="#4575b4", edgecolor="none", alpha=0.8)
    axes[0].set_xlabel("log₁₀(Footprint area, m²)", fontsize=11)
    axes[0].set_ylabel("Number of buildings", fontsize=11)
    axes[0].set_title("Ghana building footprint distribution (log scale)", fontsize=12)
    # Mark percentile thresholds
    for cls, (lo, _) in pct_thresh.items():
        if lo > 0:
            axes[0].axvline(np.log10(lo), color="red", linestyle="--", linewidth=0.8, alpha=0.7)

    # Colour-coded by size class
    for cls, color in CLASS_COLORS.items():
        lo, hi = pct_thresh[cls]
        mask = (area_pos >= lo) & (area_pos < hi)
        axes[1].hist(np.log10(area_pos[mask]), bins=80, color=color,
                     edgecolor="none", alpha=0.75, label=cls)
    axes[1].set_xlabel("log₁₀(Footprint area, m²)", fontsize=11)
    axes[1].set_ylabel("Number of buildings", fontsize=11)
    axes[1].set_title("Buildings by size class (data-driven quintiles)", fontsize=12)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    out1 = FIG_DIR / "building_size_distribution.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure saved: %s", out1.name)

    # ── Figure 2: Lorenz curve ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    sample = area_pos if len(area_pos) <= 500_000 else np.random.choice(area_pos, 500_000, replace=False)
    lp, lc = lorenz_points(sample)
    ax.plot(lp, lc, color="#d73027", linewidth=2, label=f"Ghana (Gini={gini(sample):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="Perfect equality")
    ax.fill_between(lp, lp, lc, alpha=0.15, color="#d73027")
    ax.set_xlabel("Cumulative share of buildings", fontsize=11)
    ax.set_ylabel("Cumulative share of footprint area", fontsize=11)
    ax.set_title("Lorenz curve — building footprint area inequality", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    out2 = FIG_DIR / "lorenz_curve.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure saved: %s", out2.name)

    # ── Figure 3: Threshold comparison ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    y_positions = {"data_driven": 0.7, "functional": 0.3}
    for system_label, thresholds in [("data_driven", pct_thresh), ("functional", func_thresh)]:
        y = y_positions[system_label]
        for cls, (lo, hi) in thresholds.items():
            hi_plot = min(hi, np.percentile(area_pos, 99.5))
            lo_plot = max(lo, 1)
            color = CLASS_COLORS.get(cls, "grey")
            ax.barh(y, np.log10(hi_plot) - np.log10(lo_plot),
                    left=np.log10(lo_plot), height=0.25, color=color, alpha=0.85)
            mid = (np.log10(lo_plot) + np.log10(hi_plot)) / 2
            ax.text(mid, y, cls[:2].upper(), ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")

    ax.set_yticks([0.3, 0.7])
    ax.set_yticklabels(["Functional", "Data-driven (quintiles)"])
    ax.set_xlabel("log₁₀(Footprint area, m²)", fontsize=11)
    ax.set_title("Size-class threshold comparison", fontsize=12)
    patches = [mpatches.Patch(color=c, label=k) for k, c in CLASS_COLORS.items()]
    ax.legend(handles=patches, fontsize=9, loc="upper left")
    out3 = FIG_DIR / "size_class_comparison.png"
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure saved: %s", out3.name)

    # ── Figure 4: City boxplots ───────────────────────────────────────────────
    if "urban_region_name" in gdf.columns:
        city_data = {
            city: gdf.loc[gdf["urban_region_name"] == city, "area_m2"].dropna().values
            for city in gdf["urban_region_name"].dropna().unique()
        }
        city_data = {k: v for k, v in city_data.items() if len(v) >= MIN_CITY_BUILDINGS}
        if city_data:
            fig, ax = plt.subplots(figsize=(14, 6))
            city_order = sorted(city_data.keys(),
                                key=lambda c: np.median(city_data[c]), reverse=True)
            ax.boxplot(
                [np.log10(city_data[c][city_data[c] > 0]) for c in city_order],
                labels=city_order,
                vert=True,
                showfliers=False,
                patch_artist=True,
                boxprops=dict(facecolor="#91bfdb", alpha=0.7),
                medianprops=dict(color="#d73027", linewidth=2),
            )
            ax.set_ylabel("log₁₀(Footprint area, m²)", fontsize=11)
            ax.set_title("Building footprint area distribution by urban region", fontsize=12)
            ax.tick_params(axis="x", rotation=35)
            fig.tight_layout()
            out4 = FIG_DIR / "building_size_by_city.png"
            fig.savefig(out4, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info("Figure saved: %s", out4.name)


# ─────────────────────────────────────────────────────────────────────────────
# Summary tables
# ─────────────────────────────────────────────────────────────────────────────

def national_summary(gdf: gpd.GeoDataFrame, pct_thresh: dict) -> pd.DataFrame:
    area = gdf["area_m2"].values
    area_pos = area[np.isfinite(area) & (area > 0)]
    pcts = np.percentile(area_pos, [10, 25, 50, 75, 90, 95, 99])
    counts = {}
    for cls, (lo, hi) in pct_thresh.items():
        counts[f"n_{cls}"] = int(((area_pos >= lo) & (area_pos < hi)).sum())

    row = {
        "n_buildings":    len(gdf),
        "n_with_area":    int(len(area_pos)),
        "mean_area_m2":   round(float(area_pos.mean()), 2),
        "median_area_m2": round(float(np.median(area_pos)), 2),
        "p10_area_m2":    round(float(pcts[0]), 2),
        "p25_area_m2":    round(float(pcts[1]), 2),
        "p75_area_m2":    round(float(pcts[3]), 2),
        "p90_area_m2":    round(float(pcts[4]), 2),
        "p95_area_m2":    round(float(pcts[5]), 2),
        "p99_area_m2":    round(float(pcts[6]), 2),
        "max_area_m2":    round(float(area_pos.max()), 2),
        "gini":           round(gini(area_pos), 4),
        "theil_t":        round(theil_t(area_pos), 4),
        "entropy":        round(shannon_entropy(area_pos), 4),
        "cv":             round(float(area_pos.std() / area_pos.mean()), 4),
        "p90_p10_ratio":  round(float(pcts[4] / pcts[0]), 2),
        **counts,
    }
    return pd.DataFrame([row])


def city_summary(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    if "urban_region_name" not in gdf.columns:
        return pd.DataFrame()
    rows = []
    for city, sub in gdf.groupby("urban_region_name", dropna=True):
        area = sub["area_m2"].values
        area_pos = area[np.isfinite(area) & (area > 0)]
        if len(area_pos) < 10:
            continue
        row = {
            "urban_region":   city,
            "n_buildings":    len(sub),
            "mean_area_m2":   round(float(area_pos.mean()), 2),
            "median_area_m2": round(float(np.median(area_pos)), 2),
            "p75_area_m2":    round(float(np.percentile(area_pos, 75)), 2),
            "p90_area_m2":    round(float(np.percentile(area_pos, 90)), 2),
            "gini":           round(gini(area_pos), 4),
            "cv":             round(float(area_pos.std() / area_pos.mean()), 4),
        }
        if "size_class" in sub.columns:
            vc = sub["size_class"].value_counts(normalize=True)
            for cls in ["very_small", "small", "medium", "large", "very_large"]:
                row[f"share_{cls}"] = round(float(vc.get(cls, 0)), 4)
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Processing log
# ─────────────────────────────────────────────────────────────────────────────

def append_log(step: str, status: str, notes: str = "") -> None:
    log_path = cfg.PROC_LOG / "processing_log.csv"
    if not log_path.exists():
        return
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "script", "phase", "step", "status", "notes"]
        )
        writer.writerow({
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "script": "08_calculate_building_metrics.py",
            "phase": "7-8",
            "step": step, "status": status, "notes": notes,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Building-size classification and metrics — Phases 7 & 8"
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation")
    parser.add_argument(
        "--thresholds",
        choices=["data_driven", "functional"],
        default="data_driven",
        help="Which threshold system to embed as size_class (default: data_driven)",
    )
    args = parser.parse_args()

    log.info("=== Phases 7 & 8: Building-Size Classification & Metrics ===")

    gdf = load_buildings()
    area = gdf["area_m2"].values

    # ── Compute thresholds ────────────────────────────────────────────────────
    log.info("Computing data-driven (quintile) thresholds …")
    pct_thresh = compute_percentile_thresholds(area)
    log_threshold_table("Data-driven", pct_thresh, area)
    log_threshold_table("Functional",  FUNCTIONAL_THRESHOLDS, area)
    save_thresholds(pct_thresh, FUNCTIONAL_THRESHOLDS)
    append_log("Compute thresholds", "complete", str({k: f"{v[0]:.1f}–{v[1]:.1f}" for k, v in pct_thresh.items()}))

    # ── Assign size class ─────────────────────────────────────────────────────
    chosen = pct_thresh if args.thresholds == "data_driven" else FUNCTIONAL_THRESHOLDS
    log.info("Assigning size_class using '%s' thresholds …", args.thresholds)
    gdf["size_class"]           = assign_size_class(area, chosen)
    gdf["size_class_functional"] = assign_size_class(area, FUNCTIONAL_THRESHOLDS)
    gdf["size_class_data_driven"] = assign_size_class(area, pct_thresh)
    append_log("Assign size class", "complete", f"threshold_system={args.thresholds}")

    # ── Enrich metrics ────────────────────────────────────────────────────────
    log.info("Adding enriched metrics …")
    gdf = add_volume(gdf)
    urban_regions = load_urban_regions()
    pois          = load_pois()
    gdf = add_urban_region(gdf, urban_regions)
    gdf = add_nearest_poi_type(gdf, pois)
    append_log("Enrich metrics", "complete", "volume, urban_region, nearest_poi")

    # ── Save classified buildings ─────────────────────────────────────────────
    log.info("Saving classified buildings …")
    cols = [c for c in gdf.columns if c != "geometry"] + ["geometry"]
    gdf[cols].to_parquet(OUT_FILE, engine="pyarrow", index=False)
    size_mb = OUT_FILE.stat().st_size / 1e6
    log.info("Saved → %s  (%.1f MB, %d buildings)", OUT_FILE.name, size_mb, len(gdf))
    append_log("Save classified buildings", "complete", f"{len(gdf):,} buildings")

    # ── Figures ───────────────────────────────────────────────────────────────
    if not args.no_plots:
        log.info("Generating distribution figures …")
        make_figures(gdf, pct_thresh, FUNCTIONAL_THRESHOLDS)
        append_log("Generate figures", "complete", "histogram, lorenz, comparison, boxplot")

    # ── Summary tables ────────────────────────────────────────────────────────
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    nat = national_summary(gdf, pct_thresh)
    nat.to_csv(TAB_DIR / "building_size_summary_national.csv", index=False)
    log.info("National summary → building_size_summary_national.csv")

    city_df = city_summary(gdf)
    if not city_df.empty:
        city_df.to_csv(TAB_DIR / "building_size_summary_by_city.csv", index=False)
        log.info("City summary → building_size_summary_by_city.csv  (%d cities)", len(city_df))

    append_log("Phase 7-8 complete", "complete", f"output: {OUT_FILE.name}")

    log.info(
        "\nPhase 7 & 8 (building level) complete.\n"
        "  Classified buildings: %s\n"
        "  Thresholds:           %s\n"
        "  Figures:              %s\n"
        "Next: scripts/09_aggregate_to_grid.py",
        OUT_FILE.relative_to(cfg.PROJECT_ROOT),
        THRESH_FILE.relative_to(cfg.PROJECT_ROOT),
        FIG_DIR.relative_to(cfg.PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
