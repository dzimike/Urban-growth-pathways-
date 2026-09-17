"""
Paper 2 v3 -- Manuscript figures. Not one of the 15 numbered requirements'
analytical scripts; generates the visuals the manuscript references.

Rewritten to fix issues found in a GIS review of the first version:
  - Plots actual 500 m grid-cell polygons (geopandas, grid_500m.parquet's
    `geometry` column), not centroid scatter points -- true choropleths,
    no marker-overplotting artefacts at dense urban cores.
  - Reprojects to the paper's own analysis CRS (cfg.PROJ_CRS, EPSG:32630 /
    UTM Zone 30N) before plotting, matching Section 4.5's "true equal-area
    (UTM)" methodology, instead of a naive equal-aspect plot of raw WGS84
    degrees.
  - Adds a national boundary overlay (GADM), a scale bar, and a north
    arrow -- all absent from the first version.
  - Fig 1 and Fig 2 now disclose N and the colour-scale cap consistently
    (the first version only disclosed this on Fig 1); both figures also
    explicitly flag that their N (296,677, cells with valid GHSL height
    data) is 9 cells larger than the regression/typology sample (296,668),
    which additionally requires non-missing 1975/2010/2015 built-coverage
    covariates that a univariate height map does not need.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg

FIG = cfg.PAPER2_V3_FIG
FIG.mkdir(parents=True, exist_ok=True)

N_REGRESSION_SAMPLE = 296_668  # regression/typology sample (Sections 4.2, 4.7, 4.9)


def add_scalebar(ax, crs_units_per_km: float = 1000.0, loc: str = "lower left") -> None:
    """Manual scale bar in map data units (metres, since the axes are in
    cfg.PROJ_CRS by the time this is called). Length is chosen as a round
    number close to 20% of the map width."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    width_km = (xmax - xmin) / crs_units_per_km
    candidates_km = [10, 20, 25, 50, 100, 150, 200]
    target = width_km * 0.2
    bar_km = min(candidates_km, key=lambda c: abs(c - target))
    bar_m = bar_km * crs_units_per_km

    if loc == "lower right":
        x0 = xmax - 0.06 * (xmax - xmin) - bar_m
    else:
        x0 = xmin + 0.06 * (xmax - xmin)
    y0 = ymin + 0.04 * (ymax - ymin)
    ax.plot([x0, x0 + bar_m], [y0, y0], color="black", lw=2.0, solid_capstyle="butt", zorder=10)
    for x in (x0, x0 + bar_m):
        ax.plot([x, x], [y0 - 0.006 * (ymax - ymin), y0 + 0.006 * (ymax - ymin)],
                color="black", lw=1.2, zorder=10)
    ax.text(x0 + bar_m / 2, y0 + 0.012 * (ymax - ymin), f"{bar_km} km",
            ha="center", va="bottom", fontsize=7.5, zorder=10)


def add_north_arrow(ax) -> None:
    ax.annotate("N", xy=(0.945, 0.90), xytext=(0.945, 0.82), xycoords="axes fraction",
                textcoords="axes fraction", ha="center", va="center", fontsize=10,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))


def add_region_boxes(ax, regions: gpd.GeoDataFrame) -> None:
    """Overlay the paper's own 15 named urban regions/corridors (Section 3),
    NOT GADM administrative regions -- this study explicitly does not use
    administrative-boundary-derived regions (Section 3), so drawing GADM
    regions here would risk being mistaken for the geography the text
    actually discusses. Cities (`urban_region`) get a solid outline and a
    label; corridors get a lighter dotted outline and no label, since most
    corridors overlap their parent city's box and a second label would only
    duplicate it (e.g. Accra-Kasoa/Accra-Prampram/Accra-Dodowa all extend
    from Accra-Tema)."""
    # NOTE: use regions["type"], not regions.type -- geopandas shadows a
    # column named "type" with its own geometry-type property (returns
    # "Polygon" for every row), so attribute access silently matches nothing.
    cities = regions[regions["type"] == "urban_region"]
    corridors = regions[regions["type"] == "corridor"]
    corridors.plot(ax=ax, facecolor="none", edgecolor="#555555", linewidth=0.5,
                    linestyle=":", zorder=6)
    cities.plot(ax=ax, facecolor="none", edgecolor="#303030", linewidth=0.7, zorder=7)
    for row in cities.itertuples():
        cx = (row.geometry.bounds[0] + row.geometry.bounds[2]) / 2
        ytop = row.geometry.bounds[3]
        ax.text(cx, ytop, row.name, fontsize=5.6, ha="center", va="bottom", zorder=8,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.75))


def load_grid_with(values: pd.DataFrame) -> gpd.GeoDataFrame:
    grid = gpd.read_parquet(cfg.GRIDS / "grid_500m.parquet", columns=["grid_id", "geometry"])
    gdf = grid.merge(values, on="grid_id", how="right")
    return gpd.GeoDataFrame(gdf, geometry="geometry", crs=grid.crs).to_crs(cfg.PROJ_CRS)


boundary = gpd.read_file(cfg.BOUNDARIES_RAW / "gadm41_GHA.gpkg", layer="ADM_ADM_0").to_crs(cfg.PROJ_CRS)
regions = gpd.read_file(cfg.CITY_REGIONS / "ghana_admin_levels.gpkg", layer="urban_regions").to_crs(cfg.PROJ_CRS)

samp = pd.read_parquet(cfg.GHSL_LAYERS_V3 / "sample_definitions.parquet",
                        columns=["grid_id", "cell_agbh_mean_m", "cell_anbh_derived_m", "in_national_footprint"])
built = samp[samp.in_national_footprint].copy()

# ------------------------------------------------------------- Fig 1: ANBH -
sub = load_grid_with(built[built.cell_anbh_derived_m.notna()][["grid_id", "cell_anbh_derived_m"]])
vmax = sub.cell_anbh_derived_m.quantile(0.98)

fig, ax = plt.subplots(figsize=(7.2, 9.2))
sub.plot(column="cell_anbh_derived_m", ax=ax, cmap="YlOrRd", vmin=0, vmax=vmax,
         linewidth=0, antialiased=False)
boundary.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.7, zorder=5)
add_region_boxes(ax, regions)
sm = ScalarMappable(norm=Normalize(vmin=0, vmax=vmax), cmap="YlOrRd")
cb = plt.colorbar(sm, ax=ax, shrink=0.65)
cb.set_label("ANBH, derived (m)\n(darker = greater implied building height)")
ax.set_title("Vertical intensity: ANBH derived from AGBH/built fraction, 2018", fontsize=10.5)
ax.set_axis_off()
ax.set_aspect("equal")
add_scalebar(ax)
add_north_arrow(ax)
note = (f"N = {len(sub):,} cells with valid GHSL height data (national-footprint sample minus "
        f"true GHSL nodata; Section 4.5). {len(sub) - N_REGRESSION_SAMPLE} more than the "
        f"n={N_REGRESSION_SAMPLE:,} regression/typology sample, which additionally requires "
        f"non-missing 1975/2010/2015 built-coverage covariates not needed for this map.\n"
        f"Colour scale capped at the 98th percentile ({vmax:.2f} m) for legibility. "
        f"CRS: {cfg.PROJ_CRS} (UTM Zone 30N).")
fig.text(0.06, 0.015, note, fontsize=6.0, wrap=True)
plt.tight_layout(rect=[0, 0.045, 1, 1])
plt.savefig(FIG / "fig01_anbh_map.png", dpi=200)
plt.close(fig)
print("saved fig01_anbh_map.png")

# ------------------------------------------------------------- Fig 2: AGBH -
sub2 = load_grid_with(built[built.cell_agbh_mean_m.notna()][["grid_id", "cell_agbh_mean_m"]])
vmax2 = sub2.cell_agbh_mean_m.quantile(0.98)

fig, ax = plt.subplots(figsize=(7.2, 9.2))
sub2.plot(column="cell_agbh_mean_m", ax=ax, cmap="YlOrRd", vmin=0, vmax=vmax2,
          linewidth=0, antialiased=False)
boundary.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.7, zorder=5)
add_region_boxes(ax, regions)
sm2 = ScalarMappable(norm=Normalize(vmin=0, vmax=vmax2), cmap="YlOrRd")
cb = plt.colorbar(sm2, ax=ax, shrink=0.65)
cb.set_label("AGBH, mean (m)\n(volumetric height density; robustness outcome)")
ax.set_title("Volumetric vertical density: AGBH, 2018 (robustness outcome)", fontsize=10.5)
ax.set_axis_off()
ax.set_aspect("equal")
add_scalebar(ax)
add_north_arrow(ax)
note2 = (f"N = {len(sub2):,} cells with valid GHSL height data (same sample as Fig. 1; "
         f"{len(sub2) - N_REGRESSION_SAMPLE} more than the n={N_REGRESSION_SAMPLE:,} "
         f"regression/typology sample -- see Fig. 1 note).\n"
         f"Colour scale capped at the 98th percentile ({vmax2:.2f} m) for legibility -- note "
         f"this is a different absolute scale from Fig. 1 (AGBH is a coverage-weighted density, "
         f"not a height, and is much smaller in magnitude; Section 2.3). CRS: {cfg.PROJ_CRS} (UTM Zone 30N).")
fig.text(0.06, 0.010, note2, fontsize=6.0, wrap=True)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(FIG / "fig02_agbh_map.png", dpi=200)
plt.close(fig)
print("saved fig02_agbh_map.png")

# --------------------------------------------------------- Fig 3: typology -
typ = pd.read_parquet(cfg.GHSL_LAYERS_V3 / "typology_v3.parquet", columns=["grid_id", "typology_class"])
typ_gdf = load_grid_with(typ)
colors = {
    "High height / High prior expansion": "#7A0177",
    "High height / Low prior expansion": "#2B8CBE",
    "Low height / High prior expansion": "#F16913",
    "Low height / Low prior expansion": "#D9D9D9",
}
fig, ax = plt.subplots(figsize=(7.2, 9.2))
legend_handles = []
for cls, col in colors.items():
    s = typ_gdf[typ_gdf.typology_class == cls]
    s.plot(ax=ax, color=col, linewidth=0, antialiased=False)
    legend_handles.append(Patch(facecolor=col, label=f"{cls} (n={len(s):,})"))
boundary.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.7, zorder=5)
add_region_boxes(ax, regions)
ax.set_title("Typology: ANBH (height) x growth 1975-2015 (prior expansion)", fontsize=10)
ax.set_axis_off()
ax.set_aspect("equal")
add_scalebar(ax, loc="lower right")
add_north_arrow(ax)
ax.legend(handles=legend_handles, loc="lower left", fontsize=6.3, framealpha=0.9)
note3 = (f"N = {len(typ_gdf):,}. Median split on each axis (ANBH median = 2.500 m; growth 1975-2015 "
         f"median = 0.0005); ties assigned to the low class. CRS: {cfg.PROJ_CRS} (UTM Zone 30N).")
fig.text(0.06, 0.012, note3, fontsize=6.0, wrap=True)
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(FIG / "fig03_typology_map.png", dpi=200)
plt.close(fig)
print("saved fig03_typology_map.png")

# ------------------------------------------------ Fig 4: regression coefs -
ols = pd.read_csv(cfg.PAPER2_V3_TABLES / "T_models_ols.csv")
model_c_anbh = ols[(ols.outcome == "anbh_primary") & (ols.model == "C_full_pre2018") & (ols.variable != "const")]
vif = pd.read_csv(cfg.PAPER2_V3_TABLES / "T_models_vif.csv")
vif_c = vif[(vif.outcome == "anbh_primary") & (vif.model == "C_full_pre2018")].set_index("variable")["vif"].to_dict()

label_map = {
    "log_distance_to_nearest_cbd_m": "Log distance to\nnearest CBD",
    "built_coverage_2010": "Built coverage\n2010",
    "growth_1975_2010": "Growth\n1975-2010",
    "growth_2010_2015": "Growth\n2010-2015",
}
model_c_anbh = model_c_anbh.copy()
model_c_anbh["label"] = model_c_anbh["variable"].map(label_map)
model_c_anbh = model_c_anbh.sort_values("coefficient")

fig, ax = plt.subplots(figsize=(7.2, 4.4))
for i, row in enumerate(model_c_anbh.itertuples()):
    color = "#C0392B" if vif_c.get(row.variable, 0) > 10 else "#2166AC"
    ax.errorbar([row.coefficient], [i], xerr=[1.96 * row.std_error], fmt="o", color=color,
                ecolor=color, elinewidth=2.2, capsize=4, markersize=7, zorder=5)
ax.set_yticks(range(len(model_c_anbh)))
ax.set_yticklabels(model_c_anbh["label"])
ax.axvline(0, color="grey", lw=0.8, ls="--")
ax.set_xlabel("OLS coefficient on log(1+ANBH), 95% CI")
fit = pd.read_csv(cfg.PAPER2_V3_TABLES / "T_models_fit_and_moran.csv")
row_fit = fit[(fit.outcome == "anbh_primary") & (fit.model == "C_full_pre2018")].iloc[0]
ax.set_title(f"Predictors of vertical intensity (log(1+ANBH), pre-2018 only)\n"
             f"n={int(row_fit.n):,}, R²={row_fit.ols_r2:.3f}, all VIF<4", fontsize=10)
handles = [Line2D([0], [0], color="#2166AC", lw=2.2, marker="o", label="VIF ≤ 10"),
           Line2D([0], [0], color="#C0392B", lw=2.2, marker="o", label="VIF > 10")]
ax.legend(handles=handles, loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "fig04_regression_coefficients.png", dpi=200)
plt.close(fig)
print("saved fig04_regression_coefficients.png")

# --------------------------------------------- Fig A2: sample-flow diagram -
# Added in response to Reviewer #4 point 1 / Reviewer #6 (paper2_response_to_
# reviewers_v3.md), which flagged this as requested but not carried into the
# manuscript. Counts hard-coded from config/facts.json's disclosed values
# (Section 4.2) rather than recomputed here, since this is a display of
# already-established, tested numbers, not a new analytical step.
fig, ax = plt.subplots(figsize=(6.2, 8.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 16); ax.axis("off")
box_y = [14.2, 11.0, 7.8, 4.6]
box_text = [
    "Ghana national 500 m grid\nN = 961,858 cells",
    "National-footprint sample\n(Open Buildings V3, building_count > 0)\nN = 296,703 cells",
    "Valid GHSL height data\n(used in Figs. 1-2)\nN = 296,677 cells",
    "Final analytical sample\n(regression, typology: Tables 2-5)\nN = 296,668 cells",
]
box_color = ["#F2F2F2", "#D9E2F3", "#D9E2F3", "#C6E0B4"]
for y, text, color in zip(box_y, box_text, box_color):
    ax.add_patch(plt.Rectangle((5.0 - 3.2, y - 0.9), 6.4, 1.8, facecolor=color, edgecolor="#404040", linewidth=0.9, zorder=2))
    ax.text(5.0, y, text, ha="center", va="center", fontsize=8.3, zorder=3)

arrow_labels = [
    "26 cells excluded:\nno overlapping GHSL raster pixel\n(true nodata, Section 4.5)",
    "9 cells excluded:\nmissing 1975/2010/2015 built coverage\nor CBD-distance covariates",
]
for i in range(3):
    y_top, y_bot = box_y[i] - 0.9, box_y[i + 1] + 0.9
    ax.annotate("", xy=(5.0, y_bot), xytext=(5.0, y_top),
                arrowprops=dict(arrowstyle="-|>", color="#404040", lw=1.3), zorder=1)
    if i > 0:
        ax.text(6.9, (y_top + y_bot) / 2, arrow_labels[i - 1], ha="left", va="center", fontsize=7.0, color="#404040")

# side branch: 15 named regions are a labelled subset of the national-footprint
# sample, not a further sequential exclusion, so drawn to the side rather than below.
side_x, side_y = 8.6, 11.0
ax.annotate("", xy=(side_x - 1.1, side_y), xytext=(5.0 + 3.2, side_y),
            arrowprops=dict(arrowstyle="-|>", color="#808080", lw=1.0, linestyle="dotted"), zorder=1)
ax.add_patch(plt.Rectangle((side_x - 1.1, side_y - 1.05), 2.2, 2.1, facecolor="#FCE4D6", edgecolor="#808080",
                            linewidth=0.8, linestyle="dotted", zorder=2))
ax.text(side_x, side_y, "15 named urban\nregions/corridors\n(Section 3)\nN = 28,133 cells\n(9.5% of national-\nfootprint sample)",
        ha="center", va="center", fontsize=6.6, zorder=3)

ax.set_title("Sample-flow diagram (Section 4.2)", fontsize=10.5, pad=14)
plt.tight_layout()
plt.savefig(FIG / "figA2_sample_flow.png", dpi=200)
plt.close(fig)
print("saved figA2_sample_flow.png")
