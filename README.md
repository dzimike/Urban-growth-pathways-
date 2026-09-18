# Ghana Building Height Analysis (2018)

Analysis code for: **"Building height and horizontal expansion reveal distinct urban-form regimes in Ghana"** (Michael Gameli Dziwornu, Michael Mensah). Citation details will be added here once the paper is published.

This repository is the code deposit accompanying that manuscript. It contains every script that produced a number, table, or figure in the paper — both the v3-specific analysis and the upstream pipeline that builds the national grid and covariates it runs on — plus the identity/range test suite and the consolidated audit trail. It does **not** contain raw or processed data — see [Data](#data) below for why, and where to get it.

## Scope

Two layers of code, in two different `scripts/` locations, requiring two different Python environments (see [Requirements](#requirements)):

1. **`scripts/` (top level)** — the upstream pipeline: acquires and cleans building footprints and covariates, builds the national 500 m/250 m analytical grids, and aggregates everything to `grid_500m.parquet` and `grid_model_ready_500m.parquet`. Written for, and part of, the wider multi-paper Ghana building-size project this analysis draws its grid and covariates from.
2. **`scripts/revision_v3/`** — the paper-specific analysis: ANBH (Average Net Building Height) as the primary vertical outcome, AGBH (Average Gross Building Height) as a volumetric-density robustness check, for the 2018 GHSL epoch. It supersedes an earlier internal revision built around AGBH alone; that history is narrated in the paper itself (Sections 2.3, 4.3) and in `scripts/revision_v3/audit_report.md`.

## Data

Raw inputs are third-party datasets with their own licenses and are not redistributed here:

- GHSL GHS-BUILT-H (AGBH/ANBH) and GHS-BUILT-S (built-up surface) — JRC
- Google Open Buildings V3 (footprint polygons) and Open Buildings 2.5D Temporal
- Overture Maps buildings and places
- OpenStreetMap roads and POIs
- Ghana 2021 Population and Housing Census
- WorldPop gridded population
- GADM Ghana administrative boundaries
- DEM/terrain and night-time lights layers (used by the upstream pipeline; not part of the v3-specific analysis)

Sources, DOIs, and access dates for the layers the v3 analysis actually uses are listed in the paper's Table 1 and in `scripts/revision_v3/01_acquire_ghsl_height.py`'s provenance output; the upstream scripts document their own sources in-line (see each script's docstring).

**This repository still requires you to acquire that raw data yourself before anything will run.** What changed from the previous version of this README: the upstream pipeline's *code* is now included (previously it wasn't, and the reproducibility claim below was narrower as a result), so every script between raw data and the paper's final tables and figures is now visible and checkable. Two setup steps remain on you: obtaining the raw datasets above, and installing the two separate environments described next.

## Requirements

**Two separate environments, because the upstream and v3 scripts were developed under different GDAL toolchains:**

- **Upstream pipeline** (`scripts/`, top level): `conda env create -f environment.yml` (declares Python 3.11, GeoPandas, rasterio, rioxarray, GDAL, libpysal/esda/spreg, scikit-learn, and the rest of the wider project's stack, including `earthengine-api` and `osmnx` for the datasets that need them). This environment has **not** been re-verified in the environment this repository was prepared in (system GDAL/rasterio bindings were not installable there — see `scripts/revision_v3/audit_report.md` §6) — it is deposited as the project's own declared environment, not re-tested here.
- **v3 analysis** (`scripts/revision_v3/`): Python 3.14, packages pinned in `requirements.txt` (`pip install -r requirements.txt`). Deliberately avoids rasterio/GDAL Python bindings (see below).
- Node.js (for the manuscript `.docx` build only) — `cd scripts/revision_v3/docx_build && npm install`
- **QGIS installed locally**, used only by the v3 scripts for its bundled GDAL command-line tools (no system GDAL/rasterio bindings required for this half). `config.py`'s `QGIS_GDAL_BIN_DIR` and `QGIS_GDAL_ENV` point at a specific machine's QGIS install path — update these to your own before running scripts 01–02 of `scripts/revision_v3/`.

## Running

Upstream pipeline first (produces the grid and covariate files the v3 analysis reads):

```bash
conda activate ghana_buildings
python scripts/01_setup_project.py
python scripts/02_download_boundaries.py
python scripts/03_prepare_grids.py
python scripts/03b_generate_250m_grid.py
python scripts/04_clean_building_footprints.py
python scripts/05_process_temporal_buildings.py
python scripts/06_process_ghsl.py
python scripts/07_add_population_roads_pois.py
python scripts/08_calculate_building_metrics.py
python scripts/09_aggregate_to_grid.py
```

Then the v3 analysis, in its own environment:

```bash
python scripts/revision_v3/run_all.py
```

runs the v3 pipeline (acquisition → processing → sample definition → models → weights diagnostics → typology → validation → facts.json) with one command. Flags: `--from N` to resume from stage N, `--only N` for a single stage, `--skip-tests`, `--skip-facts`.

```bash
python -m pytest scripts/revision_v3/tests/
```

runs the 11-test identity/range suite independently of the pipeline.

`scripts/revision_v3/docx_build/build_docx.js` regenerates the manuscript `.docx` directly from its Markdown source; it expects a `04_outputs/` tree (manuscript Markdown, figure PNGs) that is not part of this deposit, since that's the paper draft, not analysis code.

## Scripts

### Upstream pipeline (`scripts/`)

| Script | Produces |
|---|---|
| `01_setup_project.py` | Verifies folder structure, writes the data inventory template and a CRS decision note |
| `02_download_boundaries.py` | Ghana administrative boundaries (GADM) and the 15 named urban-region/corridor bounding boxes |
| `03_prepare_grids.py` | National 500 m and 1 km analytical grids |
| `03b_generate_250m_grid.py` | 250 m grid for intra-urban sensitivity checks |
| `04_clean_building_footprints.py` | Cleaned Open Buildings V3 / Overture footprint layers: invalid geometry repair, confidence filtering, area/perimeter/compactness metrics |
| `05_process_temporal_buildings.py` | Open Buildings 2.5D Temporal layers aggregated to the grid, 2016–2023 |
| `06_process_ghsl.py` | GHSL Built-Up Surface aggregated to the grid across all epochs (1975–2020) |
| `07_add_population_roads_pois.py` | WorldPop density, OSM road density, POI proximity, CBD distance |
| `08_calculate_building_metrics.py` | Per-building and per-cell size/shape/inequality metrics |
| `09_aggregate_to_grid.py` | Produces `grid_500m.parquet` and `grid_model_ready_500m.parquet`, the two files every `scripts/revision_v3/` script reads |

### v3 analysis (`scripts/revision_v3/`)

| Script | Produces |
|---|---|
| `01_acquire_ghsl_height.py` | Downloads AGBH/ANBH directly from JRC (not Earth Engine — see script docstring and `audit_report.md` §1–2 for why), with full checksummed provenance |
| `02_process_agbh_anbh_grid.py` | Pixel-level identity checks, area-weighted aggregation to the 500 m grid |
| `03_sample_definitions_and_sensitivity.py` | National-footprint vs. valid-height sample definitions; built-fraction threshold sensitivity |
| `04_models_primary.py` | Nested OLS + spatial error models of vertical intensity |
| `05_weights_diagnostics.py` | Spatial weights audit (islands, components) and sensitivity |
| `06_typology_v3.py` | Four-class typology, tertile/quartile sensitivity, AGBH-matched comparison, predictor-by-class profile |
| `07_validation_module.py` | Spatial plausibility and cross-dataset validation checks; manual-validation sample |
| `08_manuscript_figures.py` | All manuscript figures, including the sample-flow diagram |
| `build_facts_json.py` | Consolidates every script's findings into `facts.json` |
| `run_all.py` | One-command orchestrator; rebuilds `output_manifest.csv` |

`facts.json`, `audit_report.md`, and `output_manifest.csv` in this repository are this project's own outputs, included as a record of what the pipeline produced — rerunning the scripts against freshly acquired data will regenerate them.

## License

MIT — see `LICENSE`.
