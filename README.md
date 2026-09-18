# Ghana Building Height Analysis (2018)

Analysis code for: **"Building height and horizontal expansion reveal distinct urban growth pathways in Ghana"** (Michael Gameli Dziwornu, Michael Mensah). Citation details will be added here once the paper is published.

This repository is the code deposit accompanying that manuscript. It contains every script that produced a number, table, or figure in the paper, plus the identity/range test suite and the consolidated audit trail. It does **not** contain raw or processed data — see [Data](#data) below for why, and where to get it.

## Scope

This is the **v3** analysis: ANBH (Average Net Building Height) as the primary vertical outcome, AGBH (Average Gross Building Height) as a volumetric-density robustness check, for the 2018 GHSL epoch. It supersedes an earlier internal revision built around AGBH alone; that history is narrated in the paper itself (Sections 2.3, 4.3) and in `scripts/revision_v3/audit_report.md`, but its code is not part of this deposit.

## Data

Raw inputs (GHSL GHS-BUILT-H / GHS-BUILT-S, Google Open Buildings V3, WorldPop, Ghana 2021 PHC, OpenStreetMap, GADM boundaries) are third-party datasets with their own licenses and are not redistributed here. Sources, DOIs, and access dates are listed in the paper's Table 1 and in `scripts/revision_v3/01_acquire_ghsl_height.py`'s provenance output.

**This repository is not sufficient on its own to reproduce the analysis from raw data.** The v3 scripts assume the national 500 m analytical grid (`grid_500m.parquet`), its model-ready covariates (`grid_model_ready_500m.parquet` — pre-2018 built coverage, CBD distance, Open Buildings footprint variables), and the 15 named-region bounding boxes already exist. These are produced by an earlier, wider project pipeline that this deposit does not include and that is not yet separately published. Someone starting from raw data alone would need to reconstruct that upstream pipeline first; what this repository guarantees is that every number, table, and figure in the paper traces to visible, checkable code operating on those already-built inputs.

## Requirements

- Python 3.14, packages pinned in `requirements.txt` (`pip install -r requirements.txt`)
- Node.js (for the manuscript `.docx` build only) — `cd scripts/revision_v3/docx_build && npm install`
- **QGIS installed locally**, used only for its bundled GDAL command-line tools (no system GDAL/rasterio bindings are required). `config.py`'s `QGIS_GDAL_BIN_DIR` and `QGIS_GDAL_ENV` point at a specific machine's QGIS install path — update these to your own before running scripts 01–02.

## Running

```bash
python scripts/revision_v3/run_all.py
```

runs the v3 pipeline (acquisition → processing → sample definition → models → weights diagnostics → typology → validation → facts.json) with one command, assuming both the raw third-party data above **and** the upstream grid/covariate files described in [Data](#data) already exist where `config.py` expects them. Flags: `--from N` to resume from stage N, `--only N` for a single stage, `--skip-tests`, `--skip-facts`.

```bash
python -m pytest scripts/revision_v3/tests/
```

runs the 11-test identity/range suite independently of the pipeline.

`scripts/revision_v3/docx_build/build_docx.js` regenerates the manuscript `.docx` directly from its Markdown source; it expects a `04_outputs/` tree (manuscript Markdown, figure PNGs) that is not part of this deposit, since that's the paper draft, not analysis code.

## Scripts

| Script | Produces |
|---|---|
| `01_acquire_ghsl_height.py` | Downloads AGBH/ANBH directly from JRC (not Earth Engine — see script docstring and `audit_report.md` §1–2 for why), with full checksummed provenance |
| `02_process_agbh_anbh_grid.py` | Pixel-level identity checks, area-weighted aggregation to the 500 m grid |
| `03_sample_definitions_and_sensitivity.py` | National-footprint vs. valid-height sample definitions; built-fraction threshold sensitivity |
| `04_models_primary.py` | Nested OLS + spatial error models of vertical intensity |
| `05_weights_diagnostics.py` | Spatial weights audit (islands, components) and sensitivity |
| `06_typology_v3.py` | Four-pathway growth typology, tertile/quartile sensitivity, AGBH-matched comparison, predictor-by-pathway profile |
| `07_validation_module.py` | Spatial plausibility and cross-dataset validation checks; manual-validation sample |
| `08_manuscript_figures.py` | All manuscript figures, including the sample-flow diagram |
| `build_facts_json.py` | Consolidates every script's findings into `facts.json` |
| `run_all.py` | One-command orchestrator; rebuilds `output_manifest.csv` |

`facts.json`, `audit_report.md`, and `output_manifest.csv` in this repository are this project's own outputs, included as a record of what the pipeline produced — rerunning the scripts against freshly acquired data will regenerate them.

## License

MIT — see `LICENSE`.
