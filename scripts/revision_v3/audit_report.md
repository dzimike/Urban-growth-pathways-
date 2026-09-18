# Paper 2 v3 — Analytical Audit Report

Status as of scripts 01–03 (data acquisition, aggregation, sample definitions). Updated as later scripts run. All numbers below are re-derived independently in `build_facts_json.py` and stored in `facts.json`; nothing here is asserted from memory alone.

---

## 1. Provenance finding that motivated this revision

`02_processed_data/ghsl_layers/ghsl_height_grid_500m.parquet` and `height_typology_grid_500m.parquet` (used throughout the Paper 2 v2 analysis) have **no producer script anywhere in the codebase** and **no entry in `processing_log.csv`**. `scripts/06_process_ghsl.py`'s own docstring and code declare its only outputs as `ghsl_grid_metrics.parquet` and `builtup_age_class_100m.tif`. This was confirmed by exhaustive grep and by cross-checking every processing-log line for Phase 5.

Digging further: the codebase's hard-coded Earth Engine asset ID for AGBH, `JRC/GHSL/P2023A/GHS_BUILT_H_AGBH` (in `scripts/01_download_data/download_ghsl_gee.py`), **does not exist** on Earth Engine — confirmed directly against the API (`EEException: not found`). The real EE asset, `JRC/GHSL/P2023A/GHS_BUILT_H/2018`, exposes only ANBH per Google's own catalog documentation. This means the documented download script could never have produced an AGBH file if run as written, consistent with the missing processing-log entry.

The old file's summary statistics (mean 0.163 m, matching the newly verified AGBH's footprint-sample mean of 0.163 m almost exactly) suggest it probably was genuine AGBH data, acquired by some undocumented process — but its exact acquisition chain, epoch alignment, and masking rules remain unverifiable, which is why v3 does not reuse it.

## 2. Acquisition (Script 01)

- **Both AGBH and ANBH acquired directly from JRC's distribution server** (`jeodpp.jrc.ec.europa.eu`), bypassing Earth Engine entirely for this step, since JRC's own files are unambiguously labeled by filename and there is no asset-identity ambiguity.
- DOI: `10.2905/85005901-3A49-48DD-9D19-6261354F56FE` (GHS-BUILT-H R2023A, Pesaresi & Politis 2023), epoch 2018, both products.
- **Tile identification was empirically verified, not guessed.** Two independent automated guesses at which of GHSL's global 10°-tiles cover Ghana (R3/R4 × C20/C21) were both wrong — verified by downloading one candidate and reading its real embedded bounding box, which showed **Scandinavia** (lon 10–20°E, lat 59–69°N), not West Africa. The correct tiles (R8/R9 × C18/C19) were then derived from that real data point and verified remotely via GDAL's `/vsizip/vsicurl/` virtual filesystem (reading each candidate's header over HTTP without downloading the full file) before any bulk download was attempted.
- Every tile's SHA-256 is recorded in `01_raw_data/ghsl/provenance_agbh_anbh_2018.json`, along with the two final clipped rasters' checksums.
- Final rasters: `agbh_2018_v3.tif`, `anbh_2018_v3.tif`, both clipped to `[-3.5, 4.3, 1.3, 11.3]`, 100 m, EPSG:4326.

### Pixel-level validation (48,384,000 pixels, full population, not sampled)

| Check | Result |
|---|---|
| NaN pixels | 0 in both rasters |
| Negative pixels | 0 in both rasters |
| AGBH==0 / ANBH==0 co-location | Exactly identical: 42,251,166 pixels each, fully overlapping. Zero mismatches either direction. |
| Core identity AGBH ≤ ANBH | Holds in 48,383,999 of 48,384,000 pixels (99.999998%). The single exception is a 0.0505 m excess, almost certainly a mosaic-seam rounding artefact. |
| Built fraction range | [0.00012, 1.0066] over built pixels; the one value above 1.0 is the same single pixel above. |
| No separate nodata sentinel | Confirmed: for this extract, every pixel is a valid observation; zero means "not built," not "missing." |

**Conclusion**: the source data for v3 is materially cleaner than what v2 relied on, and the one known imperfection (a single pixel) is quantified and tolerance-coded into the test suite (`tests/test_identity_range.py`), not silently absorbed.

## 3. Aggregation to the 500 m grid (Script 02)

Method: reprojected AGBH/ANBH to EPSG:32630 (nearest-neighbour, preserves raw values), rasterised the **actual** `grid_500m.parquet` polygons (not a reconstructed grid) onto the same pixel grid, and aggregated by plain pixel count/sum in that true-equal-area space — this is the area-weighted mean, without needing a separate cos(latitude) correction, because UTM pixels are genuinely equal-area.

- 961,858 national cells processed. 57.8% of raster pixels fall inside a grid cell (the remainder is ocean or neighbouring-country land within the rectangular download bbox — expected).
- **716 cells have zero overlapping pixels.** Investigated, not assumed benign: median polygon area of these cells is ~91,000× smaller than a typical 500 m cell — they are boundary-clip slivers left over from clipping the national grid to Ghana's coastline (`scripts/03_prepare_grids.py`), too thin for any 100 m pixel centre to fall inside under `gdal_rasterize`'s default convention. Correctly encoded as **NaN**, not 0. 26 of these fall inside the 296,703-cell building-footprint sample.
- **Cell-level identity check**: 0 cells where AGBH_mean > ANBH_derived — holds perfectly after aggregation.
- **v2 regression check**: v2's "implied height" (AGBH ÷ current V3 coverage, a double-division error) produced outliers up to 263 m. v3's correctly-derived `cell_anbh_derived_m` maxes out at **21.0 m** nationally — no comparable outlier tail.
- On the 296,677-cell footprint sample with valid data: AGBH mean 0.146 m, median 0.004 m; ANBH-derived mean 2.45 m, median 2.50 m, max 21.0 m.

### Investigated finding: ANBH quantization spike

40.1% of all "built" (ANBH > 0) pixels nationally fall in a single narrow 0.05 m-wide bin, 2.50–2.55 m (2,458,649 of 6,132,834 built pixels). This is not a physical clustering of real building heights — it is consistent with a hard-coded fallback/prior value in GHSL's height-retrieval model, applied where the underlying stereo-photogrammetric/SAR signal is too weak to trust (small, sparse, or low-confidence structures). This explains the median-ANBH-clustering-at-~2.5m pattern observed at every built-fraction threshold below 0.05 (Section 4). **This should be stated as a named limitation of the ANBH product in any manuscript text**, not smoothed over.

## 4. Sample definitions and threshold sensitivity (Script 03)

Two distinct sample concepts, kept explicitly separate:
- **National-footprint sample** (296,703 cells): Open Buildings V3, `building_count > 0`. This is the sample used throughout the wider project.
- **Valid-height sample**: GHSL `built_fraction` exceeding a stated threshold — a different population, defined purely from the GHSL side.

| Threshold | n cells | % of national grid | Covers % of footprint sample | ANBH mean (m) | ANBH median (m) |
|---|---|---|---|---|---|
| >0 | 404,722 | 42.08% | 85.36% | 2.73 | 2.50 |
| 0.005 | 102,048 | 10.61% | 33.82% | 3.42 | 2.50 |
| 0.01 | 78,865 | 8.20% | 26.37% | 3.69 | 2.50 |
| 0.02 | 58,806 | 6.11% | 19.75% | 4.07 | 2.56 |
| 0.05 | 35,771 | 3.72% | 12.04% | 4.98 | 4.48 |

### Investigated finding: asymmetric cross-dataset mismatch

The `>0` threshold sample only overlaps 62.6% with the footprint sample. This asymmetry was investigated in both directions rather than left as a caveat:

- **V3 has it, GHSL (`>0`) doesn't** (43,439 cells): median `building_count` = 1, vs. 4 for the footprint sample overall. Consistent with sparse, isolated, or newly built (2018→2023 gap, since V3 is a ~2023 snapshot against this GHSL 2018 epoch) small structures below GHSL's 100 m detection floor.
- **GHSL (`>0`) has it, V3 doesn't** (151,458 cells): this group's mean built_fraction (0.00047) is roughly **41× smaller** than the overall `>0`-sample mean (0.0192). This is overwhelmingly the noise floor of an overly permissive `>0` threshold — near-zero estimation noise or non-building surface texture, not genuine missed buildings. **This is precisely why the `>0` threshold should not be used alone**; the sensitivity table shows overlap with genuine V3 buildings climbing to 98.3–99.8% once a threshold of 0.005 or higher is applied.

### Investigated finding: the raw maximum ANBH is not a robust statistic (prompted by manuscript figure review)

A careful re-check of the manuscript figures against the underlying data (not a scripted pipeline step — an ad hoc investigation triggered by reviewing Fig. 1) found that `anbh_derived_max` is **identical, 21.0092 m, at every one of the five built-fraction thresholds** in the table above, from `>0` through `≥0.05`. Built-fraction thresholding — the primary safeguard this script provides — does **not** filter out whatever is producing that value. `anbh_derived_p99`, by contrast, is stable and much lower across the same thresholds (8.21 → 10.69 m), and rises *with* stricter thresholds rather than staying pinned like the max does.

Tracing the maximum to its source cell (`grid_id 500m_00772288`, centroid 0.3284°E/7.7017°N — rural Volta region, not near any of the 15 named urban regions) and pulling the raw 100 m AGBH/ANBH pixels directly from `agbh_2018_v3.tif`/`anbh_2018_v3.tif` (via `gdal_translate -projwin`, not through the aggregation script) shows this is not simple single-pixel noise: a small block of adjacent pixels genuinely reads AGBH up to 11.2 m and ANBH up to 26.5 m, concentrated over only a few of the cell's 25 pixels (cell-level built fraction = 6.3%). Whether this reflects a real small tall structure or shoreline-adjacent retrieval noise (the cell sits at the edge of a generous Lake Volta bounding envelope) cannot be determined from this data alone — 8 of the 177 nationally cells with derived ANBH > 12 m fall within that envelope, while most of the rest (e.g. near Obuasi, Tarkwa/Prestea) have much higher built fraction (8–51%) and plausibly reflect real mining/industrial infrastructure instead.

**Conclusion, applied in the manuscript (Section 5.1, Limitation 4):** report the 99th percentile, not the maximum, as the ANBH ceiling statistic, and describe the maximum's instability explicitly rather than calling it "a physically plausible ceiling" as an earlier draft of this section did. Full detail in `facts.json` → `sample_sensitivity.max_vs_p99_robustness_check`. (Note: this was Limitation 3 when first written; renumbered to 4 after Limitation 2 -- the AGBH-confound finding, item 12 below -- was inserted. Every cross-reference to this Limitation across the manuscript, response letter, and this file was re-checked and corrected at that point, not just this one.)

## 5. Test suite

`tests/test_identity_range.py`: **11/11 passing**, covering pixel-level NaN/negative/zero-colocation/identity/range checks (independent of script 02's aggregation logic) and cell-level row-count/nodata-disclosure/identity/no-extreme-outlier checks. Re-run via:

```bash
python -m pytest scripts/revision_v3/tests/test_identity_range.py -v
```

## 6. Environment notes for reproducibility

- `rasterio` and standalone `osgeo` Python bindings are not installable in this environment (no system GDAL, no working `gdal-config`; the project's declared `environment.yml` conda env, `ghana_buildings`, does not exist on this machine). All raster processing in v3 uses QGIS's bundled GDAL command-line tools (`gdalwarp`, `gdal_translate`, `gdal_rasterize`, `gdal_calc.py`) via subprocess, with `PROJ_LIB`/`PROJ_DATA` explicitly pointed at QGIS's bundled `proj.db` (required for any reprojection step; plain `gdalinfo` works without it, which is why this requirement wasn't obvious until `gdal_merge.py`/`gdalwarp -t_srs` failed).
- Python 3.14's `urllib` has no configured CA bundle in this environment; direct downloads use `curl` instead (confirmed working against the macOS system trust store).
- Earth Engine: authenticated successfully with project ID `instigis` (the account's actual lowercase GCP project ID; the display name "InstiGIS" does not work directly as an Earth Engine project argument).

## 7. Nested OLS and spatial models (Script 04)

**Sample**: national-footprint sample (296,703) minus 26 nodata cells (Section 3) minus 9 cells with missing 1975/2010/2015 coverage covariates → **final n = 296,668**, fully disclosed and saved to `sample_construction_report.json`.

**Predictors, primary nested sequence (pre-2018 only, per Requirement 6)**: `log_distance_to_nearest_cbd_m` (time-invariant, no leakage risk) → + `built_coverage_2010` → + `growth_1975_2010` + `growth_2010_2015`. Population density (WorldPop 2020) is deliberately **excluded from the primary sequence**: it postdates the 2018 AGBH/ANBH epoch by two years, which is exactly the kind of leakage Requirement 6 rules out for the growth predictors, and there is no principled reason to hold population to a looser standard than growth. All VIFs are below 4 (max 3.97, `growth_1975_2010` in the full model) — no multicollinearity concern, unlike the v2 model's `distance_to_cbd`/`population_density` pair, which is absent here by construction.

**Primary outcome (ANBH, log1p-transformed)**:

| Model | OLS R² | Spatial pseudo-R² | λ | Moran's I: outcome | OLS residuals | Spatial residuals (correct) |
|---|---|---|---|---|---|---|
| A: CBD only | 0.0255 | 0.0255 | 0.532 | 0.444 | 0.431 | −0.026 |
| B: + built coverage 2010 | 0.2133 | 0.2120 | 0.427 | 0.444 | 0.313 | −0.016 |
| C: + growth 1975–2010, 2010–2015 | 0.2346 | 0.2330 | 0.407 | 0.444 | 0.293 | −0.013 |

**Robustness outcome (AGBH, log1p-transformed)**:

| Model | OLS R² | Spatial pseudo-R² | λ | Moran's I: outcome | OLS residuals | Spatial residuals (correct) |
|---|---|---|---|---|---|---|
| A: CBD only | 0.1132 | 0.1132 | 0.851 | 0.822 | 0.796 | 0.025 |
| B: + built coverage 2010 | 0.9168 | 0.9160 | 0.624 | 0.822 | 0.488 | 0.066 |
| C: + growth 1975–2010, 2010–2015 | 0.9462 | 0.9456 | 0.460 | 0.822 | 0.309 | 0.073 |

### Self-correction: spatial-residual Moran's I was initially computed on the wrong attribute

The first run of this script computed the "spatial residuals" Moran's I on `spreg.GM_Error`'s `.u` attribute. Checking the library's own docstring (`u : nx1 array of residuals` vs a **separate** `e_filtered : nx1 array of spatially filtered residuals`) revealed that `.u` is the plain, unfiltered residual on the original scale, not the residual the spatial-error estimator actually targets. Computed on `.u`, every one of the six model/outcome combinations showed spatial residual Moran's I *higher* than OLS residual Moran's I — a result that would have been reported as "the spatial correction does not help, and sometimes hurts," which is both surprising and, it turns out, wrong.

Re-run on the correct `e_filtered` attribute, the result flips completely: spatial residual Moran's I falls to **essentially zero for all three ANBH models (−0.026 to −0.013)** and drops substantially for AGBH (0.796→0.025 in Model A; 0.309→0.073 in Model C). The spatial error specification does what it is supposed to do, once diagnosed correctly. This is recorded here specifically so the mistake and its correction are both auditable, not just the final number.

**Interpretation, stated cautiously per the "research-grade" brief**: built_coverage_2010 (a legacy/historical-depth proxy) is the dominant predictor for both outcomes, consistent with the vertical density literature's expectation that height accumulates where urbanization has been established longest. `growth_2010_2015` has a notably larger standardized coefficient than `growth_1975_2010` for ANBH (0.142 vs 0.053), suggesting the years immediately before the 2018 epoch matter disproportionately — plausible, but this is an observational association, not a causal claim, and no attempt is made here to over-interpret it further pending script 06 (typology) and any qualitative cross-check.

## 8. Spatial weights audit and sensitivity (Script 05)

Same sample as script 04 (n=296,668; consistency asserted in code, not just assumed).

| Spec | Islands | Connected components | Min/max/mean neighbours |
|---|---|---|---|
| Queen contiguity | 9,479 | **18,998** | 0 / 8 / 5.07 |
| Queen + nearest-neighbour for islands | 0 | **10,512** | 1 / 8 / 5.13 |
| KNN-8 | 0 | 10 | 8 / 8 / 8.00 |

**Attaching islands to their nearest neighbour does not come close to producing a connected graph.** This was investigated rather than left as a bare number: of the Queen weights' 18,998 components, one giant component covers **124,529 cells (42.0% of the sample)** — plausibly Ghana's main contiguous metropolitan/peri-urban corridors — while 9,479 are pure singletons, 6,223 are tiny 2–5-cell clusters, and 2,604 more are 6–20 cells. This is consistent with, and a useful quantitative confirmation of, the qualitative picture already established elsewhere in this project: much of Ghana's built environment is genuinely dispersed (isolated compounds, small rural service centres, scattered peri-urban development) rather than physically contiguous at 500 m resolution. It is not a processing artefact, and attaching a single nearest neighbour to a singleton usually just pairs it with another small isolated cluster rather than joining it to the main network.

**Sensitivity of Model C (ANBH, full pre-2018 predictor set) to the choice of weights:**

| Spec | λ | Pseudo-R² | Moran's I: outcome | OLS residual | Spatial-filtered residual |
|---|---|---|---|---|---|
| Queen | 0.407 | 0.2330 | 0.444 | 0.293 | −0.013 |
| Queen + KNN-1 islands | 0.401 | 0.2336 | 0.459 | 0.313 | −0.023 |
| KNN-8 | 0.556 | 0.2331 | 0.399 | 0.251 | 0.002 |

Pseudo-R² is essentially identical across all three specifications (0.2330–0.2336), coefficients are stable (e.g. `built_coverage_2010`: 2.58 / 2.57 / 2.48; `growth_2010_2015`: 5.63 / 5.52 / 5.01), and spatial-filtered residual Moran's I is near-zero under every specification. **The substantive conclusions from script 04 do not depend on the specific spatial weights definition used.** λ itself varies more (0.40–0.56, expected since it is defined relative to a differently-structured W each time and is not directly comparable across specifications), but this does not affect the coefficients or model fit materially.

## 9. Typology rebuild (Script 06)

**Sample**: 296,703 → drop 26 (no ANBH) → drop 9 (missing 1975/2015 coverage) → **final n = 296,668** (identical to script 04's sample, since the same 35 cells are excluded for the same reasons).

**Axes, changed from v2**:
- Vertical: `cell_anbh_derived_m` (script 02) — the primary vertical outcome, not AGBH.
- Horizontal: `growth_1975_2015` = `built_coverage_2015` − `built_coverage_1975` — the full prior-expansion trajectory truncated at the **last pre-2018 GHSL epoch**, replacing v2's `long_term_growth` (1975–2020), which extended two years past the 2018 outcome. This is the same no-post-outcome-leakage logic from Requirement 6, applied to the typology's own horizontal axis rather than left inconsistent with the regression models.

**Classification rule, fully disclosed** (median split; ties assigned to the low class): ANBH median = 2.500403 m (148,322 above / 29 exactly equal / 148,317 below); growth_1975_2015 median = 0.000500 (147,622 above / 2,337 exactly equal / 146,709 below). Both medians are comfortably nonzero with small tie shares (<0.8% of the sample on either axis) — no repeat of the near-zero-median degeneracy that broke v2's original typology.

**Descriptive labels** (literal, per Requirement 9, not evocative names):

| Class | n | % | Mean ANBH (m) | Median ANBH (m) | Mean growth 1975–2015 |
|---|---|---|---|---|---|
| High height / High prior expansion | 94,885 | 32.0% | 3.47 | 2.50 | 0.0440 |
| Low height / Low prior expansion | 95,609 | 32.2% | 1.35 | 2.39 | 0.0001 |
| High height / Low prior expansion | 53,437 | 18.0% | 2.60 | 2.52 | 0.0002 |
| Low height / High prior expansion | 52,737 | 17.8% | 2.48 | 2.50 | 0.0141 |

**Sensitivity to threshold scheme:**

| Scheme | Diagonal (jointly concordant) | Off-diagonal |
|---|---|---|
| Median (2×2) | 64.21% | 35.79% |
| Tertile (3×3) | 43.15% | 56.85% |
| Quartile (4×4) | 46.61% | 53.39% |

### Substantive finding: height and prior expansion are far more weakly associated here than density and growth were

This ANBH-based typology shows weak diagonal concordance (64.2% at median split, barely above the ~33% chance level at tertile split). To isolate whether this is attributable specifically to ANBH's retrieval-floor problem, or to some other factor (sample, horizontal-axis definition), a **matched comparison** was run: identical sample (n=296,668), identical horizontal axis (`growth_1975_2015`), identical tie rule, with only the vertical variable swapped from ANBH to AGBH. This is the cleanest possible like-for-like test.

| Vertical variable | Median-split diagonal | Tertile-split diagonal |
|---|---|---|
| ANBH (primary) | 64.2% | 43.2% |
| AGBH (matched comparison) | **89.8%** | **79.5%** |

With everything else held constant, swapping ANBH for AGBH raises the typology's diagonal concordance by 25.6 percentage points at median split and 36.3 points at tertile split. This isolates the effect cleanly: it is not the sample, not the horizontal axis, and not the classification rule — it is specifically ANBH's own measurement quality. This is directly explained by the quantization finding in Section 3: **median ANBH is nearly identical across all four ANBH-based classes (2.39–2.52 m)**, sitting close to GHSL's ~2.5 m retrieval-floor value regardless of how much prior horizontal expansion a cell had. Only the *mean* differentiates classes clearly (1.35 m in the lowest class vs. 3.47 m in the highest), driven by a right-skewed tail of genuinely tall cells. **Height (ANBH) is a structurally noisier, more floor-dominated measure than volumetric density (AGBH) for distinguishing growth pathways**, and this should be stated plainly rather than downplayed. (Table generated by `06_typology_v3.py`, saved to `T_typology_agbh_matched_comparison.csv`.)

### Note on test-suite stability

One isolated run of `test_identity_range.py` immediately following a `build_facts_json.py` run in the same shell produced 5 passes and 6 errors (not failures) in the pixel-level fixture; two immediate re-runs both passed 11/11 cleanly. This is consistent with transient `/tmp` file contention between the two scripts' own ENVI-export temp files rather than a real regression, but is recorded here rather than silently dismissed, since an intermittent test-suite issue is exactly the kind of thing worth flagging even when it resolves on retry.

## 10. Validation module (Script 07)

No independent ground-truth building-height dataset (LiDAR, field survey) exists for this project, so three evidence-based checks were run instead of a single "validation" claim, following the existing project pattern (`scripts/15_validation_sampling.py`).

### Part A — Spatial plausibility against an independent region classification

Cross-tabulated the ANBH-based typology (Section 9) against `urban_region_name`, a classification built entirely from administrative boundaries (Section on Methods correction below), not from any GHSL height data.

| Region tier | "High height" share |
|---|---|
| Metro core (Accra-Tema, Kumasi) | **90.5%** |
| Secondary city | 62.5% |
| Peri-urban corridor | 57.3% |

This matches independent geographic knowledge of Ghana's urban hierarchy (Accra/Kumasi's central business districts are documented to have the country's tallest, densest buildings) without that knowledge having been used anywhere in constructing ANBH. A genuine plausibility check, not a tautology.

### Part B — Cross-dataset consistency against independent building-size metrics

Correlated cell-level ANBH/AGBH against building-size metrics computed from a **different dataset and different measurement process** (Open Buildings V3 footprint geometry: median footprint area, size-Gini, share of very-large buildings). All six correlations are positive and significant at p<0.001:

| Height variable | Building-size metric | Pearson r | Spearman r |
|---|---|---|---|
| ANBH derived | median_building_area | 0.174 | 0.333 |
| ANBH derived | building_area_gini | 0.408 | 0.323 |
| ANBH derived | share_very_large | 0.362 | 0.376 |
| AGBH mean | median_building_area | 0.120 | 0.474 |
| AGBH mean | building_area_gini | 0.357 | **0.661** |
| AGBH mean | share_very_large | 0.336 | **0.644** |

**AGBH's rank correlations with independent building-size metrics (up to 0.66) are consistently stronger than ANBH's (up to 0.38).** This is the third independent line of evidence for the same underlying pattern documented in Sections 3 and 9: AGBH (volumetric density) is a cleaner, more differentiated signal than ANBH (height alone, dominated by the ~2.5 m GHSL retrieval floor). Three separate checks — the pixel-level quantization histogram, the typology's weak diagonal dominance, and now this cross-dataset correlation — all point the same direction, which is itself a form of validation of the underlying finding.

### Part C — Structured manual-validation sample and coding template

Stratified sample (region tier × ANBH-derived height class: Metro core / Secondary city / Peri-urban corridor, each × High/Low ANBH) drawn following the `assign_strata`/`draw_stratified_sample`/`build_coding_template` pattern already established in `scripts/15_validation_sampling.py`. **120 cells** (20 per stratum, all six strata filled to quota). The coding template pre-fills each sampled cell's `rs_anbh_derived_m`, `rs_agbh_mean_m`, and an implied storey count (ANBH ÷ 3 m), leaving fields for a human validator (field visit or high-resolution imagery) to record an independently observed dominant storey count and flag whether it matches, over-, or under-estimates the GHSL-derived value.

**This produces the infrastructure for manual validation; it is not itself completed field validation**, which is outside what this pipeline can execute and should be stated as such wherever this module is referenced.

---

## Summary: three-way convergent evidence on the ANBH quantization limitation

Sections 3, 9, and 10 each independently found the same thing through different methods (pixel histogram, typology diagonal-dominance, cross-dataset correlation). This convergence should be treated as a load-bearing limitation in any manuscript built on this analysis, not a minor caveat: **ANBH alone is a noisier measure of vertical intensity than AGBH**, and any claim resting on ANBH's ability to finely discriminate between moderate levels of "height" should be qualified accordingly, even though ANBH remains the correct primary outcome per Requirement 5 (AGBH is a volumetric-density measure, not a height measure, and conflating the two was part of what made v2's "implied height" calculation break).

---

## 11. Manuscript figure cartography review and fixes (Script 08)

Not one of the 15 numbered requirements — `08_manuscript_figures.py` produces manuscript-support visuals only — but reviewed with the same rigor after a GIS-expert-level pass on the first version of the four figures found several concrete problems, detailed here rather than fixed silently.

**The built-fraction/maximum-ANBH finding (Section 4, "Investigated finding" above) was triggered by this review**: tracing Fig. 1's single darkest pixel led to the `anbh_derived_max` vs. `anbh_derived_p99` robustness check now in the manuscript's Section 5.1 and Limitation 4.

**Separately, four cartographic problems were found and fixed:**

1. **Rendering engine.** The original figures plotted grid-cell *centroids* as small scatter markers in raw WGS84 degrees with a naive `set_aspect("equal")`, rather than the actual cell polygons in the paper's own analysis CRS. Rewritten to use `geopandas` polygon choropleths of `grid_500m.parquet`'s real cell geometries, reprojected to `cfg.PROJ_CRS` (EPSG:32630, UTM Zone 30N) — the identical CRS used for the area-weighted aggregation in Section 3/4.5. This eliminates marker-overplotting artefacts in dense urban cores and makes the map's own projection consistent with the methodology it illustrates.
2. **No cartographic reference elements.** None of the three maps had a country outline, scale bar, or north arrow. Blank (unbuilt) cells were visually indistinguishable from "outside Ghana." Added a GADM national-boundary overlay (`01_raw_data/boundaries/gadm41_GHA.gpkg`, `ADM_ADM_0` layer), a scale bar sized to the map extent, and a north arrow to all three maps.
3. **Inconsistent disclosure between companion figures.** Fig. 1 (ANBH) disclosed its N and colour-scale percentile cap on the image; Fig. 2 (AGBH) disclosed neither, despite applying the same treatment. Both now disclose N, the cap value, and CRS identically — and both now explicitly state that their N (296,677) is 9 cells larger than the n=296,668 regression/typology sample, since a univariate height map does not require the 1975/2010/2015 built-coverage covariates that the regression and typology samples do (unlike the regression sample, restricting the map sample to match would mean dropping 9 cells with perfectly valid height data for a reason unrelated to what the map shows — so the two Ns are deliberately left different, with the difference disclosed, rather than forced to match).
4. **Caption overclaim.** Fig. 1's caption named only Accra-Tema, Kumasi, and Tamale as visual peaks. A small number of mining centres (Obuasi, in the raw pixel/cell data) show comparably strong localized ANBH, and are not among the 15 named study regions; the manuscript caption was corrected to name this explicitly with a caveat about plausible industrial rather than residential/commercial structures, rather than silently under-describing the map.

A live QGIS session (via the QGIS MCP) was considered for producing these maps instead, but QGIS was not running (`ping` failed), and using it would make figure regeneration depend on a manually-started desktop session rather than `run_all.py`'s unattended one-command pipeline. `geopandas` (confirmed available in this environment, including `pyogrio` for vector I/O without requiring `fiona`/`rasterio`/`osgeo` bindings) achieves the same cartographic quality while staying fully scripted.

**Two items self-flagged as incomplete in `paper2_response_to_reviewers_v3.md` were subsequently closed:**

- **Reviewer #2, Major 6** (predictors by typology class): script 06 now also merges `built_coverage_2010` and `distance_to_nearest_cbd_m`, uses the same four-covariate `dropna` list as script 04 (verified to leave n unchanged at 296,668 before writing any code, by checking the missing-value overlap directly), and writes `T_typology_predictor_profile.csv`. Manuscript Table 6.
- **Reviewer #4 point 1 / Reviewer #6** (sample-flow diagram): added to `08_manuscript_figures.py` as `figA2_sample_flow.png`, a CONSORT-style box diagram using the exact counts already disclosed in Section 4.2 prose (961,858 -> 296,703 -> -26 -> 296,677 -> -9 -> 296,668), with the 15 named regions drawn as a side branch rather than a further sequential exclusion, since region membership is a labelled subset, not an exclusion step. Manuscript Fig. A2.

**Follow-up: overlaid the paper's own 15 named urban regions/corridors.** All three maps now draw the `urban_regions` layer already produced by `scripts/02_download_boundaries.py` (`ghana_admin_levels.gpkg`) -- explicitly the 15 hand-coded bounding boxes from Section 3, not GADM administrative regions, which this study deliberately does not use as its region definition. Cities get a solid outline and label; corridors get a lighter dotted outline and no label (most corridors overlap their parent city's box, so a second label would only duplicate it). This lets a reader visually verify the "peaks at Accra-Tema, Kumasi, Tamale" claim in the Fig. 1 caption directly against the labeled boxes.

One bug caught while implementing this: filtering `regions[regions.type == "urban_region"]` silently returned an empty result — geopandas' `GeoDataFrame.type` attribute returns each row's *geometry* type ("Polygon"), shadowing a same-named data column, rather than raising an error. Fixed by using `regions["type"]` (bracket access) instead of `regions.type` (attribute access); this is worth remembering for any future column named `type`, `geometry`, `bounds`, `area`, or any other name that collides with a `GeoDataFrame` property.

---

## 12. Major finding: the AGBH-vs-ANBH comparisons are not independent of the retrieval-floor claim they were used to support

Raised by an external technical review of the manuscript, not self-caught this time — but verified independently before acting on it, the same standard applied to every other correction in this file.

**The claim reviewed:** Section 5.5 (typology) and Section 5.2 (regression) each compared ANBH against AGBH and attributed AGBH's better performance (89.8% vs. 64.2% typology diagonal concordance; R² 0.946 vs. 0.235) to ANBH's retrieval-floor artefact, describing the typology comparison specifically as "the cleanest possible like-for-like comparison, since only the height variable changes."

**Why that's wrong:** AGBH = ANBH x built fraction (the official GHSL identity, verified at 99.999998% pixel-level exactness). Built fraction is itself a built-coverage measure. Checked directly against the sample used throughout this paper (n=296,668):

```
cell_built_fraction_mean (2018, GHS-BUILT-H) vs. growth_1975_2015 (GHS-BUILT-S): Pearson r = 0.814
cell_built_fraction_mean (2018, GHS-BUILT-H) vs. built_coverage_2010 (GHS-BUILT-S):  Pearson r = 0.970
cell_built_fraction_mean (2018, GHS-BUILT-H) vs. built_coverage_2015 (GHS-BUILT-S):  Pearson r = 0.983
```

AGBH therefore mechanically shares information with the typology's horizontal axis (r=0.81) and with the regression's dominant predictor (r=0.97, built coverage 2010). Substituting AGBH for ANBH "with only the height variable changing" is not accurate: AGBH also carries a near-copy of built coverage 2010 riding along with it. Part — plausibly a substantial part — of AGBH's better performance in both comparisons is this mechanical overlap, not superior height measurement.

**What remains valid:** two diagnostics never depended on comparing against AGBH at all, and are unaffected — the pixel-level histogram (40.1% of built pixels in one 0.05 m bin, Section 5.1) and the near-flat median ANBH across all four typology classes (2.39–2.52 m, Section 5.5), both properties of ANBH's own distribution. The cross-dataset correlation gap against Open Buildings V3 building-size metrics (Section 5.6) also does not share AGBH's construction (V3 is footprint-detection, not a GHSL coverage product), though a milder version of the same concern — AGBH's coverage term could track building-size metrics for reasons unrelated to height accuracy — is now noted there too.

**Fix applied:** Section 5.2, Section 5.5 (both the pre-Table-6 and post-Table-6 paragraphs), Section 6.4 (new Limitation 2, items renumbered), the Abstract, and the response-to-reviewers letter (Reviewer #3's typology response, plus a new self-audit entry) all now state the confound explicitly rather than presenting either comparison as isolating the height variable. Table 6's framing was corrected separately (`growth_1975_2010` + `growth_2010_2015` sum *exactly* to the typology's horizontal axis — a distinct, purely arithmetic problem the same review caught; see the entries this replaces below).

No numbers in any table changed — this is a correction to interpretation and claims of independence, not to the underlying computed statistics, which remain correct.

## 13. Table 6's "independent check" framing was arithmetically wrong

`growth_1975_2015` (the typology's horizontal axis) is defined as `built_coverage_2015 - built_coverage_1975`. `growth_1975_2010` and `growth_2010_2015` (two of Table 6's four columns) are `built_coverage_2010 - built_coverage_1975` and `built_coverage_2015 - built_coverage_2010` respectively. By construction, `growth_1975_2010 + growth_2010_2015 = growth_1975_2015` exactly — these two columns are not independent of the typology's construction, they sum to it. `built_coverage_2010` is correlated with the same built-up history (r=0.97 with 2018 built fraction, per item 12 above) but not definitionally part of the typology. Only `distance_to_nearest_cbd_m` is genuinely external. The manuscript's claim that the four classes "separate cleanly on the regression predictors that do not enter the typology's own construction" was therefore inaccurate for three of its four columns. Reframed as a descriptive class profile rather than an independent validation (Section 5.5).

## 14. Six further factual/reference corrections from the same review

1. **Abstract Moran's I mislabelled.** "Reduces residual autocorrelation from Moran's I = 0.444 to -0.013" used 0.444, the *outcome* Moran's I (Table 2), as if it were a residual value. Corrected to the actual residual-to-residual comparison: OLS residual 0.293 -> spatially filtered residual -0.013. Introduced during abstract word-count trimming (a compression artefact, not a data error) and caught by external review rather than internally — worth noting since it shows the trimming pass itself needed a fact-check pass afterward, which it hadn't gotten.
2. **0/0 statement was backwards.** Section 4.4 said equal AGBH/ANBH zeros meant built fraction "is never undefined by a 0/0 case." Those 42,251,166 pixels *are* exactly the 0/0 cases; the pipeline resolves them by convention (`gdal_calc.py --calc="where(B>0, A/B, 0)"`, script 02) rather than leaving them undefined. Corrected to state this explicitly.
3. **UTM mislabelled "true-equal-area."** UTM (Transverse Mercator) is a conformal projection, not equal-area — `config.py` itself reserves a separate `EQUAL_AREA_CRS` (Africa Albers) for calculations where that distinction matters, which should have been the tell. The substantive method (counting equal-sized reprojected pixels) is fine; only the terminology was wrong. Corrected to describe pixels as equal-sized *in the projected plane*, with UTM's actual zone-scale distortion (<0.1%) stated rather than the false "equal-area" label.
4. **Weights-sensitivity "within 12%" claim was false for one coefficient.** Checked against `T_weights_sensitivity.csv` directly: built-coverage and growth coefficients do vary under 12% across the three weights specifications, but `log_distance_to_nearest_cbd_m` varies by 29.2% (Queen vs. Queen+KNN-islands) and 16.1% (Queen vs. KNN-8) — both well above 12%. Corrected to state the CBD-distance coefficient's greater relative sensitivity explicitly, while noting its absolute/standardized contribution is negligible regardless.
5. **Fig. 1 cited for content it doesn't show.** Section 5.1 cited "(Fig. 1)" immediately after the 40.1%-in-one-bin histogram statistic; Fig. 1 is the ANBH map, not a histogram — no histogram figure exists in the manuscript. Corrected to remove the false citation; an actual histogram figure would be a reasonable future addition, not added here.
6. **Wrong section number.** "This finding recurs independently in the regression results (Section 5.3)" — Section 5.3 is the spatial-diagnostics self-correction (the `.u` vs. `e_filtered` bug), not the regression results; those are Section 5.2. Corrected, and reworded given item 12 above (the regression doesn't independently re-derive the retrieval-floor finding the way the sentence implied).

Response-to-reviewers letter: one stale sentence removed (claimed two items "not yet carried forward" three paragraphs after the summary table said they were resolved — the table was updated correctly when Table 6/Fig. A2 were added; this one sentence was missed).

---

## 15. Small illustrative satellite check (6 of 120 manual-validation cells)

Not a completed validation — a bounded, honestly-scoped supplement, done because external review correctly noted that the 120-cell manual-validation infrastructure (script 07) had never actually been used. Full detail and caveats in `facts.json` → `validation.illustrative_satellite_check_6_cells`; summary here.

**What was checked:** the single most extreme high-ANBH cell and the single most extreme low-ANBH cell within each of the three region tiers (Metro core, Peri-urban corridor, Secondary city) — 6 of the 120 stratified cells, deliberately the most confident predictions the classification makes, not a random draw. Each cell's lat/lon centroid was located in Google Maps satellite view.

**Result:** 6 of 6 showed the expected pattern. High-ANBH cells: central Accra's government/administrative district (Metropolitan Assembly, ministries, GRA headquarters), a dense peri-urban residential/commercial corridor (Kasoa-Ofaakor), and Takoradi Technical University's campus. Low-ANBH cells: forested/rural land on Kumasi's periphery, farmland near Accra-Dodowa, and dense forest near Koforidua — no visible structures in any of the three.

**What this does and doesn't establish, stated plainly:** this is face validity for the classification's most extreme, most confident cases — a real, positive result, but a narrow one. It does not validate the 114 remaining cells or the moderate/ambiguous cases where classification error actually matters most (the extreme cases were never the ones in doubt). More fundamentally, overhead satellite imagery without shadow analysis, oblique views, or street-level imagery cannot distinguish a genuinely tall building from many low-rise buildings packed densely together — which is exactly the AGBH/ANBH distinction this paper's central argument turns on. No storey count or height estimate was attempted; the coding template's storey-count field remains empty. This check answers "does the classification point at real places that look like what it claims," not "is ANBH/AGBH numerically accurate." Limitation 7 (was 6 before the new Limitation 2 was inserted, item 12 above) and Section 5.6 now state this distinction explicitly rather than leaving the validation infrastructure completely untouched.

## 16. Code deposit expanded to include the upstream pipeline

The GitHub deposit previously covered only `scripts/revision_v3/`, with the upstream pipeline that builds `grid_500m.parquet` and `grid_model_ready_500m.parquet` described in words (README, manuscript Data Availability, response letter) but not deposited. At the user's request, added the 10 upstream scripts (`01_setup_project.py` through `09_aggregate_to_grid.py`, plus `03b_generate_250m_grid.py`) and the project's declared `environment.yml`.

Checked before depositing, not assumed safe: grepped all 10 files for credential/API-key/secret/password/token patterns (none), grepped for hard-coded `/Users/...` absolute paths (none — all resolve `config.py` via `Path(__file__).resolve().parents[1]`, same pattern as the v3 scripts), confirmed none of the 10 call Earth Engine authentication despite `environment.yml` declaring `earthengine-api` as a wider-project dependency, and ran `py_compile` on all 10 (all valid). Full detail in `facts.json` → `reproducibility_deposit_expansion`.

**What this does and does not close:** every script between raw data and this paper's final tables and figures is now visible. It does not make the repository clone-and-run: the raw third-party datasets are still not redistributed (licensing), and the upstream scripts' declared conda environment (`environment.yml`, Python 3.11 with real rasterio/fiona/GDAL bindings) was not re-verified in the environment this v3 revision was prepared in, which deliberately avoids those bindings (no system GDAL was installable here — Section 6). That gap predates this deposit and is disclosed, not newly introduced by it.

## 18. Figs. A1/A2 relocated to fix reading order — the same problem the v2 self-audit already caught, recurred

A full re-read of the manuscript, requested by the user, found that Fig. A1 (Section 2.1) and Fig. A2 (Section 4.2) both appear earlier in the document than Fig. 1 (Section 5.1) — a reader hits "Fig. A1" and "Fig. A2" before ever seeing "Fig. 1". This is exactly the issue the *v2* response letter documented finding and fixing ("Figs. A1 and A2 originally appeared out of numerical order... moved both schematics into the Appendix section"); it was not carried forward into this rebuild and had quietly reappeared.

Fix: added a new `## Appendix` section after the References, moved both figure captions (and, in the compiled docx, both images) there, and changed the in-text mentions in Section 2.1 and Section 4.2 from bare `(Fig. A1)` / `(Fig. A2)` to forward-pointing `(Fig. A1, Appendix)` / `(Fig. A2, Appendix)`. Verified in the rebuilt docx via `python-docx`: figure paragraphs now appear in the order Fig. 1, 2, 3, 4, [Appendix heading], A1, A2; image count still 6, table count still 6, all 11 tests still pass.

---

*Per the revision instructions, no manuscript text is written until this analytical audit is judged complete. Scripts 01-07 are now done; remaining infrastructure items (config.py path centralization already applied, run_all.py, final output_manifest.csv refresh) are the last steps before that judgement can be made.*
