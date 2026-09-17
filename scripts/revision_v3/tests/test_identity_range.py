"""
Paper 2 v3 -- Requirement 3: Identity and range tests for AGBH, ANBH, and
built fraction.

Runs at two levels:
  - Pixel level, directly on the acquired rasters (01_raw_data/ghsl/
    agbh_2018_v3.tif, anbh_2018_v3.tif), independent of the aggregation
    logic in script 02 -- so a bug in script 02 cannot mask a problem
    that already existed in the source data, and vice versa.
  - Cell level, on the aggregated output of script 02
    (02_processed_data/ghsl_layers_v3/agbh_anbh_grid_500m.parquet).

These are hard assertions (pytest), not descriptive statistics: the test
suite fails loudly if a physically required relationship is violated
beyond a documented, tiny, disclosed tolerance.

Usage
-----
    python -m pytest scripts/revision_v3/tests/test_identity_range.py -v
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config as cfg

# GDAL binary/env locations are centralised in config.py -- see the comment
# there (QGIS_GDAL_BIN_DIR, QGIS_GDAL_ENV) for why.
_GDAL_BIN_DIR = cfg.QGIS_GDAL_BIN_DIR
_GDAL_ENV = cfg.QGIS_GDAL_ENV

# Documented, disclosed tolerance for the single known mosaic-seam rounding
# artefact found during acquisition (script 01 / audit_report.md): one pixel
# with AGBH exceeding ANBH by 0.0505 m out of 48,384,000 pixels. Any
# violation count above this is a real regression, not a rounding artefact.
KNOWN_PIXEL_VIOLATIONS_MAX = 1
KNOWN_PIXEL_VIOLATION_MAGNITUDE_MAX_M = 0.06


def _gdal_translate_envi(tif_path: Path, raw_path: Path, ot: str) -> None:
    import os
    env = dict(**os.environ, **_GDAL_ENV)
    r = subprocess.run(
        [f"{_GDAL_BIN_DIR}/gdal_translate", "-of", "ENVI", "-ot", ot, str(tif_path), str(raw_path)],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0, r.stderr


def _read_dims(tif_path: Path) -> tuple[int, int]:
    import os
    env = dict(**os.environ, **_GDAL_ENV)
    out = subprocess.run([f"{_GDAL_BIN_DIR}/gdalinfo", str(tif_path)],
                          capture_output=True, text=True, env=env).stdout
    line = next(l for l in out.splitlines() if l.strip().startswith("Size is"))
    w, h = line.split("is", 1)[1].split(",")
    return int(w), int(h)


@pytest.fixture(scope="module")
def pixel_arrays(tmp_path_factory):
    agbh_tif = cfg.GHSL_RAW / "agbh_2018_v3.tif"
    anbh_tif = cfg.GHSL_RAW / "anbh_2018_v3.tif"
    if not (agbh_tif.exists() and anbh_tif.exists()):
        pytest.skip("Run scripts/revision_v3/01_acquire_ghsl_height.py first.")
    tmp = tmp_path_factory.mktemp("identity_test")
    w, h = _read_dims(agbh_tif)
    agbh_raw, anbh_raw = tmp / "agbh.bin", tmp / "anbh.bin"
    _gdal_translate_envi(agbh_tif, agbh_raw, "Float32")
    _gdal_translate_envi(anbh_tif, anbh_raw, "Float32")
    agbh = np.fromfile(agbh_raw, dtype="float32").reshape(h, w)
    anbh = np.fromfile(anbh_raw, dtype="float32").reshape(h, w)
    return agbh, anbh


# ============================================================ PIXEL LEVEL =

def test_pixel_no_nan(pixel_arrays):
    agbh, anbh = pixel_arrays
    assert np.isnan(agbh).sum() == 0, "Unexpected NaN in source AGBH raster"
    assert np.isnan(anbh).sum() == 0, "Unexpected NaN in source ANBH raster"


def test_pixel_no_negative(pixel_arrays):
    agbh, anbh = pixel_arrays
    assert (agbh < 0).sum() == 0, "Unexpected negative AGBH pixel value"
    assert (anbh < 0).sum() == 0, "Unexpected negative ANBH pixel value"


def test_pixel_zero_colocation(pixel_arrays):
    """AGBH==0 must coincide exactly with ANBH==0 (both encode 'no building');
    a mismatch would mean the 0/0 built-fraction case is not well-defined."""
    agbh, anbh = pixel_arrays
    agbh_zero, anbh_zero = (agbh == 0), (anbh == 0)
    mismatch_a = int((agbh_zero & ~anbh_zero).sum())
    mismatch_b = int((anbh_zero & ~agbh_zero).sum())
    assert mismatch_a == 0, f"{mismatch_a} pixels have AGBH==0 but ANBH!=0"
    assert mismatch_b == 0, (
        f"{mismatch_b} pixels have ANBH==0 but AGBH!=0 -- built_fraction "
        "would be 0/0 undefined at these pixels, not simply 0."
    )


def test_pixel_agbh_le_anbh_identity(pixel_arrays):
    """Core physical identity: AGBH = ANBH x built_fraction, built_fraction
    in [0,1], therefore AGBH must not exceed ANBH except by a documented,
    tiny, disclosed tolerance (see module docstring)."""
    agbh, anbh = pixel_arrays
    violation = agbh > anbh
    n_violation = int(violation.sum())
    assert n_violation <= KNOWN_PIXEL_VIOLATIONS_MAX, (
        f"{n_violation} pixels violate AGBH<=ANBH, expected at most "
        f"{KNOWN_PIXEL_VIOLATIONS_MAX} (documented mosaic-seam rounding artefact)"
    )
    if n_violation:
        magnitude = (agbh - anbh)[violation].max()
        assert magnitude <= KNOWN_PIXEL_VIOLATION_MAGNITUDE_MAX_M, (
            f"Violation magnitude {magnitude} m exceeds documented tolerance "
            f"{KNOWN_PIXEL_VIOLATION_MAGNITUDE_MAX_M} m"
        )


def test_pixel_built_fraction_range(pixel_arrays):
    agbh, anbh = pixel_arrays
    mask = anbh > 0
    bf = agbh[mask] / anbh[mask]
    n_total = mask.sum()
    n_above_1 = int((bf > 1.001).sum())  # tolerance matches the one documented violation
    assert n_above_1 <= KNOWN_PIXEL_VIOLATIONS_MAX, (
        f"{n_above_1} of {n_total} pixels have built_fraction > 1.001"
    )
    assert bf.min() >= 0, "Negative built_fraction found"


def test_pixel_range_plausible(pixel_arrays):
    """AGBH and ANBH should be within physically plausible bounds for
    buildings on Earth (loose bounds -- this catches unit/scale errors,
    e.g. the six-orders-of-magnitude error found in the excluded OB 2.5D
    Temporal dataset during the v2 audit, not fine-grained outliers)."""
    agbh, anbh = pixel_arrays
    assert agbh.max() < 200, f"AGBH max {agbh.max()} exceeds plausible bound (200 m)"
    assert anbh.max() < 400, f"ANBH max {anbh.max()} exceeds plausible bound (400 m)"
    assert 0.001 < agbh[agbh > 0].mean() < 50, "AGBH mean over built pixels implausible"
    assert 0.5 < anbh[anbh > 0].mean() < 100, "ANBH mean over built pixels implausible"


# ============================================================= CELL LEVEL =

@pytest.fixture(scope="module")
def cell_df() -> pd.DataFrame:
    path = cfg.GHSL_LAYERS_V3 / "agbh_anbh_grid_500m.parquet"
    if not path.exists():
        pytest.skip("Run scripts/revision_v3/02_process_agbh_anbh_grid.py first.")
    return pd.read_parquet(path)


def test_cell_row_count(cell_df):
    assert len(cell_df) == 961_858, "Output must cover every national grid cell, including nodata rows"


def test_cell_nodata_disclosed_separately_from_zero(cell_df):
    """Cells with zero overlapping source pixels (true nodata) must be NaN,
    never silently coerced to 0."""
    nodata_rows = cell_df[cell_df.n_source_pixels == 0]
    assert nodata_rows.cell_agbh_mean_m.isna().all(), "Nodata cells must have NaN AGBH, not 0"
    assert nodata_rows.cell_anbh_derived_m.isna().all(), "Nodata cells must have NaN ANBH, not 0"
    valid_rows = cell_df[cell_df.n_source_pixels > 0]
    assert valid_rows.cell_agbh_mean_m.notna().all(), "Cells with source pixels must not be NaN"


def test_cell_agbh_le_anbh(cell_df):
    valid = cell_df[cell_df.n_source_pixels > 0]
    violation = valid[valid.cell_agbh_mean_m > valid.cell_anbh_derived_m + 1e-6]
    assert len(violation) == 0, f"{len(violation)} cells have AGBH_mean > ANBH_derived"


def test_cell_built_fraction_range(cell_df):
    valid = cell_df[cell_df.n_source_pixels > 0]
    assert valid.cell_built_fraction_mean.min() >= 0
    assert valid.cell_built_fraction_mean.max() <= 1.001, (
        f"built_fraction max {valid.cell_built_fraction_mean.max()} exceeds 1"
    )


def test_cell_no_extreme_implied_height_outliers(cell_df):
    """Regression guard for the v2 bug: dividing an already-ANBH-like
    quantity by coverage a second time produced 'implied height' outliers
    up to 263 m. Correctly derived ANBH must not reproduce this."""
    valid = cell_df[cell_df.n_source_pixels > 0]
    assert valid.cell_anbh_derived_m.max() < 60, (
        f"cell_anbh_derived_m max {valid.cell_anbh_derived_m.max()} m is implausibly high "
        "-- check for a double-division bug (see v2 Limitation 3 in the prior manuscript)."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
