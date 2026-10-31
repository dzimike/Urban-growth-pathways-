"""
Round-2 edits to the submitted manuscript (Reviewer #4's qualifications).

The submitted manuscript ("New Revision/manuscript (1).docx") was edited by
the authors in Word after it was generated from Markdown, so it -- not
paper2_v3_draft.md -- is the master. This script applies the round-2
changes to a COPY of it with python-docx, leaving the original untouched:

  * replaces the text of specific paragraphs (each target paragraph is
    checked to contain the expected opening words, so an edit cannot land on
    the wrong paragraph if the document changes);
  * keeps bold caption labels ("Table 3", "Table A6") by editing only the
    run after the label;
  * replaces the embedded images for Fig. 2 (sample flow) and Fig. 5
    (coefficients), resizing Fig. 5 to its new aspect ratio;
  * appends Appendix Tables A7-A9, built by copying Table A6's XML so that
    borders, fonts and cell margins match.

Usage: python round2_edit_manuscript.py
"""
from __future__ import annotations
import copy
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Emu

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT.parent / "New Revision" / "manuscript (1).docx"
OUT = ROOT.parent / "New Revision" / "manuscript_round2.docx"
T = ROOT / "04_outputs/paper2_v3/tables"
FIG = ROOT / "04_outputs/paper2_v3/figures"
SCRATCH_ABSTRACT = Path(sys.argv[1]) if len(sys.argv) > 1 else None

doc = Document(SRC)
P = doc.paragraphs


def set_text(idx: int, expect_start: str, new: str, keep_label: bool = False) -> None:
    """Replace paragraph text. keep_label: keep run 0 (bold caption label)."""
    p = P[idx]
    assert p.text.startswith(expect_start), (idx, p.text[:80])
    runs = p.runs
    if keep_label:
        label = runs[0].text
        assert new.startswith(label), (label, new[:30])
        runs[1].text = new[len(label):]
        for r in runs[2:]:
            r._r.getparent().remove(r._r)
    else:
        fmts = {(r.bold, r.italic, r.font.superscript) for r in runs}
        assert len(fmts) == 1, (idx, fmts)          # uniform formatting only
        runs[0].text = new
        for r in runs[1:]:
            r._r.getparent().remove(r._r)


def append_text(idx: int, expect_start: str, extra: str) -> None:
    set_text(idx, expect_start, P[idx].text + " " + extra)


def replace_in(idx: int, old: str, new: str) -> None:
    t = P[idx].text
    assert t.count(old) == 1, (idx, old[:60], t.count(old))
    set_text(idx, t[:20], t.replace(old, new))


# ------------------------------------------------------------------ text ---
set_text(0, "Building height and horizontal expansion",
         "What satellite building height data reveal about urban form and horizontal expansion in Ghana")

ABSTRACT = (Path(__file__).with_name("round2_abstract.txt")).read_text().strip()
assert len(ABSTRACT.split()) <= 250, len(ABSTRACT.split())
set_text(7, "Urban expansion in sub-Saharan Africa", ABSTRACT)

replace_in(8, "urban form regimes", "urban form")

replace_in(12, "Third, how sensitive is a height-expansion typology to the concentrated distribution of ANBH values?",
           "Third, how sensitive is a descriptive height-expansion classification to the concentrated distribution "
           "of ANBH values and to cells without a GHSL height signal?")

set_text(14, "The contribution is therefore both substantive and methodological.",
         "The contribution is therefore primarily methodological, with a descriptive substantive component. "
         "Methodologically, the study shows that the distinction between net height and gross built density "
         "materially affects inference, identifies an unusually concentrated ANBH distribution around 2.5 m, and "
         "shows that cells without a height signal, rather than measured differences in height, account for most of "
         "the apparent agreement between height and prior expansion. Substantively, it provides a national "
         "description of how earlier horizontal development relates to estimated building height in Ghana. The "
         "implications are relevant not only to Ghana but also to comparative work using GHSL height products "
         "across rapidly urbanising regions.")

set_text(41, "The two rasters were clipped to the Ghana analysis envelope",
         "The two rasters were clipped to a rectangular analysis envelope enclosing Ghana at 100 m resolution. "
         "Across the envelope's 48,384,000 pixels, neither raster contained NaN or negative values, and zero values "
         "coincided at 42,251,166 pixels. For positive ANBH, built fraction was calculated as AGBH divided by ANBH; "
         "where both products were zero, built fraction was assigned a value of zero. AGBH was less than or equal to "
         "ANBH at 48,383,999 pixels (99.999998%). The single exception, which lies inside Ghana, was an AGBH value "
         "0.0505 m greater than ANBH, producing a built fraction of 1.0066; it is consistent with minor rounding or "
         "mosaicking effects and is retained in the audit rather than truncated. Restricting the checks to the "
         "28,293,000 pixels inside Ghana's boundary (GADM level 0) gives the same result: no mismatched zeros and "
         "the same single exception. Because built fraction is itself derived as AGBH/ANBH, AGBH ≤ ANBH is the only "
         "condition the identity imposes on the released rasters. The check therefore confirms that the two products "
         "are mutually consistent; it is not an independent validation of either height or built fraction.")

append_text(49, "We model log(1 + ANBH)",
            "Table 3 and Fig. 5 report OLS estimates; conventional OLS standard errors assume independent residuals. "
            "Because the OLS residuals are spatially autocorrelated, the full model is also estimated with "
            "heteroskedasticity-robust standard errors and with standard errors clustered in 10 km and 25 km square "
            "blocks, and compared with the spatial error model (Table A7).")

append_text(53, "The typology combines derived ANBH",
            "Cells with ANBH = 0 have no GHSL height signal rather than a measured height of zero; they enter the "
            "low-height class by construction, so the classification and the full model are repeated on cells with "
            "a positive signal (Table A8). Because diagonal shares have different chance baselines under two-, three- "
            "and four-class splits (approximately 50%, 33% and 25%), agreement between the axes is also summarised "
            "by the diagonal share expected under independence and by Cohen's κ (Table A9).")

replace_in(60, "The pixel-level checks confirm that AGBH and ANBH satisfy their expected mathematical relationship almost everywhere.",
           "The pixel-level checks show that AGBH and ANBH are mutually consistent almost everywhere (Section 4.4).")
replace_in(60, "Among the 6,132,834 pixels with positive ANBH, 2,458,649 (40.1%) fall within the narrow 2.50-2.55 m interval.",
           "Within Ghana's boundary, 1,529,150 of the 3,976,154 pixels with positive ANBH (38.5%) fall within the "
           "narrow 2.50-2.55 m interval; over the full rectangular acquisition envelope, which also includes parts of "
           "neighbouring countries, the share is 40.1% (2,458,649 of 6,132,834).")

replace_in(65, "while others may be structures not captured by GHSL.",
           "while others may be structures not captured by GHSL. These cells should be read as having no GHSL height "
           "signal, not as measured zero-height buildings; their treatment affects both the regression and the "
           "typology (Sections 5.2 and 5.5).")

set_text(74, "Table 3", "Table 3 Model C OLS coefficients with conventional standard errors (log(1+ANBH), n=296,668); "
                        "spatially robust and spatial error estimates are in Table A7", keep_label=True)

append_text(76, "Built coverage in 2010 has the largest standardised coefficient",
            "The intervals in Table 3 use conventional OLS standard errors, which assume independent residuals; given "
            "the residual spatial dependence (Moran's I = 0.293), they are too narrow. Clustering standard errors in "
            "25 km blocks widens them by factors of 2.6 to 9.2 (Table A7; Fig. 5). Built coverage in 2010 and both "
            "growth terms remain clearly positive under clustered errors and in the spatial error model, whereas the "
            "95% interval for CBD distance then includes zero; in the spatial error model the CBD-distance "
            "coefficient is larger in magnitude (-0.026 against -0.004). Excluding the 43,412 cells without a height "
            "signal raises the OLS R² from 0.235 to 0.770 and the standardised coefficient on 2010 built coverage "
            "from 0.35 to 0.67 (Table A8). Because ANBH rises above its 2.5 m concentration mainly where built "
            "density is high, this stronger association is consistent with both a real height gradient and the "
            "retrieval process itself, and is not interpreted as evidence of a stronger substantive relationship.")

set_text(78, "Fig. 5.",
         "Fig. 5. Standardised coefficients from Model C (log(1+ANBH), n = 296,668) with 95% intervals under three "
         "estimators: OLS with conventional standard errors (as in Table 3), OLS with standard errors clustered in "
         "25 km blocks, and the spatial error model. Conventional OLS intervals ignore the residual spatial dependence "
         "reported in Table 2. All VIFs are below 4.")

set_text(93, "At the median thresholds",
         "At the median thresholds, the high-height/high-expansion and low-height/low-expansion classes together "
         "contain 64.2% of cells (Fig. 6; Table 5). Diagonal shares are not comparable across classification schemes "
         "without reference to chance: if the two axes were independent, the expected diagonal would be 50.0% for "
         "the median split, 33.3% for tertiles and 25.0% for quartiles. The observed shares (64.2%, 43.2% and 46.6%; "
         "Table A5) correspond to Cohen's κ of 0.28, 0.15 and 0.29 (Table A9). Replacing ANBH with AGBH while "
         "holding the sample and horizontal-growth axis fixed gives κ of 0.80, 0.69 and 0.69 (diagonal shares "
         "89.8%, 79.4% and 76.4%). This comparison is descriptive rather than a clean test of height measurement "
         "because AGBH includes built fraction, which correlates with the horizontal-growth axis (r = 0.81). Within "
         "the ANBH typology, median estimated height varies little across classes (2.39-2.52 m), while class means "
         "range from 1.35 to 3.47 m. The low-height/low-expansion class contains 43,111 of the 43,412 cells without "
         "a GHSL height signal, 45.1% of the class, so its position reflects signal availability as well as "
         "estimated height. This matters for the classification as a whole: among the 253,256 cells with a positive "
         "signal, the median split agrees with prior expansion at chance level (diagonal share 50.2%; κ = 0.005; "
         "Table A8). The above-chance agreement of the full classification therefore arises mainly from the absence "
         "of a height signal in sparsely built cells rather than from differences in measured height. Tertile "
         "cut-points differ by only 5 mm, further illustrating the limited additional height information gained "
         "from finer ANBH splits. The four classes are best read as a description of where GHSL records a height "
         "signal and where prior expansion occurred, not as independently validated height groups.")

replace_in(97, "It is modest evidence that the typology captures a geographical pattern beyond the median split itself.",
           "It is modest evidence that the typology captures a geographical pattern beyond the median split itself, "
           "although part of that pattern may reflect where GHSL records a height signal.")

replace_in(103, "The pixel-level consistency checks establish that the two products are mathematically coherent over Ghana;",
           "The pixel-level checks establish that the two released rasters are mutually consistent over Ghana; "
           "because built fraction is derived from them, this is not an independent validation of either height or "
           "coverage, and")

replace_in(105, "Built coverage in 2010 is the strongest pre-2018 predictor of ANBH in 2018.",
           "Built coverage in 2010 is the strongest pre-2018 predictor of ANBH in 2018 under conventional, spatially "
           "clustered and spatial error estimation.")
append_text(105, "Built coverage in 2010 is the strongest",
            "Part of the association also reflects where GHSL records a height signal at all, because cells without "
            "a signal are concentrated in sparsely built areas.")

replace_in(107, "The main regression coefficients remain similar under alternative weights and at 250 m, 500 m and 1 km resolutions.",
           "The main regression coefficients remain similar under alternative weights and at 250 m, 500 m and 1 km "
           "resolutions. Their conventional OLS intervals, however, understate uncertainty: clustering standard "
           "errors in 25 km blocks widens them up to ninefold, and the small CBD-distance association is not "
           "distinguishable from zero under that adjustment.")

replace_in(109, "40.1% of positive pixels fall within a 0.05 m interval",
           "38.5% of positive pixels inside Ghana (40.1% over the rectangular acquisition envelope) fall within a "
           "0.05 m interval")
append_text(109, "Several features of the data",
            "A further 43,412 footprint cells (14.6%) carry no GHSL height signal. They are coded as ANBH = 0 in the "
            "models and fall almost entirely in the low-height/low-expansion class, so that class and the regression "
            "combine signal availability with estimated height.")

replace_in(110, "built fraction bconstruction", "built fraction by construction")

append_text(112, "Spatial inference introduces further uncertainty.",
            "Conventional OLS standard errors (Table 3) assume independent residuals and are therefore too narrow; "
            "clustered and spatial error alternatives are reported in Table A7.")

append_text(115, "National analysis of building height in Ghana",
            "The four-class classification is descriptive: its agreement with prior expansion arises mainly from "
            "cells without a height signal, and it should not be read as a set of validated height regimes.")

append_text(116, "Future work should test",
            "More detailed robustness analysis of cells without a height signal and of classification precision "
            "would also strengthen future applications.")

CODE_REF = (Path(__file__).with_name("round2_code_reference.txt")).read_text().strip()
assert P[124].text.startswith("GHSL-BUILT-H R2023A (AGBH and ANBH)")
P[124].add_run(" " + CODE_REF)          # new run: keeps the existing runs (incl. the 10 pt facts.json) intact

set_text(134, "During preparation of the manuscript",
         "During preparation of the manuscript, the authors used Claude (Anthropic) to assist with data-processing "
         "code, statistical-model implementation and English-language editing. All numerical results were produced "
         "by the deposited code and verified by the authors, who retain responsibility for all analytical decisions, "
         "interpretations and conclusions. No AI tool was used to fabricate or alter data, results or citations.")

# ---------------------------------------------------------------- images ---
shapes = doc.inline_shapes
assert len(shapes) == 6


def replace_image(shape_idx: int, png: Path, rescale_height: bool) -> None:
    from PIL import Image
    shape = shapes[shape_idx]
    rid = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
    part = doc.part.related_parts[rid]
    part._blob = png.read_bytes()
    if rescale_height:
        w, h = Image.open(png).size
        new_h = int(shape.width * h / w)
        shape.height = Emu(new_h)
        # keep the pic's own extent in step with the inline extent
        ext = shape._inline.graphic.graphicData.pic.spPr.xfrm.ext
        ext.cy = new_h


replace_image(1, FIG / "figA2_sample_flow.png", rescale_height=True)          # Fig. 2
replace_image(4, FIG / "fig04_regression_coefficients.png", rescale_height=True)  # Fig. 5

# --------------------------------------------------------- appendix tables -
body = doc.element.body
template_tbl = doc.tables[11]._tbl                       # Table A6
template_caption = P[206]._p
template_note = P[207]._p
assert P[206].text.startswith("Table A6") and P[207].text.startswith("Note:")
blank_p = template_note.getnext()                         # empty paragraph after the A6 note
assert blank_p.tag == qn("w:p")


def _set_cell(tc, text: str) -> None:
    ps = tc.findall(qn("w:p"))
    for extra in ps[1:]:
        tc.remove(extra)
    p = ps[0]
    rs = p.findall(qn("w:r"))
    for extra in rs[1:]:
        p.remove(extra)
    rs[0].find(qn("w:t")).text = text


def build_table(header: list[str], rows: list[list[str]]):
    tbl = copy.deepcopy(template_tbl)
    n = len(header)
    weights = [2.0] + [1.0] * (n - 1)
    widths = [int(9360 * w / sum(weights)) for w in weights]
    grid = tbl.find(qn("w:tblGrid"))
    for g in list(grid):
        grid.remove(g)
    for wdt in widths:
        g = copy.deepcopy(template_tbl.find(qn("w:tblGrid"))[1])
        g.set(qn("w:w"), str(wdt))
        grid.append(g)
    trs = tbl.findall(qn("w:tr"))
    hdr_tpl, body_tpls = trs[0], trs[1:3]
    for tr in trs:
        tbl.remove(tr)

    def make_row(tpl, values):
        tr = copy.deepcopy(tpl)
        tcs = tr.findall(qn("w:tc"))
        first_tpl, other_tpl = tcs[0], tcs[1]
        for tc in tcs:
            tr.remove(tc)
        for j, v in enumerate(values):
            tc = copy.deepcopy(first_tpl if j == 0 else other_tpl)
            tc.find(qn("w:tcPr")).find(qn("w:tcW")).set(qn("w:w"), str(widths[j]))
            _set_cell(tc, v)
            tr.append(tc)
        return tr

    tbl.append(make_row(hdr_tpl, header))
    for k, r in enumerate(rows):
        tbl.append(make_row(body_tpls[k % 2], r))
    return tbl


def make_para(tpl, label: str | None, text: str):
    p = copy.deepcopy(tpl)
    rs = p.findall(qn("w:r"))
    for extra in rs[2:]:
        p.remove(extra)
    if label is None:                       # note paragraph: single run
        rs[0].find(qn("w:t")).text = text
        rs[0].find(qn("w:t")).set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        p.remove(rs[1])
    else:
        rs[0].find(qn("w:t")).text = label
        t = rs[1].find(qn("w:t"))
        t.text = text
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return p


def fmt(x, d=3):
    return f"{x:.{d}f}".replace("-", "−")


# A7 -- Model C under alternative uncertainty estimates
inf = pd.read_csv(T / "T_round2_inference.csv")
est_cols = ["OLS, conventional SE (Table 3, Fig. 5)", "OLS, HC1 robust SE", "OLS, SE clustered by 10 km blocks",
            "OLS, SE clustered by 25 km blocks", "Spatial error model (GM_Error, Queen)"]
names = {"log_distance_to_nearest_cbd_m": "Log distance to CBD", "built_coverage_2010": "Built coverage 2010",
         "growth_1975_2010": "Growth 1975–2010", "growth_2010_2015": "Growth 2010–2015"}
a7_rows = []
for v, lab in names.items():
    row = [lab]
    for e in est_cols:
        r = inf[(inf.estimator == e) & (inf.variable == v)].iloc[0]
        d = 4 if v.startswith("log_distance") else 3
        row.append(f"{fmt(r.coef, d)} ({fmt(r.se, d)})")
    a7_rows.append(row)
a7 = ("Table A7", " Model C (log(1+ANBH), n=296,668) under alternative uncertainty estimates. Cells show the coefficient "
      "with its standard error in parentheses.",
      ["Predictor", "OLS, conventional SE", "OLS, HC1 robust SE", "OLS, 10 km clusters", "OLS, 25 km clusters",
       "Spatial error model"], a7_rows,
      "Note: OLS coefficients are identical in the four OLS columns; only the standard errors differ. Clusters are "
      "square blocks of 10 km (2,503 clusters) and 25 km (440 clusters) on the UTM grid. Under 25 km clustering, "
      "standard errors are 2.6 to 9.2 times the conventional values and the 95% interval for CBD distance includes "
      "zero. The spatial error model uses row-standardised Queen contiguity weights (λ = 0.407).")

# A8 -- cells with a GHSL height signal
zs = pd.read_csv(T / "T_round2_zero_signal.csv")
allc = zs[zs["sample"] == "All cells (as published)"].set_index("variable")
sig = zs[zs["sample"] == "Cells with a GHSL height signal (ANBH > 0)"].set_index("variable")
typ = zs[zs["sample"] == "Typology on ANBH > 0 cells"].set_index("variable")["ols_coef"]
r0a, r0s = allc.iloc[0], sig.iloc[0]
a8_rows = [
    ["Cells", f"{int(r0a.n):,}", f"{int(r0s.n):,}"],
    ["OLS R²", fmt(r0a.ols_r2), fmt(r0s.ols_r2)],
    ["Spatial error pseudo-R²", fmt(r0a.sem_pseudo_r2), fmt(r0s.sem_pseudo_r2)],
    ["λ (spatial error model)", fmt(r0a.sem_lambda), fmt(r0s.sem_lambda)],
    ["Moran's I, OLS residuals", fmt(r0a.moran_i_ols_resid), fmt(r0s.moran_i_ols_resid)],
    ["Moran's I, spatially filtered residuals", fmt(r0a.moran_i_sem_filtered), fmt(r0s.moran_i_sem_filtered)],
] + [[f"Standardised OLS coefficient: {names[v]}", fmt(allc.loc[v, "ols_standardized"]), fmt(sig.loc[v, "ols_standardized"])]
     for v in names] + [
    ["Median-split typology: diagonal share, %", "64.2", f"{typ['diagonal_pct']:.1f}"],
    ["Median-split typology: Cohen's κ", "0.284", "0.005"],
    ["Range of class median ANBH, m", "2.39–2.52", f"{typ['class_median_min']:.2f}–{typ['class_median_max']:.2f}"],
]
a8 = ("Table A8", " Model C and the median-split typology on all cells and on cells with a GHSL height signal "
      "(ANBH > 0).", ["Quantity", "All cells", "Cells with ANBH > 0"], a8_rows,
      "Note: Of the 43,412 cells without a height signal, 43,111 are in the Low height / Low prior expansion class "
      "(45.1% of that class) and 301 in the Low height / High prior expansion class. The typology on cells with "
      "ANBH > 0 re-derives both medians within that subset (ANBH 2.5012 m; prior expansion 0.0008), with ties "
      "assigned to the low class. Because ANBH departs from its 2.5 m concentration mainly where built density is "
      "high, the higher R² on signal-only cells is consistent with the retrieval process as well as with a real "
      "height gradient.")

# A9 -- diagonal shares against their chance baselines
ag = pd.read_csv(T / "T_round2_typology_agreement.csv")
a9_rows = [[r.vertical_axis.replace(" (primary)", "").replace(" (matched comparison)", " (matched)"),
            r.scheme.replace(" (2x2)", " (2 × 2)").replace(" (3x3)", " (3 × 3)").replace(" (4x4)", " (4 × 4)"),
            f"{r.observed_diagonal_pct:.1f}", f"{r.expected_diagonal_under_independence_pct:.1f}",
            f"{r.excess_over_chance_pct_points:.1f}", f"{r.cohens_kappa:.3f}"] for r in ag.itertuples()]
a9 = ("Table A9", " Diagonal shares of the height × prior-expansion classifications against their chance baselines, "
      "final analytical sample (n=296,668).",
      ["Height axis", "Scheme", "Observed diagonal, %", "Expected under independence, %", "Excess, points",
       "Cohen's κ"], a9_rows,
      "Note: The expected diagonal is the sum over classes of (row share × column share), from each scheme's own "
      "margins. Raw diagonal shares are therefore comparable across schemes only through κ. Tertile and quartile "
      "classes assign values equal to a cut-point to the upper class (Table A5). The AGBH rows hold the sample and "
      "the horizontal axis fixed and are not a clean test of height measurement (Section 5.5).")

anchor = blank_p
for label, cap, header, rows, note in [a7, a8, a9]:
    new_blank = copy.deepcopy(blank_p)
    cap_p = make_para(template_caption, label, cap)
    tbl = build_table(header, rows)
    note_p = make_para(template_note, None, note)
    for el in [cap_p, tbl, note_p, new_blank]:
        anchor.addnext(el)
        anchor = el

doc.save(OUT)
print("WROTE", OUT)
