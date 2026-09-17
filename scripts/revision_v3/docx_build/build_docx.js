/*
 * Paper 2 v3 -- Word (.docx) build.
 *
 * Unlike the v2 build (scripts/revision/docx_build/), which hand-transcribes
 * every paragraph into docx-js calls across five chained files, this script
 * parses 04_outputs/manuscript/paper2_v3_draft.md directly and generates the
 * docx from it. That means the .docx can never drift from the manuscript
 * text -- rerunning this script after any manuscript edit reproduces it
 * exactly, with no manual re-transcription step to forget.
 *
 * Page setup, fonts, and heading styles match the v2 build's "Discover
 * Cities" journal look (Times New Roman, US Letter, 1in margins) for
 * consistency, since this is a revision of that same target format.
 *
 * Usage: node build_docx.js
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ImageRun, ShadingType, VerticalAlign,
} = require("docx");
const { imageSize } = require("image-size");

const ROOT = path.resolve(__dirname, "../../..");
const MD_PATH = path.join(ROOT, "04_outputs/manuscript/paper2_v3_draft.md");
const OUT_PATH = path.join(ROOT, "04_outputs/manuscript/paper2_v3.docx");

// manuscript figure number -> actual image file (numbers don't match filenames
// 1:1 -- fig03_typology_map.png is manuscript Fig. 4, fig04_regression_
// coefficients.png is manuscript Fig. 3; see run_all.py's KNOWN_OUTPUTS comments)
const FIGURE_MAP = {
  "1": path.join(ROOT, "04_outputs/paper2_v3/figures/fig01_anbh_map.png"),
  "2": path.join(ROOT, "04_outputs/paper2_v3/figures/fig02_agbh_map.png"),
  "3": path.join(ROOT, "04_outputs/paper2_v3/figures/fig04_regression_coefficients.png"),
  "4": path.join(ROOT, "04_outputs/paper2_v3/figures/fig03_typology_map.png"),
  "A1": path.join(ROOT, "04_outputs/paper2_figures/figA1_two_dimensions_schematic.png"),
  "A2": path.join(ROOT, "04_outputs/paper2_v3/figures/figA2_sample_flow.png"),
};

const PAGE_W = 12240, PAGE_H = 15840, MARGIN = 1440; // US Letter, 1in margins (DXA)
const CONTENT_W_TWIPS = PAGE_W - 2 * MARGIN;

// ---------------------------------------------------------------- inline markup --
// Handles **bold** and `code` as non-overlapping top-level spans (the one
// case of bold-wrapping-code in the source manuscript was fixed at the
// source rather than handled here -- see audit_report.md).
function inlineRuns(text, extra = {}) {
  const tokens = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter((s) => s.length > 0);
  return tokens.map((seg) => {
    if (seg.startsWith("**") && seg.endsWith("**")) {
      return new TextRun({ text: seg.slice(2, -2), bold: true, size: 22, ...extra });
    }
    if (seg.startsWith("`") && seg.endsWith("`")) {
      return new TextRun({ text: seg.slice(1, -1), font: "Courier New", size: 20, ...extra });
    }
    return new TextRun({ text: seg, size: 22, ...extra });
  });
}

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 260, after: 120 } });
}
function p(text) {
  return new Paragraph({ children: inlineRuns(text), spacing: { after: 160, line: 276 }, alignment: AlignmentType.JUSTIFIED });
}
function frontMatterPara(text, size) {
  return new Paragraph({ children: inlineRuns(text, { size }), alignment: AlignmentType.CENTER, spacing: { after: 80 } });
}
function figCaptionPara(text) {
  return new Paragraph({ children: inlineRuns(text, { size: 20 }), alignment: AlignmentType.CENTER, spacing: { before: 80, after: 240 } });
}
function tableCaptionPara(text) {
  return new Paragraph({ children: inlineRuns(text, { size: 20 }), spacing: { before: 200, after: 100 } });
}
function refPara(text) {
  return new Paragraph({ children: inlineRuns(text), spacing: { after: 160, line: 264 }, indent: { left: 360, hanging: 360 } });
}
function listPara(text) {
  return new Paragraph({ children: inlineRuns(text), spacing: { after: 140, line: 276 }, alignment: AlignmentType.JUSTIFIED, indent: { left: 360, hanging: 360 } });
}

function cell(text, opts = {}) {
  const { bold = false, width, shade, align = AlignmentType.LEFT } = opts;
  return new TableCell({
    width: width ? { size: width, type: WidthType.DXA } : undefined,
    shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: align, children: inlineRuns(String(text), { size: 18, bold }) })],
  });
}

function dataTable(headers, rows) {
  const nCols = headers.length;
  const colWidths = Array(nCols).fill(Math.floor(CONTENT_W_TWIPS / nCols));
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((htext, i) => cell(htext, { bold: true, width: colWidths[i], shade: "D9E2F3", align: AlignmentType.CENTER })),
  });
  const bodyRows = rows.map((r, ri) => new TableRow({
    children: r.map((c, i) => cell(c, { width: colWidths[i], shade: ri % 2 === 1 ? "F2F2F2" : undefined })),
  }));
  return new Table({
    width: { size: CONTENT_W_TWIPS, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...bodyRows],
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: "888888" },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: "888888" },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "BBBBBB" },
      insideVertical: { style: BorderStyle.NONE },
    },
  });
}

function figureImage(filePath, maxWidthPx = 380) {
  const buf = fs.readFileSync(filePath);
  const dim = imageSize(buf);
  let w = maxWidthPx, h = Math.round(maxWidthPx * dim.height / dim.width);
  const maxH = 600;
  if (h > maxH) { h = maxH; w = Math.round(maxH * dim.width / dim.height); }
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 160, after: 40 },
    children: [new ImageRun({ data: buf, transformation: { width: w, height: h }, type: "png" })],
  });
}

// ------------------------------------------------------------------- parse --
const raw = fs.readFileSync(MD_PATH, "utf8");
const lines = raw.split("\n");

function isTableRow(line) { return /^\s*\|.*\|\s*$/.test(line); }
function isTableSep(line) { return /^\s*\|[\s:|-]+\|\s*$/.test(line); }
function splitRow(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
}

const children = [];
let inReferences = false;
let inFrontMatter = false; // between the title line and the first "---"
let i = 0;

while (i < lines.length) {
  const line = lines[i];
  const trimmed = line.trim();

  if (trimmed === "") { i++; continue; }

  if (trimmed === "---") { inFrontMatter = false; i++; continue; }

  // Title
  if (line.startsWith("# ")) {
    children.push(new Paragraph({
      children: [new TextRun({ text: line.slice(2).trim(), bold: true, size: 30 })],
      alignment: AlignmentType.CENTER, spacing: { after: 240 },
    }));
    inFrontMatter = true;
    i++; continue;
  }

  if (inFrontMatter) {
    // author line first (larger), affiliations/email smaller
    const size = /^\*?Corresponding author/.test(trimmed) || /^[¹²³⁴⁵]/.test(trimmed) ? 18 : 22;
    children.push(frontMatterPara(trimmed, size));
    i++; continue;
  }

  // H1 ("## ...")
  if (line.startsWith("## ")) {
    const title = line.slice(3).trim();
    if (title === "References") inReferences = true;
    children.push(h1(title));
    i++; continue;
  }

  // H2 ("### ...")
  if (line.startsWith("### ")) {
    children.push(h2(line.slice(4).trim()));
    i++; continue;
  }

  // Figure caption -> insert image, then caption
  const figMatch = trimmed.match(/^\*\*Fig\.\s*([0-9]+|A[0-9]+)\*\*/);
  if (figMatch) {
    const imgPath = FIGURE_MAP[figMatch[1]];
    if (imgPath && fs.existsSync(imgPath)) children.push(figureImage(imgPath));
    else console.warn("WARNING: no image found for Fig.", figMatch[1]);
    children.push(figCaptionPara(trimmed));
    i++; continue;
  }

  // Table caption
  if (/^\*\*Table\s*[0-9]+\*\*/.test(trimmed)) {
    children.push(tableCaptionPara(trimmed));
    i++; continue;
  }

  // Table block
  if (isTableRow(line) && i + 1 < lines.length && isTableSep(lines[i + 1])) {
    const headers = splitRow(line);
    i += 2;
    const rows = [];
    while (i < lines.length && isTableRow(lines[i])) { rows.push(splitRow(lines[i])); i++; }
    children.push(dataTable(headers, rows));
    children.push(new Paragraph({ text: "", spacing: { after: 160 } }));
    continue;
  }

  if (inReferences) {
    children.push(refPara(trimmed));
    i++; continue;
  }

  // Numbered list item (Limitations, Section 6.4)
  if (/^\d+\.\s/.test(trimmed)) {
    children.push(listPara(trimmed));
    i++; continue;
  }

  // Plain paragraph
  children.push(p(trimmed));
  i++;
}

// ------------------------------------------------------------------- pack --
const doc = new Document({
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: PAGE_H },
        margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN },
      },
    },
    children,
  }],
  styles: {
    default: { document: { run: { font: "Times New Roman", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Times New Roman" }, paragraph: { spacing: { before: 320, after: 160 } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, italics: true, font: "Times New Roman" }, paragraph: { spacing: { before: 260, after: 120 } } },
    ],
  },
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT_PATH, buf);
  console.log("WROTE", OUT_PATH, buf.length, "bytes");
}).catch((e) => { console.error("PACK ERROR", e); process.exit(1); });
