/*
 * Response to reviewers (v3) -- Word (.docx) build.
 *
 * Parses 04_outputs/manuscript/paper2_response_to_reviewers_v3.md directly,
 * the same approach used for the manuscript itself (build_docx.js), so the
 * .docx can never drift from the letter's text -- rerunning this script
 * after any edit to the letter reproduces the .docx exactly.
 *
 * Page setup and fonts match the manuscript build (Times New Roman, US
 * Letter, 1in margins) for a consistent submission package. The letter has
 * no figures and only one table (the summary-of-changes table at the end),
 * so this script is a trimmed-down version of build_docx.js's parser:
 * title, "##" section headings, "---" horizontal rules between sections,
 * plain paragraphs (including the bold reviewer-quote + "Response:" pairs),
 * and one Markdown table.
 *
 * Usage: node build_response_letter_docx.js
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType, VerticalAlign,
  LevelFormat,
} = require("docx");

const ROOT = path.resolve(__dirname, "../../..");
// Optional argument: path to another letter (e.g. the second-round response).
const MD_PATH = process.argv[2] ? path.resolve(process.argv[2])
  : path.join(ROOT, "04_outputs/manuscript/paper2_response_to_reviewers_v3.md");
const OUT_PATH = MD_PATH.replace(/\.md$/, ".docx");

const PAGE_W = 12240, PAGE_H = 15840, MARGIN = 1440; // US Letter, 1in margins (DXA)
const CONTENT_W_TWIPS = PAGE_W - 2 * MARGIN;

// ---------------------------------------------------------------- inline markup --
// **bold**, *italic* and `code` as non-overlapping spans.
function inlineRuns(text, extra = {}) {
  const tokens = text.split(/(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\^-?\d+)/g).filter((s) => s.length > 0);
  return tokens.map((seg) => {
    if (/^\^-?\d+$/.test(seg)) {  // exponent, e.g. "10^-6" -> 10 with superscript "\u22126"
      return new TextRun({ text: seg.slice(1).replace("-", "\u2212"), superScript: true, size: 22, ...extra });
    }
    if (seg.startsWith("**") && seg.endsWith("**")) {
      return new TextRun({ text: seg.slice(2, -2), bold: true, size: 22, ...extra });
    }
    if (seg.length > 2 && seg.startsWith("*") && seg.endsWith("*")) {
      return new TextRun({ text: seg.slice(1, -1), italics: true, size: 22, ...extra });
    }
    if (seg.startsWith("`") && seg.endsWith("`")) {
      return new TextRun({ text: seg.slice(1, -1), font: "Courier New", size: 20, ...extra });
    }
    return new TextRun({ text: seg, size: 22, ...extra });
  });
}

function titlePara(text) {
  return new Paragraph({
    children: [new TextRun({ text, bold: true, size: 30 })],
    alignment: AlignmentType.CENTER, spacing: { after: 240 },
  });
}
function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 100 } });
}
// Reviewer/editor comments are quoted verbatim as indented italic paragraphs.
function quote(text) {
  return new Paragraph({
    children: inlineRuns(text, { italics: true, color: "333333" }),
    indent: { left: 567, right: 283 }, spacing: { after: 80, line: 264 },
    border: { left: { style: BorderStyle.SINGLE, size: 12, color: "999999", space: 8 } },
  });
}
function bullet(text) {
  return new Paragraph({ children: inlineRuns(text), numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80, line: 276 } });
}
function p(text) {
  return new Paragraph({ children: inlineRuns(text), spacing: { after: 160, line: 276 }, alignment: AlignmentType.JUSTIFIED });
}
function hr() {
  return new Paragraph({
    text: "",
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "999999", space: 1 } },
    spacing: { before: 160, after: 200 },
  });
}
function tableCaptionPara(text) {
  return new Paragraph({ children: inlineRuns(text, { size: 20 }), spacing: { before: 200, after: 100 } });
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

// widthFractions: optional array of relative widths (defaults to equal columns)
function dataTable(headers, rows, widthFractions) {
  const nCols = headers.length;
  const fractions = widthFractions && widthFractions.length === nCols
    ? widthFractions
    : Array(nCols).fill(1 / nCols);
  const fracSum = fractions.reduce((a, b) => a + b, 0);
  const colWidths = fractions.map((f) => Math.floor((CONTENT_W_TWIPS * f) / fracSum));
  const headerRow = new TableRow({
    tableHeader: true,
    cantSplit: true,
    children: headers.map((htext, i) => cell(htext, { bold: true, width: colWidths[i], shade: "D9E2F3", align: AlignmentType.CENTER })),
  });
  const bodyRows = rows.map((r, ri) => new TableRow({
    cantSplit: true,
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

// ------------------------------------------------------------------- parse --
const raw = fs.readFileSync(MD_PATH, "utf8");
const lines = raw.split("\n");

function isTableRow(line) { return /^\s*\|.*\|\s*$/.test(line); }
function isTableSep(line) { return /^\s*\|[\s:|-]+\|\s*$/.test(line); }
function splitRow(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
}

const children = [];
let i = 0;

while (i < lines.length) {
  const line = lines[i];
  const trimmed = line.trim();

  if (trimmed === "") { i++; continue; }

  // Section divider
  if (trimmed === "---") { children.push(hr()); i++; continue; }

  // Title
  if (line.startsWith("# ")) {
    children.push(titlePara(line.slice(2).trim()));
    i++; continue;
  }

  // Sub-heading ("### ...")
  if (line.startsWith("### ")) {
    children.push(h2(line.slice(4).trim()));
    i++; continue;
  }

  // Quoted comment ("> ..."); a bare ">" is a paragraph break inside the quote
  if (/^>\s?/.test(line)) {
    const q = line.replace(/^>\s?/, "").trim();
    if (q) children.push(quote(q));
    i++; continue;
  }

  // Bulleted item ("- ...")
  if (/^- /.test(trimmed)) {
    children.push(bullet(trimmed.slice(2)));
    i++; continue;
  }

  // Section heading ("## ...")
  if (line.startsWith("## ")) {
    children.push(h1(line.slice(3).trim()));
    i++; continue;
  }

  // Table block (only the closing summary-of-changes table)
  if (isTableRow(line) && i + 1 < lines.length && isTableSep(lines[i + 1])) {
    const headers = splitRow(line);
    i += 2;
    const rows = [];
    while (i < lines.length && isTableRow(lines[i])) { rows.push(splitRow(lines[i])); i++; }
    // "Section" column narrow, "Nature of change" column wide
    const widthFractions = headers.length === 2 ? [0.18, 0.82] : undefined;
    children.push(dataTable(headers, rows, widthFractions));
    children.push(new Paragraph({ text: "", spacing: { after: 160 } }));
    continue;
  }

  // Plain paragraph (covers reviewer-quote / "Response:" pairs and body text)
  children.push(p(trimmed));
  i++;
}

// ------------------------------------------------------------------- pack --
const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
                 style: { paragraph: { indent: { left: 567, hanging: 283 } } } }],
    }],
  },
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
        run: { size: 23, bold: true, font: "Times New Roman" }, paragraph: { spacing: { before: 240, after: 100 } } },
    ],
  },
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT_PATH, buf);
  console.log("WROTE", OUT_PATH, buf.length, "bytes");
}).catch((e) => { console.error("PACK ERROR", e); process.exit(1); });
