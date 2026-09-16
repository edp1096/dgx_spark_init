import JSZip from "jszip";
import * as D from "docx";
import fs from "node:fs/promises";
import path from "node:path";
import { blocks, runs, plain } from "./blocks.mjs";
import { FONT } from "./layout.mjs";
import { imageSize } from "./document-images.mjs";
const twip = (n) => Math.round(n * 20);
function runOptions(r) {
  const s = r.style || {};
  return {
    text: r.text,
    font: s.font || FONT,
    size: (s.size || 11) * 2,
    bold: s.bold,
    italics: s.italic,
    underline: s.underline ? {} : undefined,
    strike: s.strike,
    color: s.color,
    shading: s.background ? { fill: s.background } : undefined,
  };
}
function rich(text, base = {}) {
  return runs(text || "").map((r) => {
    const merged = { ...r, style: { ...base, ...r.style } };
    const child = new D.TextRun(runOptions(merged));
    return r.link
      ? new D.ExternalHyperlink({ link: r.link, children: [child] })
      : child;
  });
}
function para(text, style = {}, options = {}) {
  return new D.Paragraph({
    children: rich(text, style),
    alignment: style.align,
    border: style.border
      ? Object.fromEntries(
          ["top", "bottom", "left", "right"].map((k) => [
            k,
            { color: style.border, style: D.BorderStyle.SINGLE, size: 6 },
          ]),
        )
      : undefined,
    indent: style.indent ? { left: twip(style.indent) } : undefined,
    spacing: {
      after: twip(style.spacing ?? 8),
      line: Math.round((style.line_spacing || 1.15) * 240),
    },
    ...options,
  });
}
export async function renderRichDocx(input, dir) {
  const headings = [];
  let headingIndex = 0;
  function collect(items) {
    for (const b of items) {
      if (b.type === "heading")
        headings.push({
          level: b.level || 1,
          title: plain(b.text),
          href: "doc_heading_" + headings.length,
        });
      for (const c of b.columns || []) collect(c);
      for (const r of b.rows || [])
        for (const c of r) if (c.blocks) collect(c.blocks);
    }
  }
  for (const section of input.sections) collect(blocks(section));
  const footnotes = {},
    comments = [];
  let ref = 0;
  const numbering = [
    {
      reference: "ordered",
      levels: [
        {
          level: 0,
          format: D.LevelFormat.DECIMAL,
          text: "%1.",
          alignment: D.AlignmentType.START,
          style: { paragraph: { indent: { left: 720, hanging: 260 } } },
        },
      ],
    },
  ];
  function content(items) {
    const out = [];
    for (const b of items) {
      const s = b.style || {};
      switch (b.type) {
        case "paragraph":
        case "quote":
          out.push(
            para(
              b.text,
              s,
              b.type === "quote"
                ? {
                    indent: { left: 720 },
                    border: {
                      left: {
                        color: "AABBCD",
                        size: 12,
                        style: D.BorderStyle.SINGLE,
                      },
                    },
                  }
                : {},
            ),
          );
          break;
        case "heading":
          out.push(
            new D.Paragraph({
              heading: D.HeadingLevel["HEADING_" + (b.level || 1)],
              children: [
                new D.Bookmark({
                  id: "doc_heading_" + headingIndex++,
                  children: rich(b.text, s),
                }),
              ],
            }),
          );
          break;
        case "list":
          for (const text of b.items)
            out.push(
              para(
                text,
                s,
                b.ordered
                  ? { numbering: { reference: "ordered", level: 0 } }
                  : { bullet: { level: 0 } },
              ),
            );
          break;
        case "table":
          out.push(
            new D.Table({
              width: { size: 100, type: D.WidthType.PERCENTAGE },
              columnWidths: b.widths?.map((x) =>
                Math.round((9360 * x) / b.widths.reduce((a, n) => a + n, 0)),
              ),
              rows: b.rows.map(
                (row, i) =>
                  new D.TableRow({
                    tableHeader: i === 0,
                    children: row.map((v) => {
                      const c = typeof v === "string" ? { text: v } : v,
                        st = { ...s, ...c.style };
                      return new D.TableCell({
                        columnSpan: c.col_span,
                        rowSpan: c.row_span,
                        shading: st.background
                          ? { fill: st.background }
                          : i === 0
                            ? { fill: "E7EEF5" }
                            : undefined,
                        verticalAlign: st.valign,
                        margins: {
                          top: twip(st.margin ?? 4),
                          bottom: twip(st.margin ?? 4),
                          left: twip(st.margin ?? 4),
                          right: twip(st.margin ?? 4),
                        },
                        borders: st.border
                          ? Object.fromEntries(
                              ["top", "bottom", "left", "right"].map((k) => [
                                k,
                                {
                                  style: D.BorderStyle.SINGLE,
                                  color: st.border,
                                  size: 6,
                                },
                              ]),
                            )
                          : undefined,
                        children: c.blocks
                          ? content(c.blocks)
                          : [para(c.text, { bold: i === 0, ...st })],
                      });
                    }),
                  }),
              ),
            }),
          );
          break;
        case "image": {
          const z = imageSize(b.image);
          const opts = {
            type: "png",
            data: Buffer.from(b.image.data, "base64"),
            transformation: {
              width: Math.round((z.width * 96) / 72),
              height: Math.round((z.height * 96) / 72),
            },
          };
          if (b.wrap && b.wrap !== "inline")
            opts.floating = {
              horizontalPosition: {
                relative: D.HorizontalPositionRelativeFrom.COLUMN,
                offset: Math.round((b.x || 0) * 914400),
              },
              verticalPosition: {
                relative: D.VerticalPositionRelativeFrom.PARAGRAPH,
                offset: 0,
              },
              wrap: {
                type:
                  b.wrap === "square"
                    ? D.TextWrappingType.SQUARE
                    : D.TextWrappingType.TOP_AND_BOTTOM,
              },
              allowOverlap: false,
            };
          out.push(
            new D.Paragraph({
              alignment: s.align,
              children: [new D.ImageRun(opts)],
            }),
          );
          if (b.image.caption) out.push(para(b.image.caption, { size: 10 }));
          break;
        }
        case "page_break":
          out.push(new D.Paragraph({ children: [new D.PageBreak()] }));
          break;
        case "toc":
          out.push(
            new D.TableOfContents("목차", {
              hyperlink: true,
              headingStyleRange: "1-6",
              cachedEntries: headings,
              beginDirty: true,
            }),
          );
          break;
        case "bookmark":
          out.push(
            new D.Paragraph({
              children: [
                new D.Bookmark({ id: b.name, children: rich(b.text) }),
              ],
            }),
          );
          break;
        case "footnote": {
          const id = ++ref;
          footnotes[id] = { children: [para(b.note)] };
          out.push(
            new D.Paragraph({
              children: [...rich(b.text), new D.FootnoteReferenceRun(id)],
            }),
          );
          break;
        }
        case "comment": {
          const id = ++ref;
          comments.push({
            id,
            author: "SparkTalk",
            initials: "ST",
            date: new Date(),
            children: [para(b.note)],
          });
          out.push(
            new D.Paragraph({
              children: [
                new D.CommentRangeStart(id),
                ...rich(b.text),
                new D.CommentRangeEnd(id),
                new D.TextRun({ children: [new D.CommentReference(id)] }),
              ],
            }),
          );
          break;
        }
        case "equation": {
          const children =
            b.numerator !== undefined
              ? [
                  new D.MathFraction({
                    numerator: [new D.MathRun(b.numerator)],
                    denominator: [new D.MathRun(b.denominator || "1")],
                  }),
                ]
              : [new D.MathRun(b.text)];
          out.push(new D.Paragraph({ children: [new D.Math({ children })] }));
          break;
        }
        case "checkbox":
          out.push(
            new D.Paragraph({
              children: [
                new D.CheckBox({ checked: Boolean(b.checked) }),
                ...rich(" " + (b.text || "")),
              ],
            }),
          );
          break;
        case "columns":
          out.push(
            new D.Table({
              width: { size: 100, type: D.WidthType.PERCENTAGE },
              borders: Object.fromEntries(
                [
                  "top",
                  "bottom",
                  "left",
                  "right",
                  "insideHorizontal",
                  "insideVertical",
                ].map((k) => [
                  k,
                  { style: D.BorderStyle.NONE, size: 0, color: "FFFFFF" },
                ]),
              ),
              rows: [
                new D.TableRow({
                  children: b.columns.map(
                    (c) => new D.TableCell({ children: content(c) }),
                  ),
                }),
              ],
            }),
          );
          break;
        default:
          throw Error("DOCX block unsupported: " + b.type);
      }
    }
    return out;
  }
  function hf(value, footer = false, numbers = false) {
    return new (footer ? D.Footer : D.Header)({
      children: [
        new D.Paragraph({
          alignment: D.AlignmentType.RIGHT,
          children: [
            ...rich(value || ""),
            ...(numbers
              ? [
                  new D.TextRun({
                    children: [
                      " ",
                      D.PageNumber.CURRENT,
                      " / ",
                      D.PageNumber.TOTAL_PAGES,
                    ],
                  }),
                ]
              : []),
          ],
        }),
      ],
    });
  }
  const sections = input.sections.map((s, i) => {
    const p = { ...input.page, ...s.page },
      size = { A4: [11906, 16838], A3: [16838, 23811], LETTER: [12240, 15840] }[
        p.size || "A4"
      ];
    return {
      properties: {
        type: p.break_before
          ? D.SectionType.NEXT_PAGE
          : D.SectionType.CONTINUOUS,
        titlePage: p.first_header !== undefined || p.first_footer !== undefined,
        page: {
          size: {
            width: size[0],
            height: size[1],
            orientation: p.orientation || "portrait",
          },
          margin: Object.fromEntries(
            ["top", "bottom", "left", "right"].map((k) => [
              k,
              twip(p.margin ?? 42),
            ]),
          ),
        },
        column: p.columns > 1 ? { count: p.columns, space: 360 } : undefined,
      },
      headers: {
        default: hf(p.header),
        ...(p.first_header !== undefined ? { first: hf(p.first_header) } : {}),
        ...(p.even_header !== undefined ? { even: hf(p.even_header) } : {}),
      },
      footers: {
        default: hf(p.footer, true, p.page_numbers),
        ...(p.first_footer !== undefined
          ? { first: hf(p.first_footer, true, p.page_numbers) }
          : {}),
        ...(p.even_footer !== undefined
          ? { even: hf(p.even_footer, true, p.page_numbers) }
          : {}),
      },
      children: [
        ...(i === 0
          ? [
              new D.Paragraph({
                text: input.title,
                heading: D.HeadingLevel.TITLE,
              }),
            ]
          : []),
        ...content(blocks(s)),
      ],
    };
  });
  const doc = new D.Document({
    title: input.title,
    creator: "SparkTalk",
    styles: {
      default: {
        document: { run: { font: FONT, size: 22 } },
        ...Object.fromEntries(
          Array.from({ length: 6 }, (_, i) => [
            "heading" + (i + 1),
            { paragraph: { outlineLevel: i, keepNext: true } },
          ]),
        ),
      },
    },
    numbering: { config: numbering },
    footnotes,
    comments: { children: comments },
    features: { updateFields: true },
    evenAndOddHeaderAndFooters: true,
    sections,
  });
  let data = await D.Packer.toBuffer(doc);
  if (input.sections.some((s) => blocks(s).some((b) => b.type === "toc"))) {
    // A plain complex TOC field is interoperable. The library's generic SDT
    // wrapper is imported as an input form by LibreOffice instead of a TOC.
    const zip = await JSZip.loadAsync(data);
    const xml = await zip.file("word/document.xml").async("string");
    zip.file(
      "word/document.xml",
      xml.replace(
        /<w:sdt><w:sdtPr><w:alias w:val="목차"\/><\/w:sdtPr><w:sdtContent>([\s\S]*?)<\/w:sdtContent><\/w:sdt>/g,
        "$1",
      ),
    );
    data = await zip.generateAsync({
      type: "nodebuffer",
      compression: "DEFLATE",
    });
  }
  await fs.writeFile(path.join(dir, "document.docx"), data);
  return {
    name: "document.docx",
    mime: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    data: data.toString("base64"),
  };
}
