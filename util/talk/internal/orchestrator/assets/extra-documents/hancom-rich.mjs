import { blocks, runs, plain, tableRows } from "./blocks.mjs";
import { imageSize } from "./document-images.mjs";
import { FONT } from "./layout.mjs";
const j = JSON.stringify;
function ok(raw) {
  const x = JSON.parse(raw);
  if (x.ok === false)
    throw Error(x.error || x.message || x.reason || "Hancom operation failed");
  return x;
}
export function populateHancom(doc, input) {
  let force = false;
  const fontCache = new Map();
  function chars(s = {}) {
    if (!fontCache.has(s.font || FONT))
      fontCache.set(s.font || FONT, doc.findOrCreateFontId(s.font || FONT));
    return {
      fontId: fontCache.get(s.font || FONT),
      fontSize: Math.round((s.size || 11) * 100),
      bold: s.bold || false,
      italic: s.italic || false,
      underline: s.underline || false,
      strikethrough: s.strike || false,
      ...(s.color ? { textColor: "#" + s.color } : {}),
      ...(s.background ? { shadeColor: "#" + s.background } : {}),
    };
  }
  function paras(s = {}) {
    return {
      alignment: s.align || "left",
      lineSpacing: Math.round((s.line_spacing || 1.6) * 100),
      marginLeft: Math.round((s.indent || 0) * 100),
      spacingAfter: Math.round((s.spacing ?? 8) * 100),
    };
  }
  function paragraph(t = "", s = {}) {
    let p = doc.getParagraphCount(0) - 1;
    if (force || doc.getParagraphLength(0, p) > 0) {
      p++;
      ok(doc.insertParagraph(0, p));
    }
    force = false;
    let offset = 0;
    for (const r of runs(t)) {
      if (r.link) throw Error("Hancom rich-text links are not supported");
      if (r.text) {
        ok(doc.insertText(0, p, offset, r.text));
        ok(
          doc.applyCharFormat(
            0,
            p,
            offset,
            offset + r.text.length,
            j(chars({ ...s, ...r.style })),
          ),
        );
        offset += r.text.length;
      }
    }
    ok(doc.applyParaFormat(0, p, j(paras(s))));
    return p;
  }
  const p = input.page || {},
    size = { A4: [59528, 84189], A3: [84189, 119055], LETTER: [61200, 79200] }[
      p.size || "A4"
    ];
  ok(
    doc.setPageDef(
      0,
      j({
        width: size[0],
        height: size[1],
        landscape: p.orientation === "landscape",
        marginLeft: (p.margin ?? 42) * 100,
        marginRight: (p.margin ?? 42) * 100,
        marginTop: (p.margin ?? 42) * 100,
        marginBottom: (p.margin ?? 42) * 100,
      }),
    ),
  );
  if (p.columns) ok(doc.setColumnDef(0, p.columns, 0, 1, 1800));
  for (const [header, key, apply] of [
    [true, "header", 0],
    [false, "footer", 0],
    [true, "even_header", 2],
    [false, "even_footer", 2],
  ])
    if (p[key] !== undefined) {
      ok(doc.createHeaderFooter(0, header, apply));
      ok(doc.insertTextInHeaderFooter(0, header, apply, 0, 0, p[key]));
    }
  if (p.first_header !== undefined || p.first_footer !== undefined)
    throw Error(
      "Hancom first-page header variants require additional engine support",
    );
  paragraph(input.title, { size: 20, bold: true });
  for (const section of input.sections) {
    if (section.page) {
      const { break_before, ...opts } = section.page;
      if (Object.keys(opts).length)
        throw Error("Hancom page options belong at document level");
      if (break_before) {
        const idx = paragraph(" ");
        ok(doc.breakAtCursor(0, idx, 0, "section"));
        force = true;
      }
    }
    for (const b of blocks(section)) {
      const s = b.style || {};
      switch (b.type) {
        case "paragraph":
        case "quote":
        case "heading":
          paragraph(b.text, {
            ...(b.type === "heading" ? { size: 16, bold: true } : {}),
            ...(b.type === "quote" ? { indent: 18, italic: true } : {}),
            ...s,
          });
          break;
        case "list": {
          const id = b.ordered
            ? doc.ensureDefaultNumbering()
            : doc.ensureDefaultBullet("•");
          for (const t of b.items) {
            const idx = paragraph(t, s);
            ok(
              doc.applyParaFormat(
                0,
                idx,
                j({
                  ...paras(s),
                  headType: b.ordered ? "Number" : "Bullet",
                  numberingId: id,
                  paraLevel: 0,
                }),
              ),
            );
          }
          break;
        }
        case "table": {
          const cols = tableRows(b),
            anchor = paragraph(" "),
            t = ok(
              doc.createTable(
                0,
                anchor,
                doc.getParagraphLength(0, anchor),
                b.rows.length,
                cols,
              ),
            );
          ok(
            doc.setTableProperties(
              0,
              t.paraIdx,
              t.controlIdx,
              j({
                pageBreak: 2,
                repeatHeader: true,
                treatAsChar: false,
                textWrap: "TopAndBottom",
                vertRelTo: "Para",
                horzRelTo: "Para",
              }),
            ),
          );
          const used = Array.from({ length: b.rows.length }, () =>
              Array(cols).fill(false),
            ),
            merges = [];
          for (let r = 0; r < b.rows.length; r++) {
            let c = 0;
            for (const v of b.rows[r]) {
              while (used[r][c]) c++;
              const cell = typeof v === "string" ? { text: v } : v,
                cs = cell.col_span || 1,
                rs = cell.row_span || 1,
                st = { bold: r === 0, ...s, ...cell.style };
              let offset = 0;
              for (const run of runs(cell.text)) {
                ok(
                  doc.insertTextInCell(
                    0,
                    t.paraIdx,
                    t.controlIdx,
                    r * cols + c,
                    0,
                    offset,
                    run.text,
                  ),
                );
                if (run.text)
                  ok(
                    doc.applyCharFormatInCell(
                      0,
                      t.paraIdx,
                      t.controlIdx,
                      r * cols + c,
                      0,
                      offset,
                      offset + run.text.length,
                      j(chars({ ...st, ...run.style })),
                    ),
                  );
                offset += run.text.length;
              }
              ok(
                doc.applyParaFormatInCell(
                  0,
                  t.paraIdx,
                  t.controlIdx,
                  r * cols + c,
                  0,
                  j(paras(st)),
                ),
              );
              const prop = {
                paddingLeft: (st.margin ?? 4) * 100,
                paddingRight: (st.margin ?? 4) * 100,
                paddingTop: (st.margin ?? 4) * 100,
                paddingBottom: (st.margin ?? 4) * 100,
                verticalAlign:
                  st.valign === "center" ? 1 : st.valign === "bottom" ? 2 : 0,
                ...(st.background
                  ? { fillType: "solid", fillColor: "#" + st.background }
                  : {}),
                ...(st.border
                  ? Object.fromEntries(
                      ["Left", "Right", "Top", "Bottom"].map((k) => [
                        "border" + k,
                        { type: 1, width: 1, color: "#" + st.border },
                      ]),
                    )
                  : {}),
              };
              if (b.widths)
                prop.width = Math.round(
                  ((size[0] - 2 * (p.margin ?? 42) * 100) * b.widths[c]) /
                    b.widths.reduce((a, n) => a + n, 0),
                );
              ok(
                doc.setCellProperties(
                  0,
                  t.paraIdx,
                  t.controlIdx,
                  r * cols + c,
                  j(prop),
                ),
              );
              for (let y = r; y < r + rs; y++)
                for (let x = c; x < c + cs; x++) used[y][x] = true;
              if (cs > 1 || rs > 1) merges.push([r, c, r + rs - 1, c + cs - 1]);
              c += cs;
            }
          }
          for (const m of merges.reverse())
            ok(doc.mergeTableCells(0, t.paraIdx, t.controlIdx, ...m));
          force = true;
          break;
        }
        case "image": {
          const idx = paragraph(" "),
            z = imageSize(b.image),
            im = ok(
              doc.insertPicture(
                0,
                idx,
                0,
                "",
                Buffer.from(b.image.data, "base64"),
                Math.round(z.width * 100),
                Math.round(z.height * 100),
                b.image.width_px,
                b.image.height_px,
                "png",
                "",
              ),
            );
          ok(
            doc.setPictureProperties(
              0,
              im.paraIdx,
              im.controlIdx,
              j({
                treatAsChar: !b.wrap || b.wrap === "inline",
                textWrap: b.wrap === "square" ? "Square" : "TopAndBottom",
                vertRelTo: "Para",
                horzRelTo: "Para",
                vertOffset: Math.round((b.y || 0) * 7200),
                horzOffset: Math.round((b.x || 0) * 7200),
              }),
            ),
          );
          force = true;
          if (b.image.caption) paragraph(b.image.caption, { size: 10 });
          break;
        }
        case "page_break": {
          const idx = paragraph(" ");
          ok(doc.insertPageBreak(0, idx, 0));
          force = true;
          break;
        }
        case "bookmark": {
          const idx = paragraph(b.text || "");
          ok(doc.addBookmark(0, idx, 0, b.name));
          break;
        }
        case "footnote":
        case "endnote": {
          const idx = paragraph(b.text),
            n = ok(
              doc[b.type === "footnote" ? "insertFootnote" : "insertEndnote"](
                0,
                idx,
                doc.getParagraphLength(0, idx),
              ),
            );
          ok(
            doc.insertTextInFootnote(0, n.paraIdx, n.controlIdx, 0, 0, b.note),
          );
          break;
        }
        case "equation": {
          const idx = paragraph(" "),
            script =
              b.numerator !== undefined
                ? `{${b.numerator}} over {${b.denominator || "1"}}`
                : b.text;
          ok(
            doc.insertEquation(
              0,
              idx,
              0,
              script,
              Math.round((s.size || 12) * 100),
              parseInt(s.color || "000000", 16),
            ),
          );
          force = true;
          break;
        }
        case "shape": {
          const idx = paragraph(" "),
            shape = ok(
              doc.createShapeControl(
                j({
                  sectionIdx: 0,
                  paraIdx: idx,
                  charOffset: 0,
                  width: Math.round((b.w || 3) * 7200),
                  height: Math.round((b.h || 1) * 7200),
                  horzOffset: Math.round((b.x || 0) * 7200),
                  vertOffset: 0,
                  treatAsChar: true,
                  textWrap: "TopAndBottom",
                  shapeType:
                    b.shape === "rect"
                      ? "rectangle"
                      : b.shape === "arrow"
                        ? "connector-straight-arrow"
                        : b.shape,
                }),
              ),
            );
          if (b.text) {
            if (b.shape !== "textbox")
              ok(doc.setTextBoxAt(shape.paraIdx, shape.controlIdx, true));
            ok(
              doc.insertTextInCell(
                0,
                shape.paraIdx,
                shape.controlIdx,
                0,
                0,
                0,
                plain(b.text),
              ),
            );
          }
          ok(
            doc.setShapeProperties(
              0,
              shape.paraIdx,
              shape.controlIdx,
              j({
                fillType: "solid",
                fillColor: "#" + (b.fill || "E7EEF5"),
                lineColor: "#" + (b.line || "264D73"),
              }),
            ),
          );
          force = true;
          break;
        }
        default:
          throw Error("Unsupported Hancom block: " + b.type);
      }
    }
  }
}
