import JSZip from "jszip";
import PptxGenJS from "pptxgenjs";
import fs from "node:fs/promises";
import path from "node:path";
import { blocks, plain, runs, tableRows } from "./blocks.mjs";
import { FONT } from "./layout.mjs";
export function measure(text, width, size = 16) {
  const units = [...plain(text)].reduce(
    (n, c) =>
      n + (c === "\n" ? width / size : c.codePointAt(0) > 255 ? 1 : 0.58),
    0,
  );
  return Math.max(1, Math.ceil(units / (width / size))) * size * 1.4;
}
export function presentationPlan(input) {
  const theme = {
    accent: "264D73",
    background: "FFFFFF",
    font: FONT,
    ratio: "wide",
    ...({
      minimal: { accent: "222222" },
      dark: { accent: "8DC7F0", background: "17212B" },
    }[input.theme?.preset] || {}),
    ...input.theme,
  };
  const width = theme.ratio === "standard" ? 720 : 960,
    height = 540;
  const pages = [];
  let tableId = 0;
  for (const [sourceIndex, source] of input.slides.entries()) {
    let current;
    const fresh = () => {
      current = {
        sourceSlide: sourceIndex + 1,
        title: source.title,
        notes: source.notes || "",
        items: [],
        width,
        height,
        theme,
      };
      pages.push(current);
      return 110;
    };
    let y = fresh();
    const bottom = 482,
      left = 48,
      bodyWidth = width - 96;
    function put(b, x = left, w = bodyWidth, flow = true) {
      const size = b.style?.size || 16;
      let h;
      if (b.type === "columns") {
        const count = b.columns.length,
          colw = (w - 18 * (count - 1)) / count,
          start = y;
        let end = start;
        for (let i = 0; i < count; i++) {
          y = start;
          for (const child of b.columns[i])
            put(child, x + i * (colw + 18), colw, false);
          end = Math.max(end, y);
        }
        y = end;
        return;
      }
      if (b.type === "page_break") {
        if (!flow)
          throw Error("Page break is not allowed inside slide columns");
        y = fresh();
        return;
      }
      if (b.type === "table") {
        const id = ++tableId;
        const cols = tableRows(b),
          weights = b.widths || Array(cols).fill(1),
          sum = weights.reduce((a, c) => a + c, 0),
          colw = weights.map((v) => (v * w) / sum),
          heights = b.rows.map((row) => {
            let c = 0;
            return Math.max(
              26,
              ...row.map((v) => {
                const cell = typeof v === "string" ? { text: v } : v;
                const ww = colw
                  .slice(c, c + (cell.col_span || 1))
                  .reduce((a, n) => a + n, 0);
                c += cell.col_span || 1;
                return (
                  measure(cell.text, ww - 12, cell.style?.size || size) + 12
                );
              }),
            );
          });
        let chunk = [],
          hs = [];
        for (let i = 0; i < b.rows.length; i++) {
          h = heights[i];
          if (h > bottom - 110 - heights[0])
            throw Error(
              "A table row is taller than a slide; split its text into shorter rows",
            );
          if (y + h > bottom) {
            if (chunk.length)
              current.items.push({
                block: { ...b, rows: chunk },
                x,
                y: y - hs.reduce((a, n) => a + n, 0),
                w,
                h: hs.reduce((a, n) => a + n, 0),
                tableId: id,
            rowHeights: hs,
              });
            if (!flow)
              throw Error(
                "Table exceeds column height; move it to a separate slide",
              );
            if (b.rows.some((r) => r.some((c) => (c?.row_span || 1) > 1)))
              throw Error(
                "Merged table crosses slides; divide it at unmerged rows",
              );
            y = fresh();
            chunk = i ? [b.rows[0]] : [];
            hs = i ? [heights[0]] : [];
            if (i) y += heights[0];
          }
          chunk.push(b.rows[i]);
          hs.push(h);
          y += h;
        }
        if (chunk.length)
          current.items.push({
            block: { ...b, rows: chunk },
            x,
            y: y - hs.reduce((a, n) => a + n, 0),
            w,
            h: hs.reduce((a, n) => a + n, 0),
            tableId: id,
            rowHeights: hs,
          });
        y += 12;
        return;
      }
      if (b.type === "list")
        h = b.items.reduce((n, t) => n + measure(t, w - 24, size) + 7, 0);
      else if (b.type === "image") {
        w = b.w ? b.w*72 : Math.min(w,b.image.width_cm*72/2.54);
        h = Math.min(280, (b.image.height_px / b.image.width_px) * w) +
          (b.image.caption ? 24 : 0);
      } else if (["chart", "shape", "media"].includes(b.type))
        h = b.h ? b.h * 72 : 260;
      else h = measure(b.text || "", w, size) + 10;
      if (b.h) h = b.h * 72;
      if (h > bottom - 110)
        throw Error("Block is taller than a slide; split its content");
      if (y + h > bottom) {
        if (!flow)
          throw Error("Column content exceeds slide; shorten or split it");
        y = fresh();
      }
      current.items.push({
        block: b,
        x: b.x !== undefined ? b.x * 72 : x,
        y: b.y !== undefined ? b.y * 72 : y,
        w: b.w ? b.w * 72 : w,
        h,
      });
      const item = current.items.at(-1);
      if (item.x + item.w > width - 20 || item.y + item.h > height - 30)
        throw Error("Slide object exceeds page bounds");
      y += h + 12;
    }
    const items = blocks(source);
    if (source.layout === "two-column" && items.length > 1)
      put({
        type: "columns",
        columns: [
          items.slice(0, Math.ceil(items.length / 2)),
          items.slice(Math.ceil(items.length / 2)),
        ],
      });
    else items.forEach((b) => put(b));
  }
  if (pages.length > 160)
    throw Error("Presentation exceeds 160 generated slides");
  return pages;
}
function pptRuns(text, base = {}) {
  return runs(text).map((r) => ({
    text: r.text,
    options: {
      fontFace: r.style?.font || base.font || FONT,
      fontSize: r.style?.size || base.size || 16,
      bold: r.style?.bold ?? base.bold,
      italic: r.style?.italic ?? base.italic,
      underline: r.style?.underline ?? base.underline,
      color: r.style?.color || base.color || "202830",
      ...(r.link ? { hyperlink: { url: r.link } } : {}),
    },
  }));
}
export async function renderPresentation(input, dir) {
  const pages = presentationPlan(input),
    p = new PptxGenJS();
  p.defineLayout({ name: "DOCUMENT", width: pages[0].width / 72, height: 7.5 });
  p.layout = "DOCUMENT";
  p.title = input.title;
  p.author = "SparkTalk";
  p.theme = {
    headFontFace: pages[0].theme.font,
    bodyFontFace: pages[0].theme.font,
    lang: "ko-KR",
  };
  p.defineSlideMaster({
    title: "DOCUMENT_MASTER",
    background: { color: pages[0].theme.background },
    objects: [
      {
        rect: {
          x: 0,
          y: 0,
          w: 0.16,
          h: 7.5,
          fill: { color: pages[0].theme.accent },
          line: { color: pages[0].theme.accent },
        },
      },
    ],
  });
  for (const [i, page] of pages.entries()) {
    const slide = p.addSlide("DOCUMENT_MASTER");
    slide.addText(page.title, {
      x: 0.65,
      y: 0.4,
      w: (page.width - 96) / 72,
      h: 0.85,
      fontSize: 26,
      bold: true,
      color: page.theme.accent,
      fit: "shrink",
    });
    if (page.notes) slide.addNotes(page.notes);
    for (const item of page.items) {
      const { block: b } = item,
        s = {font:page.theme.font,...b.style},
        pos = {
          x: item.x / 72,
          y: item.y / 72,
          w: item.w / 72,
          h: item.h / 72,
        };
      switch (b.type) {
        case "table":
          slide.addTable(
            b.rows.map((row, r) =>
              row.map((v) => {
                const c = typeof v === "string" ? { text: v } : v,
                  st = { ...s, ...c.style };
                return {
                  text: pptRuns(c.text, st),
                  options: {
                    colspan: c.col_span,
                    rowspan: c.row_span,
                    bold: r === 0,
                    fill: st.background || (r === 0 ? "E7EEF5" : "FFFFFF"),
                    color: st.color || "202830",
                    align: st.align || "left",
                    valign: st.valign || "top",
                    border: {
                      type: "solid",
                      color: st.border || "BBC7D1",
                      pt: 0.5,
                    },
                  },
                };
              }),
            ),
            {
              ...pos,
              fontFace: FONT,
              fontSize: s.size || 16,
              margin: 6,
              rowH: item.rowHeights.map((x) => x / 72),
              colW: b.widths
                ? b.widths.map(
                    (x) => (x / b.widths.reduce((a, n) => a + n, 0)) * pos.w,
                  )
                : undefined,
              autoPage: false,
            },
          );
          break;
        case "image": {
          const caption = b.image.caption ? 24 : 0;
          slide.addImage({
            data: "image/png;base64," + b.image.data,
            x: pos.x,
            y: pos.y,
            w: b.image.width_px / 96,
            h: b.image.height_px / 96,
            sizing: {
              type: b.fit || "contain",
              w: pos.w,
              h: pos.h - caption / 72,
            },
            ...(b.link ? { hyperlink: { url: b.link } } : {}),
          });
          if (caption)
            slide.addText(b.image.caption, {
              x: pos.x,
              y: pos.y + pos.h - 24 / 72,
              w: pos.w,
              h: 24 / 72,
              fontSize: 10,
              fontFace:page.theme.font,
              color:page.theme.preset==='dark'?'F2F5F8':'202830',
            });
          break;
        }
        case "chart": {
          const c = b.chart;
          slide.addChart(
            p.ChartType[c.kind],
            [
              ...(c.kind === "scatter"
                ? [{ name: "X", values: c.labels.map(Number) }]
                : []),
              ...c.series.map((s) => ({
                name: s.name,
                labels: c.labels,
                values: s.values,
              })),
            ],
            {
              ...pos,
              showTitle: Boolean(c.title),
              title: c.title,
              showLegend: c.legend !== false,
              showValue: false,
              catAxisLabelFontSize: 10,
              valAxisLabelFontSize: 10,
              chartColors: c.series.map(
                (s, i) =>
                  s.color || ["264D73", "438C8C", "D69B42", "9D6085"][i % 4],
              ),
              showBorder: false,
            },
          );
          break;
        }
        case "shape":
          slide.addShape(
            p.ShapeType[
              b.shape === "arrow"
                ? "rightArrow"
                : b.shape === "textbox"
                  ? "rect"
                  : b.shape
            ],
            {
              ...pos,
              fill: { color: b.fill || "E7EEF5" },
              line: { color: b.line || "264D73" },
            },
          );
          if (b.text)
            slide.addText(pptRuns(b.text, s), {
              ...pos,
              margin: 8,
              fit: "shrink",
            });
          break;
        case "media":
          slide.addMedia({
            ...pos,
            type: b.media.mime.startsWith("audio/") ? "audio" : "video",
            data: b.media.mime + ";base64," + b.media.data,
            extn:
              b.media.mime === "video/mp4"
                ? "mp4"
                : b.media.mime === "audio/wav"
                  ? "wav"
                  : "mp3",
          });
          break;
        case "list":
          slide.addText(
            b.items.flatMap((t, i) =>
              pptRuns(t, {
                color: page.theme.preset === "dark" ? "F2F5F8" : "202830",
                ...s,
              }).map((r, j) => ({
                ...r,
                options: {
                  ...r.options,
                  breakLine: j === runs(t).length - 1,
                  ...(j === 0
                    ? {
                        bullet: b.ordered
                          ? {
                              type: "number",
                              numberType: "arabicPeriod",
                              numberStartAt: i + 1,
                            }
                          : {},
                      }
                    : {}),
                },
              })),
            ),
            {
              ...pos,
              fontSize: s.size || 16,
              paraSpaceAfter: 7,
              margin: 0,
              fit: "shrink",
              valign: "top",
            },
          );
          break;
        case "paragraph":
        case "heading":
        case "quote":
          slide.addText(
            pptRuns(b.text, {
              color: page.theme.preset === "dark" ? "F2F5F8" : "202830",
              ...s,
              bold: b.type === "heading" || s.bold,
            }),
            {
              ...pos,
              align: s.align || "left",
              margin: 0,
              breakLine: false,
              fit: "shrink",
              valign: "top",
            },
          );
          break;
        default:
          throw Error("Unsupported PPTX block: " + b.type);
      }
    }
    slide.addText(`${i + 1} / ${pages.length}`, {
      x: (page.width - 140) / 72,
      y: 7.08,
      w: 1.3,
      h: 0.2,
      fontSize: 9,
      color: page.theme.preset === "dark" ? "F2F5F8" : "68798B",
      align: "right",
    });
  }
  const file = path.join(dir, "document.pptx");
  await p.writeFile({ fileName: file });
  return {
    name: "document.pptx",
    mime: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    presentation: await inspectPresentation(await fs.readFile(file), pages, input.slides.length),
    data: (await fs.readFile(file)).toString("base64"),
  };
}

// Read the saved PPTX, not just the requested input or pagination estimate.
export async function inspectPresentation(buffer, pages, inputCount) {
  const zip = await JSZip.loadAsync(buffer);
  const names = Object.keys(zip.files).filter(n => /^ppt\/slides\/slide\d+\.xml$/.test(n))
    .sort((a,b) => Number(a.match(/slide(\d+)/)[1])-Number(b.match(/slide(\d+)/)[1]));
  if (names.length !== pages.length) throw Error("PPTX slide count differs from render plan");
  const tables = new Map();
  for (const [i,name] of names.entries()) {
    const xml = await zip.file(name).async("string");
    const actual = [...xml.matchAll(/<a:tbl>([\s\S]*?)<\/a:tbl>/g)];
    const planned = pages[i].items.filter(item => item.block.type === "table");
    if (actual.length !== planned.length) throw Error("PPTX table count differs from render plan");
    actual.forEach((match,j) => {
      const item = planned[j];
      const widths = [...match[1].matchAll(/<a:gridCol\b[^>]*\bw="(\d+)"/g)].map(m => Number(m[1])/914400);
      const rowCount = [...match[1].matchAll(/<a:tr\b/g)].length;
      if (!widths.length || rowCount !== item.block.rows.length) throw Error("PPTX table structure mismatch");
      if (!tables.has(item.tableId)) tables.set(item.tableId, {table_id:item.tableId,source_slide:pages[i].sourceSlide,slide_numbers:[],rows_per_slide:[],column_widths_inches:widths});
      const table = tables.get(item.tableId);
      table.slide_numbers.push(i+1); table.rows_per_slide.push(rowCount);
    });
  }
  return {slide_count:names.length,input_slide_count:inputCount,
    tables:[...tables.values()].map(t => ({...t,split:t.slide_numbers.length>1})),
    evidence:"generated_pptx_structure",visually_verified:false};
}
