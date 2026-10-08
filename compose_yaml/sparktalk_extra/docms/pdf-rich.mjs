import { blocks, runs, plain, tableRows } from "./blocks.mjs";
import { presentationPlan } from "./presentation.mjs";
import { FONT } from "./layout.mjs";
import { imageSize } from "./document-images.mjs";
const palette = ["#264D73", "#438C8C", "#D69B42", "#9D6085"];
function text(t, base = {}) {
  return runs(t || "").map((r) => {
    const s = { ...base, ...r.style };
    return {
      text: r.text,
      bold: s.bold,
      italics: s.italic,
      decoration: s.underline
        ? "underline"
        : s.strike
          ? "lineThrough"
          : undefined,
      color: s.color ? "#" + s.color : undefined,
      background: s.background ? "#" + s.background : undefined,
      fontSize: s.size,
      link: r.link,
    };
  });
}
const escape = (s) =>
  String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
export function chartSvg(c, width = 700, height = 260) {
  const pad = 40,
    w = width - 80,
    h = height - 65,
    values = c.series.flatMap((s) => s.values),
    lo = Math.min(0, ...values),
    hi = Math.max(1, ...values),
    y = (v) => 15 + h - ((v - lo) / (hi - lo)) * h,
    zero = y(0);
  let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><rect width="${width}" height="${height}" fill="white"/>`;
  if (["pie", "doughnut"].includes(c.kind)) {
    const vals = c.series[0].values,
      total = vals.reduce((a, n) => a + n, 0),
      cx = width / 2,
      cy = height / 2,
      r = Math.min(h, w) / 2;
    let a = -Math.PI / 2;
    vals.forEach((v, i) => {
      if (!v) return;
      const end = a + (v / total) * Math.PI * 2;
      if (v === total)
        svg += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${palette[i % 4]}"/>`;
      else
        svg += `<path d="M ${cx} ${cy} L ${cx + r * Math.cos(a)} ${cy + r * Math.sin(a)} A ${r} ${r} 0 ${end - a > Math.PI ? 1 : 0} 1 ${cx + r * Math.cos(end)} ${cy + r * Math.sin(end)} Z" fill="${palette[i % 4]}"/>`;
      a = end;
    });
    if (c.kind === "doughnut")
      svg += `<circle cx="${cx}" cy="${cy}" r="${r / 2}" fill="white"/>`;
  } else {
    svg += `<path d="M ${pad} 15 V ${h + 15} M ${pad} ${zero} H ${width - pad}" stroke="#68798B" fill="none"/>`;
    c.series.forEach((s, j) => {
      const color = s.color ? "#" + s.color : palette[j % 4],
        points = [];
      s.values.forEach((v, i) => {
        const x =
          c.kind === "scatter"
            ? pad +
              ((Number(c.labels[i]) - Math.min(...c.labels.map(Number))) /
                (Math.max(...c.labels.map(Number)) -
                  Math.min(...c.labels.map(Number)) || 1)) *
                w
            : pad + ((i + 0.5) * w) / c.labels.length;
        points.push([x, y(v)]);
        if (c.kind === "bar") {
          const bw = w / c.labels.length / (c.series.length + 1);
          svg += `<rect x="${x - w / c.labels.length / 2 + j * bw}" y="${Math.min(y(v), zero)}" width="${bw * 0.9}" height="${Math.abs(zero - y(v))}" fill="${color}"/>`;
        } else if (c.kind === "scatter")
          svg += `<circle cx="${x}" cy="${y(v)}" r="4" fill="${color}"/>`;
      });
      if (c.kind === "area")
        svg += `<polygon points="${points[0][0]},${zero} ${points.map((p) => p.join(",")).join(" ")} ${points.at(-1)[0]},${zero}" fill="${color}" fill-opacity=".3"/>`;
      if (["line", "area"].includes(c.kind))
        svg += `<polyline points="${points.map((p) => p.join(",")).join(" ")}" stroke="${color}" stroke-width="2" fill="none"/>`;
    });
  }
  return svg + "</svg>";
}
function pdfTable(b) {
  const cols = tableRows(b),
    body = Array.from({ length: b.rows.length }, () => Array(cols).fill(null));
  for (let r = 0; r < b.rows.length; r++) {
    let c = 0;
    for (const v of b.rows[r]) {
      while (body[r][c] !== null) c++;
      const cell = typeof v === "string" ? { text: v } : v,
        s = { ...b.style, ...cell.style },
        cs = cell.col_span || 1,
        rs = cell.row_span || 1;
      body[r][c] = {
        ...(cell.blocks
          ? { stack: cell.blocks.map(pdfBlock) }
          : { text: text(cell.text, s) }),
        colSpan: cs,
        rowSpan: rs,
        bold: r === 0 || s.bold,
        alignment: s.align,
        fillColor: s.background
          ? "#" + s.background
          : r === 0
            ? "#E7EEF5"
            : undefined,
        margin: s.margin ?? 4,
        borderColor: s.border ? Array(4).fill("#" + s.border) : undefined,
      };
      for (let y = r; y < r + rs; y++)
        for (let x = c; x < c + cs; x++)
          if (y !== r || x !== c) body[y][x] = {};
      c += cs;
    }
  }
  return {
    table: {
      headerRows: 1,
      widths: b.widths
        ? b.widths.map(
            (x) => (100 * x) / b.widths.reduce((a, n) => a + n, 0) + "%",
          )
        : Array(cols).fill("*"),
      body,
    },
    layout: {
      hLineWidth: () => 0.5,
      vLineWidth: () => 0.5,
      hLineColor: () => "#BBC7D1",
      vLineColor: () => "#BBC7D1",
    },
  };
}
export function pdfBlock(b) {
  const s = b.style || {},
    base = {
      font: FONT,
      fontSize: s.size || 11,
      alignment: s.align || "left",
      lineHeight: s.line_spacing || 1.15,
      margin: [s.indent || 0, 0, 0, s.spacing ?? 8],
    };
  switch (b.type) {
    case "paragraph":
    case "quote":
    case "heading":
      return {
        ...base,
        text: text(b.text, s),
        ...(b.type === "heading"
          ? {
              fontSize: s.size || 18 - (b.level || 1),
              bold: true,
              tocItem: true,
            }
          : {}),
        ...(b.type === "quote"
          ? { margin: [18, 4, 0, 10], italics: true }
          : {}),
      };
    case "list":
      return {
        ...base,
        [b.ordered ? "ol" : "ul"]: b.items.map((t) => ({ text: text(t, s) })),
      };
    case "table":
      return { ...base, ...pdfTable(b) };
    case "image": {
      const z = imageSize(b.image);
      return {
        stack: [
          {
            image: "data:image/png;base64," + b.image.data,
            width: z.width,
            height: z.height,
          },
          ...(b.image.caption ? [{ text: b.image.caption, fontSize: 10 }] : []),
        ],
        alignment: s.align,
        margin: [0, 6, 0, 10],
      };
    }
    case "chart":
      return {
        stack: [
          { text: b.chart.title || "", bold: true },
          { svg: chartSvg(b.chart), width: 480 },
          { text: b.chart.labels.join(" · "), fontSize: 9 },
          ...b.chart.series.map((s) => ({
            text: s.name + ": " + s.values.join(", "),
            fontSize: 9,
          })),
        ],
      };
    case "columns":
      return {
        columns: b.columns.map((c) => ({ width: "*", stack: c.map(pdfBlock) })),
        columnGap: 18,
      };
    case "page_break":
      return { text: "", pageBreak: "before" };
    case "toc":
      return { toc: { title: { text: "목차", bold: true, fontSize: 18 } } };
    case "bookmark":
      return { ...base, text: text(b.text), id: b.name };
    case "footnote":
    case "endnote":
    case "comment":
      return {
        stack: [
          { text: text(b.text, s) },
          {
            text: "↳ " + b.note,
            fontSize: 9,
            color: "#68798B",
            margin: [12, 0, 0, 8],
          },
        ],
      };
    case "equation":
      return {
        text:
          b.numerator !== undefined
            ? `${b.numerator} / ${b.denominator || "1"}`
            : b.text,
        italics: true,
        margin: [0, 4, 0, 8],
      };
    case "checkbox":
      return { text: (b.checked ? "☑ " : "☐ ") + (b.text || "") };
    case "shape": {
      const w = b.w ? b.w * 72 : 200,
        h = Math.max(12, (b.h ? b.h * 72 : 90) - (b.text ? 24 : 0)),
        fill = "#" + (b.fill || "E7EEF5"),
        line = "#" + (b.line || "264D73");
      const form =
        b.shape === "ellipse"
          ? `<ellipse cx="${w / 2}" cy="${h / 2}" rx="${w / 2 - 2}" ry="${h / 2 - 2}" fill="${fill}" stroke="${line}"/>`
          : b.shape === "line"
            ? `<line x1="1" y1="${h / 2}" x2="${w - 1}" y2="${h / 2}" stroke="${line}" stroke-width="2"/>`
            : b.shape === "arrow"
              ? `<polygon points="1,${h * 0.3} ${w * 0.7},${h * 0.3} ${w * 0.7},1 ${w - 1},${h / 2} ${w * 0.7},${h - 1} ${w * 0.7},${h * 0.7} 1,${h * 0.7}" fill="${fill}" stroke="${line}"/>`
              : `<rect x="1" y="1" width="${w - 2}" height="${h - 2}" fill="${fill}" stroke="${line}"/>`;
      return {
        stack: [
          {
            svg: `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}">${form}</svg>`,
            width: w,
          },
          ...(b.text
            ? [{ text: text(b.text), alignment: "center", fontSize: 12 }]
            : []),
        ],
      };
    }
    case "media":
      return {
        text:
          "[" +
          (b.media.mime.startsWith("audio/") ? "오디오" : "동영상") +
          "] " +
          (b.media.name || ""),
        color: "#68798B",
      };
    default:
      throw Error("Unsupported PDF block: " + b.type);
  }
}
export function richPDFDefinition(input, def) {
  if (input.format === "pptx") {
    const pages = presentationPlan(input);
    def.pageSize = { width: pages[0].width, height: 540 };
    def.pageMargins = [0, 0, 0, 0];
    def.footer = null;
    def.background = () => ({
      canvas: [
        {
          type: "rect",
          x: 0,
          y: 0,
          w: pages[0].width,
          h: 540,
          color: "#" + pages[0].theme.background,
        },
        {
          type: "rect",
          x: 0,
          y: 0,
          w: 11.52,
          h: 540,
          color: "#" + pages[0].theme.accent,
        },
      ],
    });
    def.content = [];
    pages.forEach((page, i) => {
      const stack = [
        {
          text: page.title,
          fontSize: 26,
          bold: true,
          color: "#" + page.theme.accent,
          absolutePosition: { x: 48, y: 29 },
          width: page.width - 96,
        },
      ];
      for (const item of page.items) {
        let node = pdfBlock(
          item.block.type === "shape"
            ? { ...item.block, w: item.w / 72, h: item.h / 72 }
            : item.block,
        );
        node.fontSize = item.block.style?.size || 16;
        node.margin = 0;
        node.color =
          page.theme.preset === "dark" && item.block.type !== "table"
            ? "#F2F5F8"
            : "#202830";
        if (item.block.type === "image") {
          const image = item.block.image,
            available = item.h - (image.caption ? 24 : 0),
            scale = Math.min(
              item.w / image.width_px,
              available / image.height_px,
            );
          node = {
            stack: [
              {
                image: "data:image/png;base64," + image.data,
                ...(item.block.fit === "cover"
                  ? { cover: { width: item.w, height: available } }
                  : {
                      width: image.width_px * scale,
                      height: image.height_px * scale,
                    }),
              },
              ...(image.caption ? [{ text: image.caption, fontSize: 10 }] : []),
            ],
          };
        }
        if (item.block.type === "chart")
          node = {
            stack: [
              { text: item.block.chart.title || "", fontSize: 12 },
              {
                svg: chartSvg(item.block.chart, item.w, item.h - 40),
                width: item.w,
              },
              { text: item.block.chart.labels.join(" · "), fontSize: 9 },
            ],
          };
        stack.push({
          absolutePosition: { x: item.x, y: item.y },
          columns: [{ width: item.w, ...node }],
        });
      }
      stack.push({
        text: `${i + 1} / ${pages.length}`,
        color: page.theme.preset === "dark" ? "#F2F5F8" : "#68798B",
        fontSize: 9,
        absolutePosition: { x: page.width - 130, y: 510 },
      });
      def.content.push({ pageBreak: i ? "before" : undefined, stack });
    });
    return def;
  }
  const p = input.page || {};
  def.pageSize = p.size || "A4";
  def.pageOrientation = p.orientation || "portrait";
  def.pageMargins = Array(4).fill(p.margin ?? 42);
  def.header = p.header
    ? (page) => ({
        text:
          page === 1
            ? (p.first_header ?? p.header)
            : page % 2 === 0
              ? (p.even_header ?? p.header)
              : p.header,
        margin: [42, 15, 42, 0],
        fontSize: 9,
      })
    : undefined;
  def.footer = (page, pages) => ({
    text: [
      page === 1
        ? (p.first_footer ?? p.footer ?? "")
        : page % 2 === 0
          ? (p.even_footer ?? p.footer ?? "")
          : p.footer || "",
      ...(p.page_numbers ? [`${page} / ${pages}`] : []),
    ].join(" "),
    alignment: "right",
    margin: [42, 12, 42, 0],
    fontSize: 9,
  });
  def.content = [
    { text: input.title, fontSize: 24, bold: true, margin: [0, 0, 0, 18] },
  ];
  for (const [i, s] of input.sections.entries()) {
    let content = blocks(s).map(pdfBlock);
    if (s.page?.columns > 1 || p.columns > 1) {
      const count = s.page?.columns || p.columns,
        n = Math.ceil(content.length / count);
      content = [
        {
          columns: Array.from({ length: count }, (_, i) => ({
            width: "*",
            stack: content.slice(i * n, (i + 1) * n),
          })),
          columnGap: 18,
        },
      ];
    }
    if (i && s.page?.break_before)
      def.content.push({
        text: "",
        pageBreak: "before",
        pageOrientation: s.page?.orientation || p.orientation || "portrait",
      });
    def.content.push(...content);
  }
  return def;
}
function excelValue(c) {
  if (c._documentDisplay !== undefined) return c._documentDisplay;
  let v = c.formula ? c.result : c.value;
  if (v === null || v === undefined) return "";
  if (v instanceof Date) return v.toISOString().slice(0, 10);
  if (v?.hyperlink) return v.text;
  if (v?.richText) return v.richText.map((x) => x.text).join("");
  return String(v);
}
export function spreadsheetPDFDefinition(input, def, wb) {
  def.content = [];
  let first = true;
  for (const s of input.sheets) {
    if (s.hidden) continue;
    const ws = wb.getWorksheet(s.name),
      landscape =
        s.orientation === "landscape" ||
        (!s.orientation && s.columns.length > 5),
      cols = s.columns
        .map((_, i) => i + 1)
        .filter((c) => !ws.getColumn(c).hidden),
      rows = Array.from({ length: s.rows.length + 1 }, (_, i) => i + 1).filter(
        (r) => !ws.getRow(r).hidden,
      ),
      body = [];
    const fontSize = cols.length > 12 ? 7 : 10;
    for (const r of rows) {
      const row = [];
      for (const c of cols) {
        const cell = ws.getCell(r, c),
          master = cell.master;
        if (cell.isMerged && cell.address !== master.address) {
          row.push({});
          continue;
        }
        const v = excelValue(cell),
          st = cell.style || {},
          node = {
            text: v,
            fontSize,
            bold: st.font?.bold,
            italics: st.font?.italic,
            color: st.font?.color?.argb
              ? "#" + st.font.color.argb.slice(-6)
              : undefined,
            fillColor: st.fill?.fgColor?.argb
              ? "#" + st.fill.fgColor.argb.slice(-6)
              : undefined,
            alignment: st.alignment?.horizontal,
            link: cell.value?.hyperlink,
          };
        if (cell.isMerged) {
          const merge = Object.values(ws._merges).find(
            (m) => m.model.top === r && m.model.left === c,
          )?.model;
          if (merge) {
            node.colSpan = cols.filter(
              (n) => n >= merge.left && n <= merge.right,
            ).length;
            node.rowSpan = rows.filter(
              (n) => n >= merge.top && n <= merge.bottom,
            ).length;
          }
        }
        for (const cond of s.conditional_formats || []) {
          const [start, end = start] = cond.range.split(":"),
            a = ws.getCell(start),
            z = ws.getCell(end),
            n = Number(v);
          if (r >= a.row && r <= z.row && c >= a.col && c <= z.col) {
            const yes = {
              greaterThan: n > cond.value,
              lessThan: n < cond.value,
              equal: n === cond.value,
              greaterThanOrEqual: n >= cond.value,
              lessThanOrEqual: n <= cond.value,
            }[cond.operator];
            if (yes && cond.style?.background)
              node.fillColor = "#" + cond.style.background;
            if (yes && cond.style?.color) node.color = "#" + cond.style.color;
          }
        }
        row.push(node);
      }
      body.push(row);
    }
    def.content.push({
      text: s.name,
      fontSize: 18,
      bold: true,
      pageBreak: first ? undefined : "before",
      pageOrientation: landscape ? "landscape" : "portrait",
      margin: [0, 0, 0, 12],
    });
    if (first) def.pageOrientation = landscape ? "landscape" : "portrait";
    first = false;
    def.content.push({
      table: { headerRows: 1, widths: cols.map(() => "*"), body },
      layout: {
        hLineWidth: () => 0.4,
        vLineWidth: () => 0.4,
        hLineColor: () => "#BBC7D1",
        vLineColor: () => "#BBC7D1",
      },
      margin: [0, 0, 0, 15],
    });
    for (const im of s.images || [])
      def.content.push(pdfBlock({ type: "image", image: im.image }));
    const values = (range, numeric = false) => {
      const [start, end = start] = range.split(":"),
        a = ws.getCell(start),
        b = ws.getCell(end),
        out = [];
      for (let r = a.row; r <= b.row; r++)
        for (let c = a.col; c <= b.col; c++) {
          const cell = ws.getCell(r, c);
          out.push(
            numeric
              ? cell.formula
                ? cell.result
                : cell.value
              : excelValue(cell),
          );
        }
      return out;
    };
    for (const c of s.charts || [])
      def.content.push(
        pdfBlock({
          type: "chart",
          chart: {
            kind: c.kind,
            title: c.title,
            labels: values(c.categories, c.kind === "scatter").map(String),
            series: c.series.map((v) => ({
              name: v.name,
              values: values(v.values, true).map(Number),
            })),
          },
        }),
      );
    for (const p of s.pivots || []) {
      const fields = [...(p.rows || []), ...(p.columns || [])],
        indices = fields.map(
          (f) => s.columns.findIndex((c) => c.title === f) + 1,
        ),
        groups = new Map();
      const [a, z] = p.source.split(":").map((x) => ws.getCell(x));
      for (let r = a.row + 1; r <= z.row; r++) {
        const keys = indices.map((c) => excelValue(ws.getCell(r, c))),
          id = JSON.stringify(keys);
        if (!groups.has(id))
          groups.set(id, { keys, values: p.values.map(() => []) });
        p.values.forEach((v, i) => {
          const cell = ws.getCell(
            r,
            s.columns.findIndex((c) => c.title === v.field) + 1,
          );
          groups
            .get(id)
            .values[i].push(cell.formula ? cell.result : cell.value);
        });
      }
      const lines = [
        [...fields, ...p.values.map((v) => v.function + " " + v.field)],
        ...[...groups.values()].map((g) => [
          ...g.keys,
          ...g.values.map((v, i) => {
            const nums = v.filter((n) => typeof n === "number");
            switch (p.values[i].function) {
              case "Count":
                return String(v.filter((n) => n !== null && n !== "").length);
              case "Average":
                return String(
                  nums.reduce((a, n) => a + n, 0) / (nums.length || 1),
                );
              case "Min":
                return String(nums.length ? Math.min(...nums) : 0);
              case "Max":
                return String(nums.length ? Math.max(...nums) : 0);
              default:
                return String(nums.reduce((a, n) => a + n, 0));
            }
          }),
        ]),
      ];
      def.content.push({
        text: p.name + " — 집계",
        bold: true,
        margin: [0, 12, 0, 8],
      });
      def.content.push(pdfTable({ rows: lines }));
    }
    for (const shape of s.shapes || [])
      def.content.push(
        pdfBlock({ type: "shape", shape: shape.kind, text: shape.text }),
      );
  }
  return def;
}
