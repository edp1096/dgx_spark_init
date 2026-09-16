import { obj, str, num, style, url, validateBlocks } from "./blocks.mjs";
export const sheetExtras = [
  "cells",
  "merges",
  "conditional_formats",
  "images",
  "charts",
  "pivots",
  "shapes",
  "row_options",
  "column_options",
  "freeze",
  "hidden",
  "protection",
  "table_name",
  "print",
];
function address(v) {
  if (typeof v !== "string" || !/^\$?[A-Z]{1,3}\$?[1-9]\d{0,5}$/.test(v))
    throw Error("Invalid cell address");
  return v;
}
function range(v) {
  str(v, 80);
  v.split(":").forEach(address);
  if (v.split(":").length > 2) throw Error("Invalid range");
  return v;
}
export function validateSheetFeatures(s) {
  for (const key of [
    "cells",
    "merges",
    "conditional_formats",
    "images",
    "charts",
    "pivots",
    "shapes",
    "row_options",
    "column_options",
  ])
    if (
      s[key] !== undefined &&
      (!Array.isArray(s[key]) ||
        s[key].length >
          {
            cells: 2000,
            merges: 100,
            conditional_formats: 100,
            images: 6,
            charts: 8,
            pivots: 4,
            shapes: 20,
            row_options: 1001,
            column_options: 32,
          }[key])
    )
      throw Error("Too many or invalid " + key);
  if (s.hidden !== undefined && typeof s.hidden !== "boolean")
    throw Error("Invalid sheet visibility");
  const coord = (a) => {
    address(a);
    const m = /^\$?([A-Z]+)\$?(\d+)$/.exec(a);
    return {
      c: [...m[1]].reduce((n, c) => n * 26 + c.charCodeAt(0) - 64, 0),
      r: Number(m[2]),
    };
  };
  const dataRange = (a) => {
    range(a);
    const [left, right = left] = a.split(":"),
      l = coord(left),
      r = coord(right);
    if (
      l.c > r.c ||
      l.r > r.r ||
      r.c > s.columns?.length ||
      r.r > (s.rows?.length || 0) + 1
    )
      throw Error("Feature range is outside supplied table");
    return { l, r, length: (r.c - l.c + 1) * (r.r - l.r + 1) };
  };
  const anchor = (a) => {
    const q = coord(a);
    if (q.c > 1000 || q.r > 10000)
      throw Error("Object anchor too far from supplied data");
  };
  for (const c of s.cells || []) {
    obj(c, ["cell", "style", "link", "note", "validation"]);
    dataRange(c.cell);
    style(c.style);
    if (c.link) url(c.link);
    if (c.note) str(c.note, 2000);
    if (c.validation) {
      const v = c.validation;
      obj(v, ["type", "values", "min", "max", "allow_blank"]);
      if (!["list", "whole", "decimal", "textLength"].includes(v.type))
        throw Error("Invalid validation");
      if (v.type === "list") {
        if (
          !Array.isArray(v.values) ||
          v.values.length > 100 ||
          v.values.join(",").length > 250
        )
          throw Error("Invalid dropdown");
        v.values.forEach((x) => {
          str(x, 100);
          if (/[",\n]/.test(x))
            throw Error("Dropdown items cannot contain commas or quotes");
        });
      } else {
        num(v.min, -1e12, 1e12);
        num(v.max, -1e12, 1e12);
        if (v.min > v.max) throw Error("Invalid validation bounds");
      }
    }
  }
  for (const x of s.merges || []) dataRange(x);
  for (const c of s.conditional_formats || []) {
    obj(c, ["range", "operator", "value", "style"]);
    dataRange(c.range);
    if (
      ![
        "greaterThan",
        "lessThan",
        "equal",
        "greaterThanOrEqual",
        "lessThanOrEqual",
      ].includes(c.operator)
    )
      throw Error("Invalid condition");
    num(c.value, -1e12, 1e12);
    style(c.style);
  }
  for (const x of s.images || []) {
    obj(x, ["image", "cell", "width", "height"]);
    anchor(x.cell);
    if (x.width) num(x.width, 10, 1200);
    if (x.height) num(x.height, 10, 1200);
    validateBlocks([{ type: "image", image: x.image }], "xlsx");
  }
  for (const c of s.charts || []) {
    obj(c, ["kind", "title", "cell", "categories", "series"]);
    if (!["bar", "line", "pie", "doughnut", "area", "scatter"].includes(c.kind))
      throw Error("Invalid chart");
    str(c.title || "", 120);
    anchor(c.cell);
    const cats = dataRange(c.categories);
    if (cats.l.c !== cats.r.c && cats.l.r !== cats.r.r)
      throw Error("Chart categories must be a single row or column");
    if (!Array.isArray(c.series) || !c.series.length || c.series.length > 8)
      throw Error("Invalid series");
    for (const series of c.series) {
      obj(series, ["name", "values"]);
      str(series.name, 80);
      if (dataRange(series.values).length !== cats.length)
        throw Error("Chart series length mismatch");
    }
  }
  for (const p of s.pivots || []) {
    obj(p, ["name", "source", "destination", "rows", "columns", "values"]);
    str(p.name, 40);
    const source = dataRange(p.source);
    range(p.destination);
    p.destination.split(":").forEach(anchor);
    const dest = coord(p.destination.split(":")[0]);
    if (dest.r <= s.rows.length + 1 && dest.c <= s.columns.length)
      throw Error("Pivot destination overlaps supplied data");
    if (source.l.r !== 1) throw Error("Pivot source must include header row 1");
    for (const k of ["rows", "columns"])
      for (const name of p[k] || [])
        if (!s.columns.some((c) => c.title === name))
          throw Error("Unknown pivot field");
    if (!Array.isArray(p.values) || !p.values.length)
      throw Error("Pivot needs values");
    for (const v of p.values) {
      obj(v, ["field", "function"]);
      if (
        !s.columns.some((c) => c.title === v.field) ||
        !["Sum", "Count", "Average", "Min", "Max"].includes(v.function)
      )
        throw Error("Invalid pivot value");
    }
  }
  for (const x of s.shapes || []) {
    obj(x, ["cell", "text", "kind"]);
    anchor(x.cell);
    str(x.text, 1000);
    if (!["rect", "ellipse", "line", "arrow"].includes(x.kind))
      throw Error("Invalid shape");
  }
  for (const key of ["row_options", "column_options"])
    for (const r of s[key] || []) {
      obj(r, ["index", "hidden", "level", "size"]);
      if (!Number.isInteger(r.index)) throw Error("Invalid row/column index");
      num(r.index, 1, key === "row_options" ? 1001 : 32);
      if (r.level !== undefined) num(r.level, 0, 7);
      if (r.size !== undefined) num(r.size, 1, 409);
    }
  if (s.freeze) {
    obj(s.freeze, ["rows", "columns"]);
    num(s.freeze.rows || 0, 0, 1000);
    num(s.freeze.columns || 0, 0, 32);
  }
  if (s.protection) {
    obj(s.protection, ["password"]);
    str(s.protection.password || "", 64);
  }
  if (s.table_name && !/^[A-Za-z][A-Za-z0-9_]{0,39}$/.test(s.table_name))
    throw Error("Invalid Excel table name");
  if (s.print) {
    obj(s.print, [
      "area",
      "repeat_rows",
      "repeat_columns",
      "header",
      "footer",
      "margin",
      "scale",
      "paper",
    ]);
    if (s.print.area) range(s.print.area);
    if (s.print.repeat_rows && !/^\d+:\d+$/.test(s.print.repeat_rows))
      throw Error("Invalid print rows");
    if (
      s.print.repeat_columns &&
      !/^[A-Z]+:[A-Z]+$/.test(s.print.repeat_columns)
    )
      throw Error("Invalid print columns");
    for (const k of ["header", "footer"]) if (s.print[k]) str(s.print[k], 200);
    if (s.print.scale) num(s.print.scale, 10, 400);
    if (s.print.margin !== undefined) num(s.print.margin, 0, 2);
    if (s.print.paper && !["A4", "A3", "LETTER"].includes(s.print.paper))
      throw Error("Invalid print paper");
  }
}
function excelStyle(s = {}) {
  return {
    font: {
      name: s.font || "Noto Sans CJK KR",
      size: s.size || 11,
      bold: s.bold,
      italic: s.italic,
      underline: s.underline,
      strike: s.strike,
      ...(s.color ? { color: { argb: "FF" + s.color } } : {}),
    },
    ...(s.background
      ? {
          fill: {
            type: "pattern",
            pattern: "solid",
            fgColor: { argb: "FF" + s.background },
          },
        }
      : {}),
    ...(s.border
      ? {
          border: Object.fromEntries(
            ["top", "bottom", "left", "right"].map((k) => [
              k,
              { style: "thin", color: { argb: "FF" + s.border } },
            ]),
          ),
        }
      : {}),
    alignment: {
      horizontal: s.align === "justify" ? "justify" : s.align || "left",
      vertical: s.valign === "center" ? "middle" : s.valign || "top",
      wrapText: true,
    },
    ...(s.num_fmt ? { numFmt: s.num_fmt } : {}),
  };
}
export async function applySheetFeatures(wb, ws, s) {
  for (const c of s.cells || []) {
    const cell = ws.getCell(c.cell);
    if (cell.row > s.rows.length + 1 || cell.col > s.columns.length)
      throw Error("Cell option is outside supplied table");
    if (c.style) cell.style = { ...cell.style, ...excelStyle(c.style) };
    if (c.link) {
      if (typeof cell.value !== "string")
        throw Error("Hyperlinks require a text cell");
      cell.value = { text: cell.value, hyperlink: c.link };
    }
    if (c.note) cell.note = c.note;
    if (c.validation) {
      const v = c.validation;
      cell.dataValidation = {
        type: v.type,
        allowBlank: v.allow_blank !== false,
        showErrorMessage: true,
        errorTitle: "입력 오류",
        error: "허용된 값을 입력하세요.",
        ...(v.type === "list"
          ? { formulae: ['"' + v.values.join(",") + '"'] }
          : { operator: "between", formulae: [v.min, v.max] }),
      };
    }
  }
  for (const r of s.merges || []) {
    const [a, z = a] = r.split(":"),
      first = ws.getCell(a),
      last = ws.getCell(z);
    if (
      last.row > s.rows.length + 1 ||
      last.col > s.columns.length ||
      last.row < first.row ||
      last.col < first.col
    )
      throw Error("Merged range exceeds supplied table");
    for (let y = first.row; y <= last.row; y++)
      for (let x = first.col; x <= last.col; x++)
        if (
          (y !== first.row || x !== first.col) &&
          ![null, ""].includes(ws.getCell(y, x).value)
        )
          throw Error(
            "Merged cells would discard content; leave covered cells empty",
          );
    ws.mergeCells(r);
  }
  for (const c of s.conditional_formats || [])
    ws.addConditionalFormatting({
      ref: c.range,
      rules: [
        {
          type: "cellIs",
          operator: c.operator,
          formulae: [String(c.value)],
          style: excelStyle(c.style),
        },
      ],
    });
  for (const x of s.images || []) {
    const id = wb.addImage({
        base64: "data:image/png;base64," + x.image.data,
        extension: "png",
      }),
      cell = ws.getCell(x.cell);
    ws.addImage(id, {
      tl: { col: cell.col - 1, row: cell.row - 1 },
      ext: {
        width: x.width || (x.image.width_cm * 96) / 2.54,
        height:
          x.height ||
          ((x.width || (x.image.width_cm * 96) / 2.54) * x.image.height_px) /
            x.image.width_px,
      },
      editAs: "oneCell",
    });
  }
  for (const r of s.row_options || []) {
    const row = ws.getRow(r.index);
    if (r.hidden !== undefined) row.hidden = r.hidden;
    if (r.level !== undefined) row.outlineLevel = r.level;
    if (r.size) row.height = r.size;
  }
  for (const c of s.column_options || []) {
    const col = ws.getColumn(c.index);
    if (c.hidden !== undefined) col.hidden = c.hidden;
    if (c.level !== undefined) col.outlineLevel = c.level;
    if (c.size) col.width = c.size;
  }
  if (s.freeze)
    ws.views = [
      {
        state: "frozen",
        xSplit: s.freeze.columns || 0,
        ySplit: s.freeze.rows || 0,
      },
    ];
  if (s.hidden) ws.state = "hidden";
  if (s.protection)
    await ws.protect(s.protection.password || "", {
      selectLockedCells: true,
      selectUnlockedCells: true,
      spinCount: 1000,
    });
  if (s.table_name) {
    if (s.merges?.length)
      throw Error("Excel table and merged cells cannot overlap");
    ws.addTable({
      name: s.table_name,
      ref: "A1",
      headerRow: true,
      totalsRow: false,
      style: { theme: "TableStyleMedium2", showRowStripes: true },
      columns: s.columns.map((c) => ({
        name: c.title,
        filterButton: s.filter !== false,
      })),
      rows: s.rows.map((r, y) =>
        r.map((_, x) => ws.getCell(y + 2, x + 1).value),
      ),
    });
  }
  if (s.print) {
    const p = s.print;
    if (p.area) ws.pageSetup.printArea = p.area;
    if (p.repeat_rows) ws.pageSetup.printTitlesRow = p.repeat_rows;
    if (p.repeat_columns) ws.pageSetup.printTitlesColumn = p.repeat_columns;
    if (p.scale) {
      ws.pageSetup.fitToPage = false;
      ws.pageSetup.scale = p.scale;
    }
    if (p.paper) ws.pageSetup.paperSize = { A4: 9, A3: 8, LETTER: 1 }[p.paper];
    if (p.margin !== undefined)
      ws.pageSetup.margins = {
        left: p.margin,
        right: p.margin,
        top: p.margin,
        bottom: p.margin,
        header: 0.2,
        footer: 0.2,
      };
    ws.headerFooter = { oddHeader: p.header || "", oddFooter: p.footer || "" };
  }
}

export function validateComputedCharts(wb, input) {
  for (const sheet of input.sheets) {
    const ws = wb.getWorksheet(sheet.name);
    function values(range) {
      const [start, end = start] = range.split(":"),
        a = ws.getCell(start),
        z = ws.getCell(end),
        out = [];
      for (let r = a.row; r <= z.row; r++)
        for (let c = a.col; c <= z.col; c++) {
          const cell = ws.getCell(r, c);
          out.push(cell.formula ? cell.result : cell.value);
        }
      return out;
    }
    for (const chart of sheet.charts || []) {
      const ys = chart.series.flatMap((s) => values(s.values));
      if (ys.some((n) => typeof n !== "number" || !Number.isFinite(n)))
        throw Error("Chart series must contain numeric cells");
      if (
        chart.kind === "scatter" &&
        values(chart.categories).some(
          (n) => typeof n !== "number" || !Number.isFinite(n),
        )
      )
        throw Error("Scatter categories must contain numeric X coordinates");
      if (
        ["pie", "doughnut"].includes(chart.kind) &&
        (chart.series.length !== 1 ||
          ys.some((n) => n < 0) ||
          ys.every((n) => n === 0))
      )
        throw Error("Pie charts require positive, nonnegative data");
    }
  }
}
