// Ordered content shared by Office, Hancom and the companion PDF.
import { validateImages } from "./document-images.mjs";
export function obj(v, allowed) {
  if (
    !v ||
    typeof v !== "object" ||
    Array.isArray(v) ||
    Object.keys(v).some((k) => !allowed.includes(k))
  )
    throw Error("Unsupported document field");
}
export function str(v, n = 4000) {
  if (
    typeof v !== "string" ||
    v.length > n ||
    /[\x00-\x08\x0b\x0c\x0e-\x1f]/.test(v)
  )
    throw Error("Invalid document text");
  return v;
}
export function num(v, min, max) {
  if (!Number.isFinite(v) || v < min || v > max)
    throw Error("Invalid document dimension");
  return v;
}
export function color(v) {
  if (!/^[0-9a-f]{6}$/i.test(v)) throw Error("Color must be six hex digits");
  return v;
}
export function url(v) {
  str(v, 2048);
  if (!/^(https?:\/\/|mailto:)/i.test(v))
    throw Error("Link must be HTTP(S) or mailto");
  return v;
}
export function style(v = {}) {
  obj(v, [
    "font",
    "size",
    "bold",
    "italic",
    "underline",
    "strike",
    "color",
    "background",
    "align",
    "indent",
    "spacing",
    "line_spacing",
    "border",
    "valign",
    "margin",
    "num_fmt",
  ]);
  for (const k of ["color", "background", "border"])
    if (v[k] !== undefined) color(v[k]);
  for (const k of ["bold", "italic", "underline", "strike"])
    if (v[k] !== undefined && typeof v[k] !== "boolean")
      throw Error("Invalid style flag");
  if (v.font !== undefined) str(v.font, 80);
  if (v.num_fmt !== undefined) str(v.num_fmt, 100);
  if (v.size !== undefined) num(v.size, 6, 72);
  for (const k of ["indent", "spacing", "margin"])
    if (v[k] !== undefined) num(v[k], 0, 144);
  if (v.line_spacing !== undefined) num(v.line_spacing, 1, 3);
  if (v.align && !["left", "center", "right", "justify"].includes(v.align))
    throw Error("Invalid alignment");
  if (v.valign && !["top", "center", "bottom"].includes(v.valign))
    throw Error("Invalid vertical alignment");
  return v;
}
export function runs(v) {
  if (typeof v === "string") return [{ text: str(v) }];
  if (!Array.isArray(v) || v.length > 200) throw Error("Invalid text runs");
  for (const r of v) {
    obj(r, ["text", "style", "link"]);
    str(r.text);
    style(r.style);
    if (r.link) url(r.link);
  }
  return v;
}
export function plain(v) {
  return typeof v === "string" ? v : (v || []).map((r) => r.text).join("");
}
export function tableRows(b) {
  const rows = b.rows;
  if (!Array.isArray(rows) || rows.length < 1 || rows.length > 100)
    throw Error("Table needs 1–100 rows");
  if (!Array.isArray(rows[0])) throw Error("Invalid table row");
  const constraints = [];
  const width = rows[0].reduce((n, c) => n + (c?.col_span || 1), 0);
  if (width < 1 || width > 8) throw Error("Table needs 1–8 columns");
  const occupied = Array.from({ length: rows.length }, () =>
    Array(width).fill(false),
  );
  for (let r = 0; r < rows.length; r++) {
    if (!Array.isArray(rows[r])) throw Error("Invalid table row");
    let c = 0;
    for (const cell of rows[r]) {
      while (occupied[r][c]) c++;
      const x = typeof cell === "string" ? { text: cell } : cell;
      obj(x, ["text", "style", "col_span", "row_span", "blocks", "width"]);
      runs(x.text || "");
      style(x.style);
      const cs = x.col_span || 1,
        rs = x.row_span || 1;
      if (
        !Number.isInteger(cs) ||
        !Number.isInteger(rs) ||
        cs < 1 ||
        rs < 1 ||
        c + cs > width ||
        r + rs > rows.length
      )
        throw Error("Invalid merged cell");
      if (x.width !== undefined) {
        num(x.width, 0.001, 1000);
        constraints.push([...Array.from({length:width}, (_,i) => i>=c && i<c+cs ? 1 : 0), x.width]);
      }
      for (let y = r; y < r + rs; y++)
        for (let z = c; z < c + cs; z++) {
          if (occupied[y][z]) throw Error("Overlapping merged cells");
          occupied[y][z] = true;
        }
      c += cs;
    }
    if (occupied[r].some((x) => !x))
      throw Error("Table rows must have equal columns");
  }
  if (b.widths) {
    if (b.widths.length !== width) throw Error("Table column width mismatch");
    b.widths.forEach((x) => num(x, 0.001, 1000));
  }
  if (constraints.length) {
    if (b.widths) {
      for (const row of constraints)
        if (Math.abs(row.slice(0,width).reduce((v,a,i)=>v+a*b.widths[i],0)-row[width])>1e-7)
          throw Error("Conflicting cell.width and table.widths; use the same relative units");
    } else b.widths = resolveCellWidths(constraints, width);
  }
  return width;
}
// Find the widths closest to equal columns while satisfying every cell constraint.
// Row spans map to their actual grid column; col spans constrain the sum of columns.
function resolveCellWidths(rows, n) {
  const a=rows.map(row=>[...row]); let rank=0;
  for(let c=0;c<n;c++) {
    const pivot=a.findIndex((row,i)=>i>=rank && Math.abs(row[c])>1e-9);
    if(pivot<0) continue;
    [a[rank],a[pivot]]=[a[pivot],a[rank]];
    const divisor=a[rank][c]; a[rank]=a[rank].map(x=>x/divisor);
    for(let r=0;r<a.length;r++) if(r!==rank) {
      const factor=a[r][c]; a[r]=a[r].map((x,j)=>x-factor*a[rank][j]);
    }
    rank++;
  }
  if(a.slice(rank).some(row=>Math.abs(row[n])>1e-7))
    throw Error("Conflicting cell.width values in the same table grid");
  const basis=a.slice(0,rank);
  const gram=basis.map(row=>[...basis.map(other=>row.slice(0,n).reduce((v,x,j)=>v+x*other[j],0)),row[n]-row.slice(0,n).reduce((v,x)=>v+x,0)]);
  for(let c=0;c<rank;c++) {
    let pivot=c;
    for(let r=c+1;r<rank;r++) if(Math.abs(gram[r][c])>Math.abs(gram[pivot][c])) pivot=r;
    [gram[c],gram[pivot]]=[gram[pivot],gram[c]];
    const divisor=gram[c][c]; gram[c]=gram[c].map(x=>x/divisor);
    for(let r=0;r<rank;r++) if(r!==c) {
      const factor=gram[r][c]; gram[r]=gram[r].map((x,j)=>x-factor*gram[c][j]);
    }
  }
  const widths=Array.from({length:n},(_,c)=>1+basis.reduce((v,row,r)=>v+row[c]*gram[r][rank],0));
  if(widths.some(v=>!Number.isFinite(v)||v<0.001||v>1000))
    throw Error("Cell widths leave an invalid column width; specify all column widths explicitly");
  return widths;
}
export function chart(c) {
  obj(c, ["kind", "title", "labels", "series", "legend"]);
  if (!["bar", "line", "pie", "doughnut", "area", "scatter"].includes(c.kind))
    throw Error("Unsupported chart kind");
  if (c.title) str(c.title, 120);
  if (!Array.isArray(c.labels) || !c.labels.length || c.labels.length > 100)
    throw Error("Invalid chart labels");
  c.labels.forEach((x) => str(x, 80));
  if (!Array.isArray(c.series) || !c.series.length || c.series.length > 8)
    throw Error("Invalid chart series");
  for (const s of c.series) {
    obj(s, ["name", "values", "color"]);
    str(s.name, 80);
    if (s.color) color(s.color);
    if (
      !Array.isArray(s.values) ||
      s.values.length !== c.labels.length ||
      s.values.some((n) => !Number.isFinite(n))
    )
      throw Error("Chart data mismatch");
  }
  if (
    ["pie", "doughnut"].includes(c.kind) &&
    (c.series.length !== 1 ||
      c.series[0].values.some((v) => v < 0) ||
      c.series[0].values.every((v) => v === 0))
  )
    throw Error(
      "Pie charts require one nonnegative series with a positive total",
    );
  if (
    c.kind === "scatter" &&
    c.labels.some((v) => !v.trim() || !Number.isFinite(Number(v)))
  )
    throw Error("Scatter chart labels must be numeric X coordinates");
  return c;
}
export function blocks(container) {
  if (container.blocks) return container.blocks;
  if (container.table && !Array.isArray(container.table))
    obj(container.table, ["rows", "widths", "style"]);
  return [
    ...(container.heading
      ? [{ type: "heading", text: container.heading }]
      : []),
    ...(container.paragraphs || []).map((text) => ({
      type: "paragraph",
      text,
    })),
    ...(container.bullets?.length
      ? [{ type: "list", items: container.bullets }]
      : []),
    ...(container.table ? [Array.isArray(container.table) ? { type: "table", rows: container.table } : { ...container.table, type: "table" }] : []),
    ...(container.images || []).map((image) => ({ type: "image", image })),
  ];
}
export function validateBlocks(items, format, depth = 0) {
  if (!Array.isArray(items) || items.length > 500 || depth > 2)
    throw Error("Invalid content blocks");
  for (const b of items) {
    obj(b, [
      "type",
      "text",
      "style",
      "level",
      "items",
      "ordered",
      "rows",
      "widths",
      "image",
      "chart",
      "columns",
      "name",
      "note",
      "numerator",
      "denominator",
      "checked",
      "shape",
      "x",
      "y",
      "w",
      "h",
      "fill",
      "line",
      "link",
      "media",
      "wrap",
      "fit",
    ]);
    style(b.style);
    for (const flag of ["ordered", "checked"])
      if (b[flag] !== undefined && typeof b[flag] !== "boolean")
        throw Error("Invalid block flag");
    for (const k of ["x", "y", "w", "h"])
      if (b[k] !== undefined) num(b[k], 0, 100);
    if (b.link) url(b.link);
    if (b.fill) color(b.fill);
    if (b.line) color(b.line);
    switch (b.type) {
      case "paragraph":
      case "heading":
      case "quote":
        runs(b.text);
        if (b.level !== undefined) {
          num(b.level, 1, 6);
          if (!Number.isInteger(b.level))
            throw Error("Heading level must be an integer");
        }
        break;
      case "list":
        if (!Array.isArray(b.items) || b.items.length > 100)
          throw Error("Invalid list");
        b.items.forEach(runs);
        break;
      case "table":
        tableRows(b);
        for (const row of b.rows)
          for (const c of row)
            if (c?.blocks) {
              if (!["docx", "pdf"].includes(format))
                throw Error("Nested cell blocks are DOCX/PDF only");
              validateBlocks(c.blocks, format, depth + 1);
            }
        break;
      case "image":
        if (b.fit && !["contain", "cover"].includes(b.fit))
          throw Error("Invalid image fit");
        if (b.fit && !["pptx", "pdf"].includes(format))
          throw Error("Image crop is PPTX/PDF only");
        validateImages([{ images: [b.image] }]);
        if (b.wrap && !["inline", "square", "topBottom"].includes(b.wrap))
          throw Error("Invalid image wrap");
        break;
      case "chart":
        if (!["pptx", "pdf"].includes(format))
          throw Error("Charts belong to PPTX/PDF blocks or XLSX charts");
        chart(b.chart);
        break;
      case "columns":
        if (
          !["pptx", "pdf", "docx"].includes(format) ||
          !Array.isArray(b.columns) ||
          b.columns.length < 2 ||
          b.columns.length > 3
        )
          throw Error("Invalid column layout");
        b.columns.forEach((c) => validateBlocks(c, format, depth + 1));
        break;
      case "page_break":
        break;
      case "toc":
        if (!["docx", "pdf"].includes(format))
          throw Error("TOC is DOCX/PDF only");
        break;
      case "bookmark":
        str(b.name, 40);
        if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(b.name))
          throw Error("Invalid bookmark");
        runs(b.text || "");
        break;
      case "footnote":
      case "endnote":
        if (b.type === "endnote" && format === "docx")
          throw Error("DOCX endnotes are not supported");
        runs(b.text);
        str(b.note);
        break;
      case "comment":
        if (!["docx", "pdf"].includes(format))
          throw Error("Comments are DOCX/PDF only");
        runs(b.text);
        str(b.note);
        break;
      case "equation":
        str(b.text || "");
        if (b.numerator !== undefined) str(b.numerator, 200);
        if (b.denominator !== undefined) str(b.denominator, 200);
        break;
      case "checkbox":
        if (!["docx", "pdf"].includes(format))
          throw Error("Checkbox is DOCX/PDF only");
        str(b.text || "");
        break;
      case "shape":
        if (
          !["pptx", "hwp", "hwpx", "pdf"].includes(format) ||
          !["rect", "ellipse", "line", "arrow", "textbox"].includes(b.shape)
        )
          throw Error("Invalid shape");
        runs(b.text || "");
        break;
      case "media":
        if (format !== "pptx") throw Error("Media is PPTX only");
        obj(b.media, ["data", "mime", "name"]);
        if (
          !["audio/mpeg", "audio/wav", "video/mp4"].includes(b.media.mime) ||
          typeof b.media.data !== "string" ||
          b.media.data.length > 12 * 1024 * 1024
        )
          throw Error("Invalid media");
        break;
      default:
        throw Error("Unsupported block type: " + b.type);
    }
  }
}
export function page(p = {}) {
  obj(p, [
    "size",
    "orientation",
    "margin",
    "columns",
    "header",
    "footer",
    "first_header",
    "even_header",
    "first_footer",
    "even_footer",
    "page_numbers",
    "break_before",
  ]);
  for (const k of ["page_numbers", "break_before"])
    if (p[k] !== undefined && typeof p[k] !== "boolean")
      throw Error("Invalid page flag");
  if (p.size && !["A4", "A3", "LETTER"].includes(p.size))
    throw Error("Invalid paper size");
  if (p.orientation && !["portrait", "landscape"].includes(p.orientation))
    throw Error("Invalid page orientation");
  if (p.margin !== undefined) num(p.margin, 10, 100);
  if (p.columns !== undefined && ![1, 2, 3].includes(p.columns))
    throw Error("Invalid columns");
  for (const k of [
    "header",
    "footer",
    "first_header",
    "even_header",
    "first_footer",
    "even_footer",
  ])
    if (p[k] !== undefined) str(p[k], 240);
  return p;
}
export function isExtended(input) {
  return Boolean(
    input.page ||
      input.theme ||
      input.slides?.some(
        (s) => s.blocks || s.table || s.images || s.notes || s.layout,
      ) ||
      input.sections?.some((s) => s.blocks || s.page),
  );
}
export function allImages(input) {
  const found = [];
  function walk(b) {
    for (const x of b || []) {
      if (x.type === "image") found.push(x.image);
      for (const c of x.columns || []) walk(c);
      for (const row of x.rows || []) for (const c of row) walk(c?.blocks);
    }
  }
  for (const s of [...(input.sections || []), ...(input.slides || [])]) {
    walk(s.blocks);
    found.push(...(s.images || []));
  }
  for (const s of input.sheets || [])
    for (const x of s.images || []) found.push(x.image);
  return found;
}

export function validateMedia(input) {
  let count = 0,
    total = 0;
  function visit(items) {
    for (const b of items || []) {
      if (b.type === "media") {
        if (++count > 3) throw Error("At most three media attachments");
        const bytes = Buffer.from(b.media.data, "base64");
        total += bytes.length;
        if (total > 8 * 1024 * 1024) throw Error("Media exceeds 8 MiB");
        const mime = b.media.mime;
        const valid =
          mime === "audio/wav"
            ? bytes.toString("ascii", 0, 4) === "RIFF" &&
              bytes.toString("ascii", 8, 12) === "WAVE"
            : mime === "video/mp4"
              ? bytes.toString("ascii", 4, 8) === "ftyp"
              : bytes.toString("ascii", 0, 3) === "ID3" ||
                (bytes[0] === 255 && (bytes[1] & 224) === 224);
        if (!valid) throw Error("Invalid media signature");
      }
      for (const c of b.columns || []) visit(c);
    }
  }
  for (const slide of input.slides || []) visit(slide.blocks);
}
