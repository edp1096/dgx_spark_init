import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import JSZip from "jszip";
import { render } from "./renderer.mjs";
const base = [
  { type: "heading", text: "기능 검증", level: 1 },
  {
    type: "paragraph",
    text: [
      { text: "굵은 한글", style: { bold: true, color: "AA0000" } },
      { text: " 링크", link: "https://example.com" },
    ],
  },
  {
    type: "table",
    rows: [
      [{ text: "병합 헤더", col_span: 2 }],
      ["항목", "값"],
      ["검증", "성공"],
    ],
  },
  { type: "list", ordered: true, items: ["첫째 항목", "둘째 항목"] },
];
for (const format of ["docx", "pptx"])
  test("rich " + format, async () => {
    const dir = await fs.mkdtemp(
      path.join(process.env.ARTIFACT_ROOT || os.tmpdir(), "rich-"),
    );
    const input =
      format === "pptx"
        ? {
            format,
            title: "발표자료",
            slides: [
              { title: "표와 본문", blocks: base, notes: "발표자 메모" },
              {
                title: "차트",
                blocks: [
                  {
                    type: "chart",
                    chart: {
                      kind: "bar",
                      labels: ["가", "나"],
                      series: [{ name: "매출", values: [10, 20] }],
                    },
                  },
                ],
              },
            ],
          }
        : {
            format,
            title: "문서 검증",
            page: { header: "머리말", footer: "꼬리말", page_numbers: true },
            sections: [
              {
                blocks: [
                  { type: "toc" },
                  ...base,
                  { type: "footnote", text: "각주본문", note: "각주검증" },
                  { type: "comment", text: "검토본문", note: "댓글검증" },
                  { type: "equation", numerator: "1", denominator: "2" },
                  { type: "checkbox", text: "체크 항목", checked: true },
                ],
              },
            ],
          };
    const result = await render(input, dir);
    assert.equal(result.warning, undefined);
    assert.equal(result.files.length, 2);
    const zip = await JSZip.loadAsync(
      Buffer.from(result.files[0].data, "base64"),
    );
    const files = Object.keys(zip.files);
    assert.ok(
      files.some((n) =>
        n.includes(
          format === "docx" ? "word/document.xml" : "ppt/slides/slide1.xml",
        ),
      ),
    );
    const xml = await zip
      .file(format === "docx" ? "word/document.xml" : "ppt/slides/slide1.xml")
      .async("string");
    assert.match(xml, format === "docx" ? /<w:tbl>/ : /<a:tbl>/);
    assert.match(xml, /병합 헤더/);
    if (format === "pptx") {
      assert.ok(files.some((n) => n.startsWith("ppt/charts/chart")));
      assert.ok(files.some((n) => n.startsWith("ppt/notesSlides/notesSlide")));
    }
    const text = execFileSync(
      "pdftotext",
      [path.join(dir, "document.pdf"), "-"],
      { encoding: "utf8" },
    );
    assert.match(text, /검증/);
    console.log("ARTIFACT", format, dir);
  });
for (const format of ["hwp", "hwpx"])
  test("rich " + format, async () => {
    const dir = await fs.mkdtemp(
      path.join(process.env.ARTIFACT_ROOT || os.tmpdir(), "rich-hancom-"),
    );
    const result = await render(
      {
        format,
        title: "한글 확장 검증",
        page: { header: "머리말 검증", footer: "꼬리말", margin: 45 },
        sections: [
          {
            blocks: [
              ...base.map((b) =>
                b.type === "paragraph"
                  ? {
                      ...b,
                      text: [
                        {
                          text: "굵은 한글",
                          style: { bold: true, color: "AA0000" },
                        },
                      ],
                    }
                  : b,
              ),
              { type: "footnote", text: "각주 본문", note: "각주 내용 확인" },
              { type: "endnote", text: "미주 본문", note: "미주 내용 확인" },
              { type: "equation", numerator: "1", denominator: "2" },
              { type: "bookmark", name: "target", text: "책갈피 본문" },
              { type: "shape", shape: "textbox", text: "글상자 검증" },
            ],
          },
        ],
      },
      dir,
    );
    assert.equal(result.warning, undefined);
    const { HwpDocument } = await import("@rhwp/core");
    const doc = new HwpDocument(Buffer.from(result.files[0].data, "base64"));
    try {
      assert.match(JSON.parse(doc.getTextFileUnicode()), /병합 헤더/);
      assert.match(JSON.parse(doc.getTextFileUnicode()), /각주 본문/);
      console.log("HANCOM", format, doc.pageCount());
    } finally {
      doc.free();
    }
    console.log("ARTIFACT", format, dir);
  });
test("rich spreadsheet", async () => {
  const dir = await fs.mkdtemp(
    path.join(process.env.ARTIFACT_ROOT || os.tmpdir(), "rich-xlsx-"),
  );
  const result = await render(
    {
      format: "xlsx",
      title: "확장 엑셀",
      sheets: [
        {
          name: "매출",
          columns: [{ title: "분류" }, { title: "금액" }, { title: "결과" }],
          rows: [
            ["가", 10, { formula: 'SUMIFS(B2:B4,A2:A4,"가")' }],
            ["나", 20, { formula: 'XLOOKUP("나",A2:A4,B2:B4)' }],
            ["가", 30, { formula: 'COUNTIFS(A2:A4,"가")' }],
          ],
          cells: [
            {
              cell: "B2",
              style: { color: "AA0000", bold: true },
              note: "금액 메모",
              validation: { type: "whole", min: 0, max: 100 },
            },
            { cell: "A2", validation: { type: "list", values: ["가", "나"] } },
          ],
          conditional_formats: [
            {
              range: "B2:B4",
              operator: "greaterThan",
              value: 15,
              style: { background: "FFAAAA" },
            },
          ],
          freeze: { rows: 1, columns: 1 },
          row_options: [{ index: 4, level: 1 }],
          charts: [
            {
              kind: "bar",
              title: "매출 차트",
              cell: "E2",
              categories: "A2:A4",
              series: [{ name: "금액", values: "B2:B4" }],
            },
          ],
          pivots: [
            {
              name: "SalesPivot",
              source: "A1:C4",
              destination: "E20:G30",
              rows: ["분류"],
              values: [{ field: "금액", function: "Sum" }],
            },
          ],
        },
      ],
    },
    dir,
  );
  assert.equal(result.warning, undefined);
  const zip = await JSZip.loadAsync(
    Buffer.from(result.files[0].data, "base64"),
  );
  assert.ok(
    Object.keys(zip.files).some((n) => n.startsWith("xl/charts/chart")),
  );
  assert.ok(
    Object.keys(zip.files).some((n) =>
      n.startsWith("xl/pivotTables/pivotTable"),
    ),
  );
  const { default: ExcelJS } = await import("exceljs");
  const wb = new ExcelJS.Workbook();
  await wb.xlsx.readFile(path.join(dir, "document.xlsx"));
  assert.equal(wb.getWorksheet("매출").getCell("C2").result, 40);
  assert.equal(wb.getWorksheet("매출").getCell("C3").result, 20);
  assert.equal(wb.getWorksheet("매출").getCell("C4").result, 2);
  console.log("ARTIFACT xlsx", dir);
});
const pixel = {
  data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
  width_px: 1,
  height_px: 1,
  width_cm: 3,
  caption: "이미지 캡션 검증",
};
test("PPTX images, media, columns and long table preserve final content", async () => {
  const dir = await fs.mkdtemp(
    path.join(process.env.ARTIFACT_ROOT || os.tmpdir(), "rich-presentation-"),
  );
  const wav = Buffer.alloc(1644);
  wav.write("RIFF", 0);
  wav.writeUInt32LE(1636, 4);
  wav.write("WAVEfmt ", 8);
  wav.writeUInt32LE(16, 16);
  wav.writeUInt16LE(1, 20);
  wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(8000, 24);
  wav.writeUInt32LE(16000, 28);
  wav.writeUInt16LE(2, 32);
  wav.writeUInt16LE(16, 34);
  wav.write("data", 36);
  wav.writeUInt32LE(1600, 40);
  const result = await render(
    {
      format: "pptx",
      title: "실제 표와 이미지",
      slides: [
        {
          title: "두 단 비교",
          blocks: [
            {
              type: "columns",
              columns: [
                [
                  { type: "paragraph", text: "왼쪽 본문" },
                  { type: "image", image: pixel },
                ],
                [
                  { type: "paragraph", text: "오른쪽 본문" },
                  { type: "shape", shape: "arrow", text: "다음 단계", h: 1 },
                ],
              ],
            },
          ],
        },
        {
          title: "오디오",
          blocks: [
            {
              type: "media",
              media: {
                mime: "audio/wav",
                data: wav.toString("base64"),
                name: "검증음.wav",
              },
            },
          ],
        },
        {
          title: "여러 장에 걸친 표",
          table: [
            ["번호", "내용"],
            ...Array.from({ length: 50 }, (_, i) => [
              String(i + 1),
              "검증 행 " + (i + 1),
            ]),
          ],
        },
      ],
    },
    dir,
  );
  assert.equal(result.warning, undefined);
  const zip = await JSZip.loadAsync(
    Buffer.from(result.files[0].data, "base64"),
  );
  assert.ok(
    Object.keys(zip.files).some(
      (n) => n.startsWith("ppt/media/") && n.endsWith(".png"),
    ),
  );
  assert.ok(
    Object.keys(zip.files).some(
      (n) => n.startsWith("ppt/media/") && n.endsWith(".wav"),
    ),
  );
  const text = execFileSync(
    "pdftotext",
    [path.join(dir, "document.pdf"), "-"],
    { encoding: "utf8" },
  );
  assert.match(text, /검증 행 50/);
  assert.match(text, /이미지 캡션 검증/);
  const slideCount = Object.keys(zip.files).filter((n) =>
    /^ppt\/slides\/slide\d+\.xml$/.test(n),
  ).length;
  const info = execFileSync("pdfinfo", [path.join(dir, "document.pdf")], {
    encoding: "utf8",
  });
  assert.match(info, new RegExp("Pages:\\s+" + slideCount + "\\b"));
  console.log("ARTIFACT advanced-pptx", dir, slideCount);
});
test("DOCX nested cell content, landscape sections and cached TOC", async () => {
  const dir = await fs.mkdtemp(
    path.join(process.env.ARTIFACT_ROOT || os.tmpdir(), "rich-nested-"),
  );
  const result = await render(
    {
      format: "docx",
      title: "중첩 문서",
      page: {
        orientation: "landscape",
        first_header: "첫 머리말",
        even_header: "짝수 머리말",
        header: "머리말",
        page_numbers: true,
      },
      sections: [
        {
          blocks: [
            { type: "toc" },
            { type: "heading", text: "목차 대상", level: 2 },
            {
              type: "table",
              widths: [1, 2],
              rows: [
                ["구분", "내용"],
                [
                  "중첩",
                  {
                    blocks: [
                      { type: "paragraph", text: "셀 안 본문" },
                      { type: "image", image: pixel },
                      { type: "table", rows: [["안쪽 표"], ["안쪽 결과"]] },
                    ],
                  },
                ],
              ],
            },
          ],
        },
        {
          page: { break_before: true, orientation: "portrait", columns: 2 },
          blocks: [
            { type: "heading", text: "두 번째 구역" },
            { type: "paragraph", text: "새 구역의 마지막 문장" },
          ],
        },
      ],
    },
    dir,
  );
  assert.equal(result.warning, undefined);
  const zip = await JSZip.loadAsync(
    Buffer.from(result.files[0].data, "base64"),
  );
  const xml = await zip.file("word/document.xml").async("string");
  assert.match(xml, /landscape/);
  assert.match(xml, /w:num="2"/);
  assert.match(xml, /안쪽 결과/);
  assert.ok((xml.match(/<w:tbl>/g) || []).length === 2);
  console.log("ARTIFACT nested-docx", dir);
});
for (const format of ["hwp", "hwpx"])
  test("Hancom page layout and page numbers " + format, async () => {
    const dir = await fs.mkdtemp(
      path.join(process.env.ARTIFACT_ROOT || os.tmpdir(), "rich-page-"),
    );
    const result = await render(
      {
        format,
        title: "쪽 설정",
        page: {
          orientation: "landscape",
          columns: 2,
          page_numbers: true,
          header: "머리말",
        },
        sections: [
          {
            blocks: [
              { type: "paragraph", text: "첫 문단의 한글 본문" },
              { type: "page_break" },
              { type: "paragraph", text: "페이지 나눔 뒤 본문" },
            ],
          },
        ],
      },
      dir,
    );
    assert.equal(result.warning, undefined);
    const { HwpDocument } = await import("@rhwp/core");
    const doc = new HwpDocument(Buffer.from(result.files[0].data, "base64"));
    try {
      assert.ok(doc.pageCount() >= 2);
      assert.match(JSON.parse(doc.getTextFileUnicode()), /페이지 나눔 뒤 본문/);
    } finally {
      doc.free();
    }
    console.log("ARTIFACT hancom-page", format, dir);
  });

test("XLSX native merges, images, tables, rich text, protection and print options", async () => {
  const dir = await fs.mkdtemp(
    path.join(process.env.ARTIFACT_ROOT || os.tmpdir(), "rich-sheet-options-"),
  );
  const result = await render(
    {
      format: "xlsx",
      title: "셀 기능",
      sheets: [
        {
          name: "병합",
          columns: [{ title: "A" }, { title: "B" }, { title: "C" }],
          rows: [
            ["병합 셀", null, 12],
            ["링크", null, 21],
          ],
          merges: ["A2:B2"],
          cells: [
            {
              cell: "A3",
              link: "https://example.com",
              note: "셀 메모",
              style: { bold: true, color: "224488" },
            },
          ],
          images: [{ cell: "E2", image: pixel, width: 80, height: 80 }],
          protection: { password: "sample" },
          print: {
            area: "A1:C3",
            repeat_rows: "1:1",
            repeat_columns: "A:A",
            header: "검증 머리말",
            footer: "&P / &N",
            paper: "A4",
            margin: 0.3,
            scale: 90,
          },
        },
        {
          name: "테이블",
          columns: [{ title: "항목" }, { title: "값" }],
          rows: [
            [
              {
                runs: [
                  { text: "빨강", style: { color: "AA0000", bold: true } },
                  { text: " 보통" },
                ],
              },
              1,
            ],
            ["둘째", 2],
          ],
          table_name: "SampleTable",
          freeze: { rows: 1, columns: 1 },
          row_options: [{ index: 3, hidden: true, level: 1 }],
          column_options: [{ index: 2, size: 24 }],
        },
        { name: "숨김", hidden: true, columns: [{ title: "값" }], rows: [[1]] },
      ],
    },
    dir,
  );
  assert.equal(result.warning, undefined);
  const zip = await JSZip.loadAsync(
    Buffer.from(result.files[0].data, "base64"),
  );
  const names = Object.keys(zip.files);
  assert.ok(names.some((n) => n.startsWith("xl/media/")));
  assert.ok(names.some((n) => n.startsWith("xl/tables/table")));
  const xml = await zip.file("xl/worksheets/sheet1.xml").async("string");
  assert.match(xml, /mergeCell ref="A2:B2"/);
  assert.match(xml, /sheetProtection/);
  const { default: ExcelJS } = await import("exceljs");
  const wb = new ExcelJS.Workbook();
  await wb.xlsx.readFile(path.join(dir, "document.xlsx"));
  assert.equal(
    wb.getWorksheet("병합").getCell("A3").hyperlink,
    "https://example.com",
  );
  assert.equal(
    wb.getWorksheet("테이블").getCell("A2").value.richText[0].font.bold,
    true,
  );
  assert.equal(wb.getWorksheet("테이블").getRow(3).hidden, true);
  assert.equal(wb.getWorksheet("숨김").state, "hidden");
  console.log("ARTIFACT xlsx-options", dir);
});
for (const kind of ["bar", "line", "area", "pie", "doughnut", "scatter"])
  test("PPTX native chart " + kind, async () => {
    const dir = await fs.mkdtemp(
      path.join(
        process.env.ARTIFACT_ROOT || os.tmpdir(),
        "rich-chart-" + kind + "-",
      ),
    );
    const result = await render(
      {
        format: "pptx",
        title: "차트",
        slides: [
          {
            title: kind,
            blocks: [
              {
                type: "chart",
                chart: {
                  kind,
                  labels: ["1", "4"],
                  series: [{ name: "검증", values: [2, 3] }],
                },
              },
            ],
          },
        ],
      },
      dir,
    );
    assert.equal(result.warning, undefined);
    const zip = await JSZip.loadAsync(
      Buffer.from(result.files[0].data, "base64"),
    );
    const chart = await zip
      .file(
        Object.keys(zip.files).find((n) =>
          /^ppt\/charts\/chart\d+\.xml$/.test(n),
        ),
      )
      .async("string");
    assert.match(
      chart,
      new RegExp(
        "<c:" +
          {
            bar: "bar",
            line: "line",
            area: "area",
            pie: "pie",
            doughnut: "doughnut",
            scatter: "scatter",
          }[kind] +
          "Chart",
      ),
    );
    if (kind === "scatter") {
      assert.match(chart, /<c:xVal>/);
      assert.match(chart, /<c:v>4<\/c:v>/);
    }
  });


test('slide image honors requested centimeter width',async()=>{
 const {presentationPlan}=await import('./presentation.mjs');
 const pages=presentationPlan({slides:[{title:'image',blocks:[{type:'image',image:pixel}]}]});
 assert.ok(Math.abs(pages[0].items[0].w-3*72/2.54)<.001);
});

test("PPTX table paths share widths and report saved structure", async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "table-report-"));
  try {
    const rows = [["구분", "설명"], ["경복궁", "긴 설명을 위한 넓은 열"]];
    const results=[];
    for(const legacy of [true,false]) {
      const table={rows,widths:[1,4],style:{size:14}};
      const out=await render({format:"pptx",title:"너비 검사",slides:[{title:"비교",...(legacy?{table}:{blocks:[{type:"table",...table}]})}]},dir);
      assert.equal(out.files.length,2);
      assert.equal(out.presentation.slide_count,1);
      assert.equal(out.presentation.visually_verified,false);
      assert.equal(out.presentation.tables[0].split,false);
      const widths=out.presentation.tables[0].column_widths_inches;
      assert.ok(Math.abs(widths[1]/widths[0]-4)<1e-5);
      results.push(out.presentation);
    }
    assert.deepEqual(results[0],results[1]);
    const merged=await render({format:"pptx",title:"병합",slides:[{title:"병합",table:{rows:[[{text:"병합",col_span:2}],["가","나"]],widths:[1,3]}}]},dir);
    assert.equal(merged.presentation.tables[0].split,false);
    const split=await render({format:"pptx",title:"분할",slides:[{title:"분할",table:{rows:[rows[0],...Array.from({length:25},(_,i)=>[String(i),"설명"])],widths:[1,4]}}]},dir);
    assert.ok(split.presentation.slide_count>1);
    assert.equal(split.presentation.tables[0].split,true);
    assert.equal(split.presentation.tables[0].slide_numbers.length,split.presentation.slide_count);
    assert.equal(split.presentation.tables[0].rows_per_slide.reduce((a,b)=>a+b,0),26+split.presentation.slide_count-1);
  } finally { await fs.rm(dir,{recursive:true,force:true}); }
});

test("cell.width resolves grid widths, merged sums and conflicts", async () => {
  const {tableRows}=await import('./blocks.mjs');
  const rows=[[{text:'구분',width:7},{text:'건립',width:11},{text:'역할',width:19},{text:'건물',width:23},{text:'설명',width:40}],['경복궁','시기','역할','건물','긴 설명']];
  const dir=await fs.mkdtemp(path.join(os.tmpdir(),'cell-width-'));
  try {
    for(const legacy of [true,false]) {
      const table={rows:structuredClone(rows)};
      const result=await render({format:'pptx',title:'셀 너비',slides:[{title:'검증',...(legacy?{table}:{blocks:[{type:'table',...table}]})}]},dir);
      const widths=result.presentation.tables[0].column_widths_inches;
      [0.84,1.32,2.28,2.76,4.8].forEach((v,i)=>assert.ok(Math.abs(widths[i]-v)<1e-6));
      assert.equal(result.presentation.slide_count,1);
      assert.equal(result.files.length,2);
    }
    const merged={rows:[[{text:'합계',col_span:2,width:10}],[{text:'첫 열',width:3},'둘째 열']]};
    tableRows(merged);assert.deepEqual(merged.widths,[3,7]);
    const equal={rows:[[{text:'합계',col_span:2,width:1}],['가','나']]};
    tableRows(equal);assert.deepEqual(equal.widths,[.5,.5]);
    const vertical={rows:[[{text:'세로',row_span:2,width:2},{text:'옆',width:5}],['아래']]};
    tableRows(vertical);assert.deepEqual(vertical.widths,[2,5]);
    assert.throws(()=>tableRows({rows:[[{text:'가',width:2}], [{text:'나',width:3}]]}),/Conflicting/);
    assert.throws(()=>tableRows({rows:[[{text:'가',width:2}]],widths:[3]}),/Conflicting/);
    assert.throws(()=>tableRows({rows:[[{text:'가',width:0}]]}),/dimension/);
  } finally { await fs.rm(dir,{recursive:true,force:true}); }
});
