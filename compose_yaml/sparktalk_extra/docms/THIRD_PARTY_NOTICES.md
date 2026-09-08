# Document engine dependencies

- docx 9.7.1, PptxGenJS 4.0.1, ExcelJS 4.4.0, pdfmake 0.3.11: MIT.
- Excelize 2.11.0: BSD-3-Clause. The build applies corrections to AND/OR branches to preserve boolean results. Source and regression tests are included in `calc/`; rebuilding applies it automatically.
- ExcelJS 4.4.0: the build adds scoped `_xlnm._FilterDatabase` names and their hidden flag, preserving worksheet filters across LibreOffice open/save. `patch-exceljs.mjs` contains the guarded build-time correction.
- rhwp 0.8.6: MIT, pinned to `e8800c8def63449808a4092798442652ed460552`. `hwp-engine/` rebuilds its WASM with guarded corrections to table common-header length and synthetic line-position page breaks. Rust dependency licenses and notices: `/licenses/rhwp/manifest.json` and `/licenses/rhwp/`.
- JSZip 3.10.1: MIT option selected.
- JavaScript dependency versions/licenses: `/licenses/npm.json`; original notices remain in `/app/node_modules`.
- Compiled Go dependency notices: `/licenses/go`.
- Noto CJK fonts: SIL Open Font License 1.1; Debian notice in `/usr/share/doc/fonts-noto-cjk/copyright`.

Transitive libraries also use permissive ISC, Zlib, 0BSD, BlueOak and Unlicense terms. For dual MIT/GPL packages, the MIT option is used. This list describes document engine libraries; the Node/Debian runtime has its own included notices. LibreOffice and Poppler are not part of the production image.
