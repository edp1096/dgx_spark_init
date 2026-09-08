// ExcelJS 4.4.0 writes sheet autoFilter but omits the scoped FilterDatabase name.
// LibreOffice drops the filter on reopen/save without that definition.
import fs from 'node:fs';
const root='/app/node_modules/exceljs/';
if(JSON.parse(fs.readFileSync(root+'package.json')).version!=='4.4.0')throw Error('Review the ExcelJS filter patch for this version');
function replace(file,before,after){const path=root+file,source=fs.readFileSync(path,'utf8');if(source.split(before).length!==2)throw Error('ExcelJS source changed: '+file);fs.writeFileSync(path,source.replace(before,()=>after));}
const marker='    model.sheets.forEach(sheet => {\n';
replace('lib/xlsx/xform/book/workbook-xform.js',marker,marker+`
      if (sheet.autoFilter) {
        const address = v => typeof v === 'string' ? v : colCache.n2l(v.column) + v.row;
        const range = typeof sheet.autoFilter === 'string'
          ? sheet.autoFilter : address(sheet.autoFilter.from) + ':' + address(sheet.autoFilter.to);
        const absolute = range.replace(/\\$?([A-Z]+)\\$?(\\d+)/g, (_, col, row) => '$' + col + '$' + row);
        printAreas.push({name: '_xlnm._FilterDatabase', localSheetId: index, hidden: true,
          ranges: ["'" + sheet.name.replace(/'/g, "''") + "'!" + absolute]});
      }
`);
replace('lib/xlsx/xform/book/defined-name-xform.js','      localSheetId: model.localSheetId,','      localSheetId: model.localSheetId,\n      hidden: model.hidden,');
