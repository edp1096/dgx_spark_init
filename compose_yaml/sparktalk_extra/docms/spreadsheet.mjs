import ExcelJS from 'exceljs';
import fs from 'node:fs/promises';
import path from 'node:path';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {renderPDF} from './pdf.mjs';
import {FONT,COLOR} from './layout.mjs';
const exec=promisify(execFile);
const formats={general:'General',number:'#,##0.00',integer:'#,##0',currency:'#,##0.00',percent:'0.00%',date:'yyyy-mm-dd'};
const functions=new Set(['SUM','AVERAGE','COUNT','COUNTA','MIN','MAX','IF','AND','OR','NOT','ROUND','ROUNDUP','ROUNDDOWN','ABS','COUNTIF','SUMIF','IFERROR']);
function object(value,allowed){if(!value||typeof value!=='object'||Array.isArray(value)||Object.keys(value).some(k=>!allowed.includes(k)))throw Error('Unsupported spreadsheet field');}
function string(value,max){if(typeof value!=='string'||value.length>max||/[\x00-\x08\x0b\x0c\x0e-\x1f]/.test(value))throw Error('Invalid spreadsheet text');}
function columnNumber(s){return [...s.toUpperCase()].reduce((n,c)=>n*26+c.charCodeAt(0)-64,0);}
function references(formula,sheet,sheets){
 string(formula,512);formula=formula.replace(/^=/,'');if(!formula.trim())throw Error('Empty formula');
 const refs=[];let rest=formula;const stack=[];let pendingFunction;
 while(rest.length){
  let m;
  if((m=/^\s+/.exec(rest))||(m=/^"(?:[^"]|"")*"/.exec(rest))||(m=/^(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?/.exec(rest))){rest=rest.slice(m[0].length);continue;}
  if((m=/^([A-Za-z]+)\s*(?=\()/.exec(rest))){if(!functions.has(m[1].toUpperCase()))throw Error(`Unsupported formula function: ${m[1]}`);pendingFunction=m[1].toUpperCase();rest=rest.slice(m[0].length);continue;}
  m=/^(?:(?:'((?:[^']|'')+)'|([\p{L}_][\p{L}\p{N}_ ]*))!)?(\$?[A-Za-z]{1,2}\$?[1-9]\d{0,3})(?::(\$?[A-Za-z]{1,2}\$?[1-9]\d{0,3}))?/u.exec(rest);
  if(m){
   if(m[4]&&['AND','OR','NOT'].includes(stack.findLast(Boolean)))throw Error('AND/OR/NOT require individual cells or comparisons, not ranges');
   const targetName=m[1]?.replace(/''/g,"'")??m[2]??sheet.name;
   const target=sheets.find(s=>s.name.toLowerCase()===targetName.toLowerCase());if(!target)throw Error(`Unknown formula sheet: ${targetName}`);
   const parse=a=>{const v=/^([A-Za-z]+)(\d+)$/.exec(a.replaceAll('$',''));return [columnNumber(v[1]),Number(v[2])];};
   const [c1,r1]=parse(m[3]),[c2,r2]=parse(m[4]||m[3]);
   if(c2<c1||r2<r1||c2>target.columns.length||r2>target.rows.length+1)throw Error('Formula reference is outside the supplied table');
   for(let r=r1;r<=r2;r++)for(let c=c1;c<=c2;c++)refs.push(`${target.name.toLowerCase()}!${r}:${c}`);
   rest=rest.slice(m[0].length);continue;
  }
  if((m=/^(?:TRUE|FALSE)\b/i.exec(rest))||(m=/^[+\-*/^%&=<>(),]/.exec(rest))){if(m[0]==='('){stack.push(pendingFunction);pendingFunction=undefined;}else if(m[0]===')'){stack.pop();}rest=rest.slice(m[0].length);continue;}
  throw Error('Unsupported formula syntax; use cell references, basic operators and supported functions');
 }
 return refs;
}
export function validateSheets(sheets){
 if(!Array.isArray(sheets)||sheets.length<1||sheets.length>8)throw Error('Provide 1–8 sheets');
 const names=new Set();let count=0;
 for(const s of sheets){
  object(s,['name','columns','rows','freeze_header','filter','orientation']);string(s.name,31);
  if(!s.name.trim()||/[\\/*?:\[\]]/.test(s.name)||/^'|'$/.test(s.name)||names.has(s.name.toLowerCase()))throw Error('Invalid or duplicate sheet name');names.add(s.name.toLowerCase());
  if(s.orientation!==undefined&&!['portrait','landscape'].includes(s.orientation))throw Error('Invalid page orientation');
  for(const k of ['freeze_header','filter'])if(s[k]!==undefined&&typeof s[k]!=='boolean')throw Error(`Invalid ${k}`);
  if(!Array.isArray(s.columns)||s.columns.length<1||s.columns.length>32)throw Error('Provide 1–32 columns');
  for(const c of s.columns){object(c,['title','width','format']);string(c.title,120);if(c.width!==undefined&&(!Number.isFinite(c.width)||c.width<6||c.width>60))throw Error('Column width must be 6–60');if(c.format!==undefined&&!Object.hasOwn(formats,c.format))throw Error('Unsupported cell format');}
  if(!Array.isArray(s.rows)||s.rows.length>1000)throw Error('At most 1000 rows per sheet');count+=(s.rows.length+1)*s.columns.length;
  for(const row of s.rows){if(!Array.isArray(row)||row.length!==s.columns.length)throw Error('Each row must match the columns');for(const v of row){
   if(v===null||typeof v==='boolean'||(typeof v==='number'&&Number.isFinite(v)))continue;
   if(typeof v==='string'){string(v,2000);continue;}
   object(v,['formula','date']);if(Object.keys(v).length!==1)throw Error('Cell must contain formula or date');
   if(v.formula!==undefined)string(v.formula,512);
   else if(typeof v.date!=='string'||!/^\d{4}-\d{2}-\d{2}$/.test(v.date)||!Number.isFinite(Date.parse(v.date))||new Date(v.date).toISOString().slice(0,10)!==v.date||v.date<'1900-01-01')throw Error('Date must be valid YYYY-MM-DD from 1900');
  }}
 }
 if(count>20000)throw Error('Workbook exceeds 20000 cells');
 const graph=new Map();let refCount=0;
 for(const s of sheets)for(const [r,row] of s.rows.entries())for(const [c,v] of row.entries())if(v?.formula!==undefined){const refs=references(v.formula,s,sheets);refCount+=refs.length;if(refCount>100000)throw Error('Too many formula references');graph.set(`${s.name.toLowerCase()}!${r+2}:${c+1}`,refs);}
 const visited=new Set(),active=new Set();
 function visit(key,depth=0){if(active.has(key))throw Error('Circular formula reference');if(visited.has(key)||!graph.has(key))return;if(depth>128)throw Error('Formula dependency chain too long');active.add(key);for(const ref of graph.get(key))visit(ref,depth+1);active.delete(key);visited.add(key);}
 for(const key of graph.keys())visit(key);
}
export async function renderSpreadsheet(input,dir){
 const workbook=new ExcelJS.Workbook();workbook.creator='SparkTalk';workbook.title=input.title;workbook.calcProperties.fullCalcOnLoad=true;
 for(const s of input.sheets){
  const ws=workbook.addWorksheet(s.name,{views:s.freeze_header===false?[]:[{state:'frozen',ySplit:1}],pageSetup:{paperSize:9,orientation:s.orientation||(s.columns.length>5?'landscape':'portrait'),fitToPage:true,fitToWidth:1,fitToHeight:0,printTitlesRow:'1:1'}});
  ws.columns=s.columns.map(c=>({header:c.title,width:c.width||18,style:{numFmt:formats[c.format||'general'],font:{name:FONT,size:11},alignment:{vertical:'top',wrapText:true}}}));
  for(const row of s.rows)ws.addRow(row.map(v=>v?.date?new Date(v.date+'T00:00:00Z'):v?.formula?{formula:v.formula.replace(/^=/,'')}:v));
  ws.eachRow(row=>{let lines=1;row.eachCell((cell,col)=>{if(cell.value instanceof Date)cell.numFmt='yyyy-mm-dd';if(typeof cell.value==='string'){const width=s.columns[col-1].width||18;lines=Math.max(lines,...cell.value.split('\n').map(t=>Math.ceil([...t].reduce((n,c)=>n+(c.charCodeAt(0)>255?2:1),0)/Math.max(1,width-2))));}});row.height=Math.min(409,Math.max(24,lines*16));});
  ws.getRow(1).eachCell(c=>{c.font={name:FONT,size:11,bold:true,color:{argb:'FFFFFFFF'}};c.fill={type:'pattern',pattern:'solid',fgColor:{argb:'FF'+COLOR.accent}};});
  if(s.filter!==false)ws.autoFilter={from:{row:1,column:1},to:{row:s.rows.length+1,column:s.columns.length}};
  ws.pageSetup.printArea=`A1:${ws.getCell(s.rows.length+1,s.columns.length).address}`;
 }
 const inputDir=path.join(dir,'source');await fs.mkdir(inputDir,{recursive:true});
 const source=path.join(inputDir,'document.xlsx');await workbook.xlsx.writeFile(source);
 const {stdout}=await exec(process.env.DOCUMENT_CALC_BIN||'/usr/local/bin/document-calc',[source],{timeout:60000,maxBuffer:8<<20});
 const results=JSON.parse(stdout);const values=new Map(results.map(r=>[JSON.stringify([r.sheet,r.cell]),r.value]));
 for(const ws of workbook.worksheets)ws.eachRow(row=>row.eachCell(cell=>{if(cell.formula){const key=JSON.stringify([ws.name,cell.address]);if(!values.has(key))throw Error('Missing formula result');const result=values.get(key);if(!['number','string','boolean'].includes(typeof result))throw Error('Invalid formula result');cell.value={formula:cell.formula,result};}}));
 const output=path.join(dir,'document.xlsx');
 // Preserve original cell types, styles and formulas with typed calculation results.
 await workbook.xlsx.writeFile(output);
 const data=await fs.readFile(output);if(data.length>24*1024*1024)throw Error('XLSX output too large');
 const files=[{name:'document.xlsx',mime:'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',data:data.toString('base64')}];
 try{files.push(await renderPDF(input,dir,workbook));return {files};}catch{return {files,warning:'PDF could not be generated; the calculated XLSX file is available.'};}
}
