import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {render,validate} from './renderer.mjs';
test('reject invalid shape and oversized slide instead of silently clipping',()=>{
 assert.throws(()=>validate({format:'pptx',title:'test',slides:[{title:'slide',bullets:['x'.repeat(201)]}]}));
 assert.throws(()=>validate({format:'docx',title:'test',sections:[{paragraphs:[],table:[['a','b'],['c']]}]}));
 assert.throws(()=>validate({format:'html',title:'test'}));
});
for(const format of ['docx','pptx','pdf'])test(`render ${format} with searchable Korean PDF`,async()=>{
 const dir=await fs.mkdtemp(path.join(os.tmpdir(),'document-test-'));
 try{
  const input={format,title:'한글 보고서',sections:[{heading:'검증 결과',paragraphs:['한글 본문 English 123'],table:[['항목','결과'],...Array.from({length:80},(_,i)=>[`${i+1}`,'한글 표 내용'])]}],slides:[{title:'한글 발표자료',bullets:['한글 본문 English 123','생성 결과 확인']}]};
  const result=await render(input,dir);assert.equal(result.files.length,format==='pdf'?1:2);
  const pdf=result.files.at(-1);assert.equal(pdf.mime,'application/pdf');
  const extracted=execFileSync('pdftotext',[path.join(dir,'document.pdf'),'-'],{encoding:'utf8'});assert.match(extracted,/한글/);assert.match(extracted,/English 123/);
  if(format!=='pptx')assert.match(extracted,/80/);
 }finally{await fs.rm(dir,{recursive:true,force:true});}
});

const spreadsheet=()=>({format:'xlsx',title:'매출 및 예산',sheets:[
 {name:'매출',columns:[{title:'날짜',format:'date'},{title:'항목',width:24},{title:'금액',format:'currency'},{title:'부가세',format:'currency'}],rows:[[{date:'2026-09-07'},'상품 A',100,{formula:'C2*0.1'}],[{date:'2026-09-08'},'=SUM(1,2)',200,{formula:'ROUND(C3*0.1,2)'}],[null,'합계',{formula:'SUM(C2:C3)'},{formula:'SUM(D2:D3)'}]]},
 {name:'요약',columns:[{title:'항목'},{title:'결과',format:'number'}],rows:[['총 매출',{formula:"'매출'!C4"}],['평균',{formula:"AVERAGE('매출'!C2:C3)"}],['판정',{formula:'IF(B2>200,"충족","미달")'}],['참거짓',true],['빈칸',null]]}
]});
test('spreadsheet rejects unsafe formulas, cycles, invalid dates and shape',()=>{
 for(const formula of ['WEBSERVICE("https://example.com")','[book.xlsx]Sheet1!A1','B2','SUM(A1:A9999)',"'missing'!A1",'INDIRECT("A1")','OR(A1:A2,TRUE)','AND(A1:A2)']){
  const input={format:'xlsx',title:'test',sheets:[{name:'Sheet1',columns:[{title:'a'},{title:'b'}],rows:[[1,{formula}]]}]};assert.throws(()=>validate(input),formula);
 }
 const input=spreadsheet();input.sheets[0].rows[0][0]={date:'2026-02-30'};assert.throws(()=>validate(input));
 const duplicate=spreadsheet();duplicate.sheets[1].name='매출';assert.throws(()=>validate(duplicate));
 const wrong=spreadsheet();wrong.sheets[0].rows[0].pop();assert.throws(()=>validate(wrong));
});
test('XLSX recalculates cross-sheet formulas, retains types/styles and prints Korean',async()=>{
 const {default:ExcelJS}=await import('exceljs');const dir=await fs.mkdtemp(path.join(os.tmpdir(),'xlsx-test-'));
 try{
  const result=await render(spreadsheet(),dir);assert.equal(result.files.length,2);assert.equal(result.warning,undefined);
  const wb=new ExcelJS.Workbook();await wb.xlsx.readFile(path.join(dir,'document.xlsx'));
  assert.equal(wb.getWorksheet('매출').getCell('C4').result,300);assert.equal(wb.getWorksheet('매출').getCell('D4').result,30);
  assert.equal(wb.getWorksheet('요약').getCell('B2').result,300);assert.equal(wb.getWorksheet('요약').getCell('B3').result,150);assert.equal(wb.getWorksheet('요약').getCell('B4').result,'충족');
  assert.equal(wb.getWorksheet('매출').getCell('B3').value,'=SUM(1,2)');assert.ok(wb.getWorksheet('매출').getCell('A2').value instanceof Date);
  assert.equal(wb.getWorksheet('요약').getCell('B5').value,true);assert.equal(wb.getWorksheet('매출').views[0].ySplit,1);assert.ok(wb.getWorksheet('매출').autoFilter);assert.match(wb.getWorksheet('매출').getCell('C2').numFmt,/0\.00/);
  const text=execFileSync('pdftotext',[path.join(dir,'document.pdf'),'-'],{encoding:'utf8'});assert.match(text,/총 매출/);assert.match(text,/300/);assert.match(text,/충족/);
 }finally{await fs.rm(dir,{recursive:true,force:true});}
});
test('XLSX calculation errors fail instead of returning an apparently valid workbook',async()=>{
 const dir=await fs.mkdtemp(path.join(os.tmpdir(),'xlsx-error-'));
 try{const input=spreadsheet();input.sheets[0].rows[0][3]={formula:'1/0'};await assert.rejects(render(input,dir),/Formula calculation failed/);}finally{await fs.rm(dir,{recursive:true,force:true});}
});
test('Office originals remain downloadable when PDF generation fails',async()=>{
 const {default:pdfmake}=await import('pdfmake');const previous=pdfmake.createPdf;pdfmake.createPdf=()=>{throw Error('simulated PDF failure');};
 try{for(const format of ['docx','pptx','xlsx']){const dir=await fs.mkdtemp(path.join(os.tmpdir(),'pdf-original-'));try{
  const input=format==='xlsx'?spreadsheet():{format,title:'Original',sections:[{paragraphs:['text']}],slides:[{title:'Slide',bullets:['text']}]};
  const result=await render(input,dir);assert.equal(result.files.length,1);assert.equal(result.files[0].name,'document.'+format);assert.match(result.warning,/PDF/);
 }finally{await fs.rm(dir,{recursive:true,force:true});}}}finally{pdfmake.createPdf=previous;}
});
test('long Korean spreadsheet prints repeated headers and the final row',async()=>{
 const dir=await fs.mkdtemp(path.join(os.tmpdir(),'xlsx-long-'));try{
  const input={format:'xlsx',title:'긴 한글 표',sheets:[{name:'상세',orientation:'landscape',columns:Array.from({length:8},(_,i)=>({title:`항목 ${i+1}`,width:18})),rows:Array.from({length:100},(_,r)=>Array.from({length:8},(_,c)=>c===0?`행 ${r+1}`:`한글 내용 ${r+1}-${c+1}`))}]};
  const result=await render(input,dir);assert.equal(result.files.length,2);
  const text=execFileSync('pdftotext',[path.join(dir,'document.pdf'),'-'],{encoding:'utf8'});assert.match(text,/한글 내용 100-8/);assert.ok((text.match(/항목 1/g)||[]).length>1);
 }finally{await fs.rm(dir,{recursive:true,force:true});}
});

test('slides remain one PDF page each with searchable final bullet',async()=>{
 const dir=await fs.mkdtemp(path.join(os.tmpdir(),'slides-layout-'));try{
 const input={format:'pptx',title:'발표',slides:Array.from({length:3},(_,i)=>({title:`슬라이드 ${i+1}`,bullets:['한글 발표 내용 '.repeat(10),'마지막 항목 '+i]}))};
 const result=await render(input,dir);assert.equal(result.files.length,2);
 const info=execFileSync('pdfinfo',[path.join(dir,'document.pdf')],{encoding:'utf8'});assert.match(info,/Pages:\s+3/);
 const text=execFileSync('pdftotext',[path.join(dir,'document.pdf'),'-'],{encoding:'utf8'});assert.match(text,/마지막 항목 2/);
 }finally{await fs.rm(dir,{recursive:true,force:true});}
});
test('dense Korean slide content stays inside its page',async()=>{
 const dir=await fs.mkdtemp(path.join(os.tmpdir(),'slide-bounds-'));try{
  await render({format:'pptx',title:'밀도 검증',slides:[{title:'한글 제목 '.repeat(20),bullets:Array.from({length:8},(_,i)=>'한글 내용 '.repeat(16)+String(i))}]},dir);
  const bbox=execFileSync('pdftotext',['-bbox',path.join(dir,'document.pdf'),'-'],{encoding:'utf8'});assert.equal((bbox.match(/<page /g)||[]).length,1);
  for(const word of bbox.matchAll(/<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)"/g)){assert.ok(Number(word[3])<=925,'text exceeds right edge');assert.ok(Number(word[4])<=530,'text exceeds bottom edge');}
 }finally{await fs.rm(dir,{recursive:true,force:true});}
});
test('filters include a hidden scoped database name for each sheet',async()=>{
 const {default:JSZip}=await import('jszip');const dir=await fs.mkdtemp(path.join(os.tmpdir(),'xlsx-filter-'));try{
  const result=await render(spreadsheet(),dir);const zip=await JSZip.loadAsync(Buffer.from(result.files[0].data,'base64'));
  const xml=await zip.file('xl/workbook.xml').async('string');const names=[...xml.matchAll(/<definedName\b([^>]*)>([^<]*)<\/definedName>/g)].filter(m=>m[1].includes('_xlnm._FilterDatabase'));
  assert.equal(names.length,2);for(const [index,m] of names.entries()){assert.match(m[1],new RegExp(`localSheetId="${index}"`));assert.match(m[1],/hidden="(?:1|true)"/);assert.match(m[2],/!\$A\$1:\$[BD]\$[46]/);}
 }finally{await fs.rm(dir,{recursive:true,force:true});}
});

for(const format of ['hwp','hwpx'])test(`${format} preserves tables, inline images and following paragraphs without a spurious page break`,async()=>{
 const {HwpDocument}=await import('@rhwp/core');const {default:JSZip}=await import('jszip');
 const dir=await fs.mkdtemp(path.join(os.tmpdir(),'hancom-'));
 const data='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';
 try{
  const result=await render({format,title:'한글 문서 검증',sections:[{heading:'첫 번째 절',paragraphs:['본문 English 123'],table:[['항목','결과'],['검사','정상']],images:[{data,width_px:1,height_px:1,width_cm:2,caption:'그림 설명'}]},{heading:'두 번째 절',paragraphs:['마지막 본문 보존']}]},dir);
  assert.equal(result.files.length,2);assert.equal(result.page_count,1);assert.match(result.text,/마지막 본문 보존/);assert.match(result.text,/그림 설명/);assert.match(result.text,/정상/);assert.ok(result.text.includes('\r\n'));
  const bytes=Buffer.from(result.files[0].data,'base64');
  if(format==='hwp')assert.equal(bytes.subarray(0,8).toString('hex'),'d0cf11e0a1b11ae1');
  const restored=new HwpDocument(bytes);try{const svg=restored.renderPageSvg(0);assert.match(svg,/<image/);const imageTag=svg.match(/<image\b[^>]*>/)[0];const imageY=Number(imageTag.match(/\by="([^"]+)"/)[1]),imageHeight=Number(imageTag.match(/\bheight="([^"]+)"/)[1]);const captionY=Number(svg.match(/<text\b[^>]*\by="([^"]+)"[^>]*>그<\/text>/)[1]);assert.ok(captionY>imageY+imageHeight,'Caption must follow the image on a separate paragraph');const edit=JSON.parse(restored.insertText(0,0,0,'수정 '));assert.notEqual(edit.ok,false);const saved=new HwpDocument(format==='hwp'?restored.exportHwp():restored.exportHwpx());try{assert.match(JSON.parse(saved.getTextFileUnicode()),/수정 한글/);assert.match(saved.renderPageSvg(0),/<image/);}finally{saved.free();}}finally{restored.free();}
  if(format==='hwpx'){const zip=await JSZip.loadAsync(bytes);for(const name of Object.keys(zip.files))if(name.startsWith('META-INF/rhwp'))zip.remove(name);const standard=new HwpDocument(await zip.generateAsync({type:'uint8array'}));try{assert.equal(standard.pageCount(),1);assert.match(standard.renderPageSvg(0),/<image/);assert.match(JSON.parse(standard.getTextFileUnicode()),/마지막 본문 보존/);}finally{standard.free();}}
  const pdfText=execFileSync('pdftotext',[path.join(dir,'document.pdf'),'-'],{encoding:'utf8'});assert.match(pdfText,/마지막 본문 보존/);assert.match(pdfText,/그림 설명/);
 }finally{await fs.rm(dir,{recursive:true,force:true});}
});
test('images reject invalid bytes and unsupported slideshow use',()=>{
 assert.throws(()=>validate({format:'hwp',title:'test',sections:[{paragraphs:[],images:[{data:'AAAA',width_px:1,height_px:1}]}]}));
 assert.throws(()=>validate({format:'pptx',title:'test',slides:[{title:'s',bullets:[]}],sections:[{images:[{}]}]}));
});
