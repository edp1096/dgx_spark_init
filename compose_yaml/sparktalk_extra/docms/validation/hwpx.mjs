import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import init,{HwpDocument} from '@rhwp/core';
const output=path.resolve(process.argv[2]||'output');await fs.mkdir(output,{recursive:true});
await init({module_or_path:await fs.readFile(new URL('./node_modules/@rhwp/core/rhwp_bg.wasm',import.meta.url))});
function ok(raw){const value=JSON.parse(raw);assert.notEqual(value.ok,false);return value;}
const doc=HwpDocument.createEmpty();
try{
 doc.createBlankDocument();ok(doc.insertText(0,0,0,'SparkTalk 한글 문서 검증'));
 const table=ok(doc.createTable(0,0,'SparkTalk 한글 문서 검증'.length,3,2));
 for(const [row,values] of [['항목','결과'],['한글','가나다라마바사'],['English','Sample 123']].entries())for(const [col,text] of values.entries())ok(doc.insertTextInCell(0,table.paraIdx,table.controlIdx,row*2+col,0,0,text));
 const image=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=','base64');
 const picture=ok(doc.insertPicture(0,table.paraIdx+1,0,'',image,7200,7200,1,1,'png','검증용 이미지'));
 // The insertion API currently defaults to floating at paper (0, 0).
 // Explicitly bind the image to its paragraph instead of relying on that default.
 ok(doc.setPictureProperties(0,picture.paraIdx,picture.controlIdx,JSON.stringify({treatAsChar:true,vertRelTo:'Para',horzRelTo:'Para',vertOffset:0,horzOffset:0})));
 const anchor=JSON.parse(doc.getPictureProperties(0,picture.paraIdx,picture.controlIdx));
 assert.equal(anchor.treatAsChar,true,'Image must follow paragraph flow');
 const counts=[];
 for(const ext of ['hwp','hwpx']){
  const bytes=ext==='hwp'?doc.exportHwp():doc.exportHwpx();
  await fs.writeFile(path.join(output,`sample.${ext}`),bytes);
  const reopened=new HwpDocument(bytes);
  try{assert.equal(JSON.parse(reopened.getPictureProperties(0,picture.paraIdx,picture.controlIdx)).treatAsChar,true,'Inline image anchor lost on save');assert.ok(reopened.pageCount()>0);counts.push(reopened.pageCount());const pages=[];for(let i=0;i<reopened.pageCount();i++)pages.push(reopened.renderPageSvg(i));assert.ok(pages.join('').includes('<image'),'Image missing after round-trip');assert.ok(pages.join('').replace(/<[^>]*>/g,'').replace(/\s+/g,'').includes('Sample123'),'Cell text missing after round-trip');for(const [index,svg] of pages.entries())await fs.writeFile(path.join(output,`${ext}-page-${index+1}.svg`),svg);
  console.log(ext,'text characters per page:',pages.map(svg=>svg.replace(/<[^>]*>/g,'').replace(/\s+/g,'').length));console.log(ext,bytes.length,'bytes',reopened.pageCount(),'pages; rhwp round-trip passed');}finally{reopened.free();}
 }
 assert.equal(counts[0],counts[1],'HWP/HWPX page counts differ');
}finally{doc.free();}
console.log('Hancom Office open/edit/save compatibility still requires independent verification.');
