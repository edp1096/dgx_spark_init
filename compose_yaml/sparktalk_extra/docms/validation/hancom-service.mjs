// Run in the built DocMS image; this deliberately uses its patched WASM.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {deflateSync} from 'node:zlib';
import {render} from '/app/renderer.mjs';
import init,{HwpDocument} from '/app/node_modules/@rhwp/core/rhwp.js';
const output=path.resolve(process.argv[2]||'/out');await fs.mkdir(output,{recursive:true});
await init({module_or_path:await fs.readFile('/app/node_modules/@rhwp/core/rhwp_bg.wasm')});
function chunk(name,data){const body=Buffer.concat([Buffer.from(name),data]);let crc=0xffffffff;for(const b of body){crc^=b;for(let i=0;i<8;i++)crc=(crc>>>1)^((crc&1)?0xedb88320:0);}const n=Buffer.alloc(4),c=Buffer.alloc(4);n.writeUInt32BE(data.length);c.writeUInt32BE((crc^0xffffffff)>>>0);return Buffer.concat([n,body,c]);}
const header=Buffer.alloc(13);header.writeUInt32BE(4,0);header.writeUInt32BE(4,4);header[8]=8;header[9]=2;const pixels=[];for(let y=0;y<4;y++){pixels.push(0);for(let x=0;x<4;x++)pixels.push(...(y<2?(x<2?[230,70,60]:[30,160,90]):(x<2?[40,100,220]:[240,190,40])));}
const png=Buffer.concat([Buffer.from('89504e470d0a1a0a','hex'),chunk('IHDR',header),chunk('IDAT',deflateSync(Buffer.from(pixels))),chunk('IEND',Buffer.alloc(0))]);
const sections=[{heading:'본문과 표',paragraphs:['한글 본문과 영문을 함께 저장합니다.'],table:[['항목','결과'],['한글','가나다라마바사'],['English','Sample 123']],images:[{data:png.toString('base64'),width_px:4,height_px:4,width_cm:3,caption:'그림 1. 문서 흐름에 배치한 컬러 이미지'}]},{heading:'표와 이미지 다음 절',paragraphs:['마지막 본문까지 빠짐없이 보존합니다.']}];
const report={};
for(const format of ['hwp','hwpx','docx']){
 const dir=path.join(output,format);await fs.mkdir(dir,{recursive:true});const result=await render({format,title:'SparkTalk 한글 문서 검증',sections},dir);assert.equal(result.files.length,2);await fs.copyFile(path.join(dir,'document.'+format),path.join(output,'sample.'+format));await fs.copyFile(path.join(dir,'document.pdf'),path.join(output,format+'-view.pdf'));
 if(format==='docx')continue;
 assert.equal(result.page_count,1);assert.match(result.text,/마지막 본문까지/);const doc=new HwpDocument(Buffer.from(result.files[0].data,'base64'));try{
  const svg=doc.renderPageSvg(0);assert.match(svg,/<image/);await fs.writeFile(path.join(output,format+'-page-1.svg'),svg);
  assert.notEqual(JSON.parse(doc.insertText(0,0,0,'수정 검증 ')).ok,false);await fs.writeFile(path.join(output,'edited.'+format),format==='hwp'?doc.exportHwp():doc.exportHwpx());
 }finally{doc.free();}
 const long=await render({format,title:'긴 표 검증',sections:[{paragraphs:['표 시작'],table:[['행','내용'],...Array.from({length:79},(_,i)=>[String(i+1),'한글 표 행 '+(i+1)])]},{heading:'종료',paragraphs:['최종 본문 79행 뒤']}]},dir);assert.ok(long.page_count>1);assert.match(long.text,/최종 본문 79행 뒤/);const longDoc=new HwpDocument(Buffer.from(long.files[0].data,'base64'));try{let all='';for(let i=0;i<longDoc.pageCount();i++)all+=longDoc.renderPageSvg(i);assert.ok(all.replace(/<[^>]*>/g,'').replace(/\s/g,'').includes('최종본문79행뒤'));}finally{longDoc.free();}
 report[format]={pages:result.page_count,long_pages:long.page_count,text_image_edit:'passed'};
}
const plain=await render({format:'hwp',title:'SparkTalk 본문',sections:[{paragraphs:['가나다라마바사 Sample 123']}]},output);await fs.rename(path.join(output,'document.hwp'),path.join(output,'text-only.hwp'));await fs.rm(path.join(output,'document.pdf'));
await fs.writeFile(path.join(output,'service-report.json'),JSON.stringify(report,null,2));console.log(report);
