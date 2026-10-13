import test from 'node:test';
import assert from 'node:assert/strict';
import {prepareTextDocument,renderTextDocument,TEXT_DOCUMENT_LIMITS} from './text-document.mjs';
import {validate} from './renderer.mjs';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import JSZip from 'jszip';

test('whole-book text preserves every character beyond the rich tool limit',()=>{
 const text='FIRST_MARKER\n\n'+('한글 본문과 대화 “빠뜨리지 마세요.” '.repeat(12000))+'\n\nLAST_MARKER';
 const input=prepareTextDocument({format:'docx',title:'전권',text});
 assert.equal(input.sections[0].paragraphs.join(''),text.replaceAll('\n\n',''));
 assert.throws(()=>validate(input),/Invalid or oversized document text/);
 assert.equal(input.sections[0].paragraphs.length,3);
 assert.equal(input.sections[0].paragraphs[1],text.split('\n\n')[1]);
});

test('long text conversion remains bounded and cannot carry rich tool objects',()=>{
 const input={format:'pdf',title:'전권',text:'본문'};
 assert.throws(()=>prepareTextDocument({...input,text:'x'.repeat(TEXT_DOCUMENT_LIMITS.characters+1)}),/1000000 characters/);
 assert.throws(()=>prepareTextDocument({...input,text:'x\n\n'.repeat(TEXT_DOCUMENT_LIMITS.paragraphs+1)}),/20000 paragraphs/);
 assert.throws(()=>prepareTextDocument({...input,sections:[{paragraphs:['bypass']}]}),/Unsupported/);
 assert.throws(()=>prepareTextDocument({...input,format:'pptx'}),/format/);
 assert.throws(()=>prepareTextDocument({...input,text:'\0'}),/body/);
 for(const title of ['',null,42,'\0'])assert.throws(()=>prepareTextDocument({...input,title}),/title/);
 const untitled=prepareTextDocument({format:'pdf',text:'제목 없는 본문'});
 assert.throws(()=>validate(untitled),/title is required/,'rich tools must still require an explicit title');
 const text='🐧'.repeat(4001),parts=prepareTextDocument({...input,text}).sections[0].paragraphs;
 assert.equal(parts.join(''),text);assert.equal(parts.length,1);
});

for(const format of ['pdf','docx','hwp','hwpx'])test(`committed text exports to ${format} without a generated heading`,async()=>{
 const dir=await fs.mkdtemp(path.join(os.tmpdir(),'doc-no-heading-'));
 const text='인간실격\n\n다자이 오사무\n\n서문\n\n첫 문단.\n\n［＃改頁］\n\n［＃페이지 나누기］\n\n문서 처리 결과라는 표현도 본문에 있으면 보존한다.';
 try {
  const result=await renderTextDocument({format,text},dir);
  assert.equal(result.warning,undefined);
  assert.ok(result.files.some(f=>f.name==='document.pdf'));
  if(format==='docx'){
   const file=result.files.find(f=>f.name==='document.docx');
   const zip=await JSZip.loadAsync(Buffer.from(file.data,'base64'));
   const xml=await zip.file('word/document.xml').async('string');
   const paragraphs=[...xml.matchAll(/<w:p[ >][\s\S]*?<\/w:p>/g)].map(p=>[...p[0].matchAll(/<w:t[^>]*>([^<]*)<\/w:t>/g)].map(m=>m[1]).join(''));
   assert.deepEqual(paragraphs,text.split('\n\n'));
  }else if(format==='hwp'||format==='hwpx'){
   assert.equal(result.text.replaceAll(/\s/g,''),text.replaceAll(/\s/g,''));
  }
 }finally{await fs.rm(dir,{recursive:true,force:true});}
});

test('long paragraph stays one DOCX paragraph and authored newlines become real breaks',async()=>{
 const dir=await fs.mkdtemp(path.join(os.tmpdir(),'doc-lines-'));
 try {
  const paragraph='문장이 중간에서 잘리면 안 됩니다. '.repeat(350)+'\n“대사 첫째 줄”\n“대사 둘째 줄”';
  const result=await renderTextDocument({format:'docx',title:'줄바꿈 검증',text:paragraph+'\n\n마지막 문단.'},dir);
  const file=result.files.find(f=>f.name==='document.docx');
  const zip=await JSZip.loadAsync(Buffer.from(file.data,'base64'));
  const xml=await zip.file('word/document.xml').async('string');
  assert.equal((xml.match(/<w:p>/g)||[]).length,3,'transport chunks became paragraphs');
  assert.equal((xml.match(/<w:br\b/g)||[]).length,2,'authored line breaks lost');
  assert.ok(xml.includes('대사 첫째 줄')&&xml.includes('대사 둘째 줄'));
 } finally {await fs.rm(dir,{recursive:true,force:true});}
});
