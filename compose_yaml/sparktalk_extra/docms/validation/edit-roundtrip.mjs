import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import init,{HwpDocument} from '@rhwp/core';
const root=path.resolve(process.argv[2]||'output');
await init({module_or_path:await fs.readFile(new URL('./node_modules/@rhwp/core/rhwp_bg.wasm',import.meta.url))});
for(const ext of ['hwp','hwpx']){
 const doc=new HwpDocument(await fs.readFile(path.join(root,`sample.${ext}`)));
 try{
  const result=JSON.parse(doc.insertText(0,0,0,'수정 검증 / '));assert.equal(result.ok,true);
  const bytes=ext==='hwp'?doc.exportHwp():doc.exportHwpx();await fs.writeFile(path.join(root,`edited.${ext}`),bytes);
  const next=new HwpDocument(bytes);
  try{const svg=Array.from({length:next.pageCount()},(_,i)=>next.renderPageSvg(i)).join('');const text=svg.replace(/<[^>]*>/g,'').replace(/\s+/g,'');for(const expected of ['수정검증','가나다라마바사','Sample123'])assert.ok(text.includes(expected),`${ext}: missing ${expected}`);assert.ok(svg.includes('<image'));console.log(`${ext}: edit/save/reopen content and image passed (${next.pageCount()} pages)`);}finally{next.free();}
 }finally{doc.free();}
}
const plain=HwpDocument.createEmpty();try{plain.createBlankDocument();plain.insertText(0,0,0,'한글 본문 대조군 Sample 123');await fs.writeFile(path.join(root,'text-only.hwp'),plain.exportHwp());}finally{plain.free();}
