import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import init,{HwpDocument} from '@rhwp/core';
await init({module_or_path:await fs.readFile(new URL('./node_modules/@rhwp/core/rhwp_bg.wasm',import.meta.url))});
const root=path.resolve(process.argv[2]||'output');
for(const file of ['sample-standard.hwpx','edited-standard.hwpx']){
 const doc=new HwpDocument(await fs.readFile(path.join(root,file)));
 try{assert.equal(doc.pageCount(),1);const svg=doc.renderPageSvg(0);const text=svg.replace(/<[^>]*>/g,'').replace(/\s+/g,'');assert.ok(text.includes('Sample123'));assert.ok(text.includes('가나다라마바사'));assert.ok(svg.includes('<image'));if(file.startsWith('edited'))assert.ok(text.includes('수정검증'));console.log(file,'standard XML rendering passed; no rhwp origin metadata');}finally{doc.free();}
}
