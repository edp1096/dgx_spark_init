import {isExtended} from './blocks.mjs';
import {populateHancom} from './hancom-rich.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import JSZip from 'jszip';
import init,{HwpDocument} from '@rhwp/core';
import {imageSize} from './document-images.mjs';
let initialized;
function ready(){return initialized??=fs.readFile(new URL('./node_modules/@rhwp/core/rhwp_bg.wasm',import.meta.url)).then(bytes=>init({module_or_path:bytes}));}
function ok(raw){const value=JSON.parse(raw);if(value.ok===false)throw Error(value.error||value.message||'Hancom document operation failed');return value;}
export async function renderHancom(input,dir){
 await ready();const doc=HwpDocument.createEmpty();
 try{
  doc.createBlankDocument();
  if(isExtended(input)){populateHancom(doc,input);}else{
  let forceNewParagraph=false;
  function paragraph(text,size=1100,bold=false){
   let index=doc.getParagraphCount(0)-1;
   if(forceNewParagraph||doc.getParagraphLength(0,index)>0){index++;ok(doc.insertParagraph(0,index));}
   forceNewParagraph=false;
   if(text){ok(doc.insertText(0,index,0,text));ok(doc.applyCharFormat(0,index,0,doc.getParagraphLength(0,index),JSON.stringify({fontSize:size,bold})));}
   return index;
  }
  paragraph(input.title,2000,true);
  for(const section of input.sections){
   if(section.heading)paragraph(section.heading,1500,true);
   for(const text of section.paragraphs)paragraph(text);
   if(section.table){
    const anchor=doc.getParagraphCount(0)-1;
    const table=ok(doc.createTable(0,anchor,doc.getParagraphLength(0,anchor),section.table.length,section.table[0].length));
    ok(doc.setTableProperties(0,table.paraIdx,table.controlIdx,JSON.stringify({pageBreak:2,repeatHeader:true,treatAsChar:false,textWrap:'TopAndBottom',vertRelTo:'Para',horzRelTo:'Para'})));
    for(const [r,row] of section.table.entries())for(const [c,text] of row.entries()){
     const index=r*row.length+c;ok(doc.insertTextInCell(0,table.paraIdx,table.controlIdx,index,0,0,text));
     if(text)ok(doc.applyCharFormatInCell(0,table.paraIdx,table.controlIdx,index,0,0,text.length,JSON.stringify({fontSize:1100,bold:r===0})));
    }
   }
   for(const image of section.images||[]){
    // Keep a text run in the image paragraph so HWPX readers compose its inline object.
    const para=paragraph(' ');const size=imageSize(image);
    const picture=ok(doc.insertPicture(0,para,0,'',Buffer.from(image.data,'base64'),Math.round(size.width*100),Math.round(size.height*100),image.width_px,image.height_px,'png',''));
    ok(doc.setPictureProperties(0,picture.paraIdx,picture.controlIdx,JSON.stringify({treatAsChar:true,vertRelTo:'Para',horzRelTo:'Para',vertOffset:0,horzOffset:0})));
    forceNewParagraph=true;
    if(image.caption)paragraph(image.caption,1000,false);
   }
  }
  }
  let bytes;
  if(input.page?.page_numbers){
   // The engine's header/footer field helper writes a bare 0x15 marker to HWP.
   // Use its standard HWPX pageNum parser instead, yielding a real pgnp control.
   const pageZip=await JSZip.loadAsync(doc.exportHwpx());
   for(const name of Object.keys(pageZip.files))if(name.startsWith('META-INF/rhwp'))pageZip.remove(name);
   let xml=await pageZip.file('Contents/section0.xml').async('string');let depth=0,position=-1;
   for(const match of xml.matchAll(/<(\/?)hp:p\b[^>]*>/g)){
    if(match[1]){if(--depth===0){position=match.index;break;}}
    else if(!match[0].endsWith('/>'))depth++;
   }
   if(position<0)throw Error('No paragraph for page numbering');
   xml=xml.slice(0,position)+'<hp:run charPrIDRef="0"><hp:ctrl><hp:pageNum pos="BOTTOM_CENTER" formatType="DIGIT" sideChar=""/></hp:ctrl></hp:run>'+xml.slice(position);
   pageZip.file('Contents/section0.xml',xml.replace(/<hp:linesegarray>[\s\S]*?<\/hp:linesegarray>/g,''));
   const numbered=new HwpDocument(await pageZip.generateAsync({type:'uint8array'}));
   try{bytes=input.format==='hwp'?numbered.exportHwp():numbered.exportHwpx();}finally{numbered.free();}
   if(input.format==='hwp'){
    // HWPX has no OLE summary stream; preserve it from the original HWP export.
    const {default:CFB}=await import('cfb');
    const original=CFB.read(Buffer.from(doc.exportHwp()),{type:'buffer'});
    const summary=CFB.find(original,'\u0005HwpSummaryInformation');
    if(summary){const compound=CFB.read(Buffer.from(bytes),{type:'buffer'});CFB.utils.cfb_add(compound,'\u0005HwpSummaryInformation',summary.content);bytes=CFB.write(compound,{type:'buffer'});}
   }
  }else bytes=input.format==='hwp'?doc.exportHwp():doc.exportHwpx();
  // Generated HWPX mixes template line caches with newly composed paragraphs.
  // Omit optional line-position caches so readers lay out every paragraph together.
  // Remove the rhwp origin snapshot too, keeping the standard XML authoritative.
  if(input.format==='hwpx'){
   const zip=await JSZip.loadAsync(bytes);
   for(const name of Object.keys(zip.files)){
    if(name.startsWith('META-INF/rhwp'))zip.remove(name);
    if(/^Contents\/section\d+\.xml$/.test(name)){
     const xml=await zip.file(name).async('string');
     zip.file(name,xml.replace(/<hp:linesegarray>[\s\S]*?<\/hp:linesegarray>/g,''));
    }
   }
   zip.file('mimetype',await zip.file('mimetype').async('string'),{compression:'STORE'});
   bytes=await zip.generateAsync({type:'uint8array',compression:'DEFLATE'});
  }
  if(!bytes.length||bytes.length>24*1024*1024)throw Error('Invalid Hancom document size');
  const restored=new HwpDocument(bytes);
  try{
   const text=JSON.parse(restored.getTextFileUnicode());if(!text.trim())throw Error('Exported document has no readable text');
   await fs.writeFile(path.join(dir,'document.'+input.format),bytes);
   return {file:{name:'document.'+input.format,mime:input.format==='hwp'?'application/x-hwp':'application/vnd.hancom.hwpx',data:Buffer.from(bytes).toString('base64')},text,page_count:restored.pageCount()};
  }finally{restored.free();}
 }finally{doc.free();}
}
