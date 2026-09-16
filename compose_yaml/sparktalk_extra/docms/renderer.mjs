import {isExtended,validateBlocks,blocks,page,allImages,obj,color,str,validateMedia} from './blocks.mjs';
import {renderRichDocx} from './docx-rich.mjs';
import {renderPresentation,inspectPresentation} from './presentation.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import {Document, Packer, Paragraph, Table, TableRow, TableCell, HeadingLevel, ImageRun} from 'docx';
import pptxgen from 'pptxgenjs';
import {validateSheets, renderSpreadsheet} from './spreadsheet.mjs';
import {renderPDF} from './pdf.mjs';
import {FONT,COLOR,SLIDE,slideFont} from './layout.mjs';
import {validateImages,imageSize} from './document-images.mjs';
const mimes={docx:'application/vnd.openxmlformats-officedocument.wordprocessingml.document',pptx:'application/vnd.openxmlformats-officedocument.presentationml.presentation',hwp:'application/x-hwp',hwpx:'application/vnd.hancom.hwpx',xlsx:'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',pdf:'application/pdf'};
function text(value,max=4000){if(typeof value!=='string'||value.length>max||value.includes('\0'))throw Error('Invalid or oversized document text');return value;}
function keys(value,allowed){if(!value||typeof value!=="object"||Array.isArray(value)||Object.keys(value).some(k=>!allowed.includes(k)))throw Error("Unsupported document field");}
export function validate(input){
 keys(input,["format","title","filename","sections","slides","sheets","page","theme"]);
 if(!input || !Object.hasOwn(mimes,input.format))throw Error('format must be docx, pptx, xlsx, hwp, hwpx or pdf');
 page(input.page);if(input.theme){obj(input.theme,["font","accent","background","ratio","preset"]);if(input.theme.preset&&!["business","minimal","dark"].includes(input.theme.preset))throw Error("Invalid theme preset");if(input.theme.font)str(input.theme.font,80);if(input.theme.accent)color(input.theme.accent);if(input.theme.background)color(input.theme.background);if(input.theme.ratio&&!["wide","standard"].includes(input.theme.ratio))throw Error("Invalid slide ratio");}
 text(input.title,240);if(!input.title.trim())throw Error('title is required');
 if(input.format==='xlsx'){validateSheets(input.sheets);
 }else if(input.format==='pptx'){
  if(!Array.isArray(input.slides)||input.slides.length<1||input.slides.length>40)throw Error('Provide 1–40 slides');
  for(const s of input.slides){keys(s,["title","bullets","blocks","table","images","notes","layout"]);text(s.title,120);if(s.notes!==undefined)text(s.notes,8000);if(s.layout&&!['single','two-column'].includes(s.layout))throw Error("Invalid slide layout");if(s.blocks||s.table||s.images){if(s.blocks&&(s.bullets||s.table||s.images))throw Error("Use blocks or legacy slide content, not both");validateBlocks(blocks(s),input.format);continue;}if(!Array.isArray(s.bullets)||s.bullets.length>8)throw Error('At most 8 bullets per slide');for(const b of s.bullets)text(b,200);slideFont(s.bullets);if(s.bullets.join('').length>800)throw Error('Split long content into additional slides');}
 }else{
  if(!Array.isArray(input.sections)||input.sections.length<1||input.sections.length>80)throw Error('Provide 1–80 sections');
  for(const s of input.sections){keys(s,["heading","paragraphs","table","images","blocks","page"]);page(s.page);if(s.blocks){if(s.paragraphs||s.table||s.images||s.heading)throw Error("Use blocks or legacy section content, not both");validateBlocks(s.blocks,input.format);continue;}if(s.heading!==undefined)text(s.heading,240);if(!Array.isArray(s.paragraphs))throw Error('paragraphs must be an array');for(const p of s.paragraphs)text(p);if(s.table){if(!Array.isArray(s.table)||s.table.length<1||s.table.length>100)throw Error('Table needs 1–100 rows');const cols=s.table[0].length;if(cols<1||cols>8)throw Error('At most 8 columns');for(const r of s.table){if(!Array.isArray(r)||r.length!==cols)throw Error('Table rows must have equal columns');for(const c of r)text(c,500);}}}
 }
 if(['pptx','xlsx'].includes(input.format)&&input.sections?.some(s=>s.images?.length))throw Error('Section images are supported only in docx/pdf/hwp/hwpx');
 for(const s of input.sections||[])if(s.images!==undefined&&!Array.isArray(s.images))throw Error('images must be an array');
 validateImages([{images:allImages(input)}]);validateMedia(input);
 if(JSON.stringify(input,(key,value)=>key==='data'?'':value).length>160000)throw Error('Document content exceeds limit');
 return input;
}
export async function render(input,dir){
 validate(input);
 if(input.format==='xlsx')return renderSpreadsheet(input,dir);
 if(['hwp','hwpx'].includes(input.format)){const {renderHancom}=await import('./hancom.mjs');const output=await renderHancom(input,dir);const files=[output.file];try{files.push(await renderPDF(input,dir));return {files,text:output.text,page_count:output.page_count};}catch{return {files,text:output.text,page_count:output.page_count,warning:'PDF could not be generated; the original Hancom file is available.'};}}
 if(isExtended(input)&&['docx','pptx'].includes(input.format)){const file=input.format==='pptx'?await renderPresentation(input,dir):await renderRichDocx(input,dir);if(Buffer.byteLength(file.data,'base64')>24*1024*1024)throw Error('Document output exceeds limit');const {presentation,...attachment}=file;const files=[attachment];try{files.push(await renderPDF(input,dir));return {files,...(presentation?{presentation}: {})};}catch(e){return {files,...(presentation?{presentation}: {}),warning:'PDF could not be generated: '+e.message};}}
 const base='document';let original=input.format==='pptx'?'pptx':'docx';
 const source=path.join(dir,`${base}.${original}`);
 if(input.format==='pdf')return {files:[await renderPDF(input,dir)]};
 if(original==='docx'){
  const children=[new Paragraph({text:input.title,heading:HeadingLevel.TITLE})];
  for(const s of input.sections){
   if(s.heading)children.push(new Paragraph({text:s.heading,heading:HeadingLevel.HEADING_1}));
   for(const p of s.paragraphs)children.push(new Paragraph({text:p,spacing:{after:160}}));
   if(s.table)children.push(new Table({width:{size:100,type:'pct'},rows:s.table.map((row,i)=>new TableRow({tableHeader:i===0,children:row.map(cell=>new TableCell({children:[new Paragraph(cell)]}))}))}));
   for(const image of s.images||[]){const size=imageSize(image);children.push(new Paragraph({children:[new ImageRun({type:'png',data:Buffer.from(image.data,'base64'),transformation:{width:Math.round(size.width*96/72),height:Math.round(size.height*96/72)}})]}));if(image.caption)children.push(new Paragraph(image.caption));}
  }
  const doc=new Document({title:input.title,creator:'SparkTalk',styles:{default:{document:{run:{font:FONT,size:22}}}},sections:[{children}]});
  await fs.writeFile(source,await Packer.toBuffer(doc));
 }else{
  const deck=new pptxgen();deck.defineLayout({name:'SPARKTALK',width:SLIDE.width/72,height:SLIDE.height/72});deck.layout='SPARKTALK';deck.title=input.title;deck.author='SparkTalk';deck.theme={headFontFace:FONT,bodyFontFace:FONT,lang:'ko-KR'};
  for(const [i,s] of input.slides.entries()){
   const slide=deck.addSlide();slide.background={color:'FFFFFF'};
   slide.addShape(deck.ShapeType.rect,{x:0,y:0,w:0.16,h:7.5,fill:{color:COLOR.accent},line:{color:COLOR.accent}});
   slide.addText(s.title,{x:SLIDE.title.x/72,y:SLIDE.title.y/72,w:SLIDE.title.w/72,h:SLIDE.title.h/72,fontSize:SLIDE.title.font,bold:true,color:COLOR.title,breakLine:false});
   slide.addText(s.bullets.map(t=>({text:t,options:{bullet:true,breakLine:true}})),{x:SLIDE.body.x/72,y:SLIDE.body.y/72,w:SLIDE.body.w/72,h:SLIDE.body.h/72,fontSize:slideFont(s.bullets),paraSpaceAfter:12,fit:'shrink',valign:'top'});
   slide.addText(`${i+1} / ${input.slides.length}`,{x:SLIDE.footer.x/72,y:SLIDE.footer.y/72,w:SLIDE.footer.w/72,h:SLIDE.footer.h/72,fontSize:10,color:COLOR.muted,align:'right'});
  }
  await deck.writeFile({fileName:source});
 }
 const data=await fs.readFile(source);if(!data.length||data.length>24*1024*1024)throw Error('Invalid output size');
 const files=[{name:`document.${original}`,mime:mimes[original],data:data.toString('base64')}];
 const presentation=original==='pptx'?await inspectPresentation(data,input.slides.map((s,i)=>({items:[],sourceSlide:i+1})),input.slides.length):undefined;
 try{files.push(await renderPDF(input,dir));return {files,...(presentation?{presentation}:{})};}catch{return {files,...(presentation?{presentation}:{}),warning:'PDF could not be generated; the original file is available.'};}
}
