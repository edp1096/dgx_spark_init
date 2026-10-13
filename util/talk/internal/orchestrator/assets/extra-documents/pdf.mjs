import {isExtended} from './blocks.mjs';
import {richPDFDefinition,spreadsheetPDFDefinition} from './pdf-rich.mjs';
import pdfmake from 'pdfmake';
import {openSync as openFont} from 'fontkit';
import fs from 'node:fs/promises';
import path from 'node:path';
import {FONT,COLOR as HEX,SLIDE,slideFont} from './layout.mjs';
import {imageSize} from './document-images.mjs';
const COLOR=Object.fromEntries(Object.entries(HEX).map(([key,value])=>[key,'#'+value]));
const fontDir='/usr/share/fonts/opentype/noto/';
const fallbackDir='/usr/share/fonts/truetype/noto/';
const normal=[fontDir+'NotoSansCJK-Regular.ttc','NotoSansCJKkr-Regular'];
const bold=[fontDir+'NotoSansCJK-Bold.ttc','NotoSansCJKkr-Bold'];
pdfmake.setUrlAccessPolicy(()=>false);
pdfmake.addFonts({[FONT]:{normal,bold,italics:normal,bolditalics:bold}});
for(const family of ['NotoSans','NotoSansThai','NotoSansArabic'])pdfmake.addFonts({[family]:{normal:fallbackDir+family+'-Regular.ttf',bold:fallbackDir+family+'-Bold.ttf',italics:fallbackDir+family+'-Regular.ttf',bolditalics:fallbackDir+family+'-Bold.ttf'}});
pdfmake.setLocalAccessPolicy(filename=>filename.startsWith(fontDir)||filename.startsWith(fallbackDir));
const fontCoverage=new Map(),checkedGlyphs=new Set();
function fallbackFont(char){return /\p{Script_Extensions=Arabic}/u.test(char)?'NotoSansArabic':/\p{Script=Thai}/u.test(char)?'NotoSansThai':/[\uAB30-\uAB6F]/u.test(char)?'NotoSans':undefined;}
function verifyPDFGlyphs(text){
 for(const char of text){
  if(/[\p{White_Space}\p{Cc}\p{Cf}\p{Variation_Selector}]/u.test(char))continue;
  const family=fallbackFont(char)||FONT,key=family+char;
  if(checkedGlyphs.has(key))continue;
  let font=fontCoverage.get(family);
  if(!font){font=family===FONT?openFont(normal[0],normal[1]):openFont(fallbackDir+family+'-Regular.ttf');fontCoverage.set(family,font);}
  if(!font.hasGlyphForCodePoint(char.codePointAt(0)))throw Error('PDF font does not support U+'+char.codePointAt(0).toString(16).toUpperCase()+'. The saved text is unchanged; use HTML or another document format.');
  if(checkedGlyphs.size<65536)checkedGlyphs.add(key);
 }
}
function fallbackText(text){
 verifyPDFGlyphs(text);
 if(!/[\p{Script=Thai}\p{Script_Extensions=Arabic}\uAB30-\uAB6F]/u.test(text))return text;
 const runs=[];let previous,previousNumeric=false;
 for(const char of text){
  const font=fallbackFont(char),numeric=/[\u0660-\u0669\u06F0-\u06F9]/u.test(char);
  // Arabic shaping reverses a glyph run. Decimal digits retain their logical
  // left-to-right order even when their surrounding script is right-to-left.
  if(!previous||previous.font!==font||numeric||previousNumeric){previous={text:'',...(font?{font}:{})};runs.push(previous);}
  previous.text+=char;previousNumeric=numeric;
 }
 return runs;
}
function applyFallback(node){
 if(!node||typeof node!=='object')return;
 if(Array.isArray(node)){for(const child of node)applyFallback(child);return;}
 for(const [key,value] of Object.entries(node)){if(key==='text'&&typeof value==='string')node[key]=fallbackText(value);else applyFallback(value);}
}
function table(rows,widths){return {table:{headerRows:1,widths,body:rows.map((r,i)=>r.map(value=>({text:String(value??''),...(i===0?{bold:true,color:'#FFFFFF',fillColor:COLOR.accent}:{})})))},layout:{hLineWidth:()=>0.4,vLineWidth:()=>0,hLineColor:()=> '#D8DFE5',paddingLeft:()=>5,paddingRight:()=>5,paddingTop:()=>5,paddingBottom:()=>5},margin:[0,4,0,12]};}
export async function renderPDF(input,dir,workbook){
 const def={info:{title:input.title,author:'SparkTalk'},pageSize:'A4',pageMargins:[42,42,42,42],defaultStyle:{font:FONT,fontSize:11,color:'#202830'},content:[],footer:(page,pages)=>({text:`${page} / ${pages}`,alignment:'right',fontSize:9,color:COLOR.muted,margin:[42,12,42,0]})};
 if(input.format==='xlsx'&&input.sheets.some(s=>s.cells||s.merges||s.images||s.charts||s.pivots||s.shapes||s.hidden||s.row_options||s.column_options)){spreadsheetPDFDefinition(input,def,workbook);
 }else if(isExtended(input)){richPDFDefinition(input,def);
 }else if(input.format==='pptx'){
  def.pageSize={width:SLIDE.width,height:SLIDE.height};def.pageMargins=[0,0,0,0];def.footer=null;
  def.background=()=>({canvas:[{type:'rect',x:0,y:0,w:11.52,h:SLIDE.height,color:COLOR.accent}]});
  for(const [i,s] of input.slides.entries())def.content.push({pageBreak:i?'before':undefined,stack:[
   {absolutePosition:{x:SLIDE.title.x,y:SLIDE.title.y},columns:[{width:SLIDE.title.w,text:s.title,fontSize:SLIDE.title.font,bold:true,color:COLOR.title}]},
   {absolutePosition:{x:SLIDE.body.x,y:SLIDE.body.y},columns:[{width:SLIDE.body.w,stack:s.bullets.map(t=>({text:'• '+t,margin:[0,0,0,12]})),fontSize:slideFont(s.bullets)}]},
   {absolutePosition:{x:SLIDE.footer.x,y:SLIDE.footer.y},columns:[{width:SLIDE.footer.w,text:`${i+1} / ${input.slides.length}`,fontSize:10,color:COLOR.muted,alignment:'right'}]}
  ]});
 }else if(input.format==='xlsx'){
  let first=true;
  for(const s of input.sheets){const ws=workbook.getWorksheet(s.name);const landscape=s.orientation==='landscape'||(!s.orientation&&s.columns.length>5);
   // Wide sheets are divided into column bands rather than shrinking text to illegibility.
   for(let start=0;start<s.columns.length;start+=8){const indices=Array.from({length:Math.min(8,s.columns.length-start)},(_,i)=>start+i);if(start)indices.unshift(0);
    def.content.push({text:s.name+(s.columns.length>8?` (${start+1}–${Math.min(start+8,s.columns.length)})`:''),fontSize:18,bold:true,color:COLOR.title,margin:[0,0,0,12],pageBreak:first?undefined:'before',pageOrientation:landscape?'landscape':'portrait'});if(first)def.pageOrientation=landscape?'landscape':'portrait';first=false;
    const rows=[indices.map(c=>s.columns[c].title),...s.rows.map((row,r)=>indices.map(c=>displayCell(ws.getCell(r+2,c+1),s.columns[c].format)))];
    def.content.push(table(rows,indices.map(()=>'*')));
   }
  }
 }else{
  if(input.title)def.content.push({text:input.title,fontSize:24,bold:true,color:COLOR.title,margin:[0,0,0,18]});
  for(const s of input.sections){if(s.heading)def.content.push({text:s.heading,fontSize:16,bold:true,color:COLOR.title,margin:[0,12,0,8],headlineLevel:1});for(const p of s.paragraphs)def.content.push({text:p,margin:[0,0,0,8]});if(s.table)def.content.push(table(s.table,s.table[0].map(()=>'*')));for(const image of s.images||[]){const size=imageSize(image);def.content.push({image:'data:image/png;base64,'+image.data,width:size.width,height:size.height,margin:[0,8,0,4]});if(image.caption)def.content.push({text:image.caption,fontSize:10,margin:[0,0,0,8]});}}
 }
 applyFallback(def);
 const data=await pdfmake.createPdf(def).getBuffer();if(data.length>24*1024*1024||!data.subarray(0,5).equals(Buffer.from('%PDF-')))throw Error('Invalid PDF output');await fs.writeFile(path.join(dir,'document.pdf'),data);
 return {name:'document.pdf',mime:'application/pdf',data:data.toString('base64')};
}
function displayCell(cell,format){let value=cell.formula?cell.result:cell.value;if(value===null||value===undefined)return '';if(value instanceof Date)return value.toISOString().slice(0,10);if(typeof value==='boolean')return value?'TRUE':'FALSE';if(typeof value==='number'){if(format==='percent')return new Intl.NumberFormat('en-US',{style:'percent',minimumFractionDigits:2,maximumFractionDigits:2}).format(value);if(['number','currency','integer'].includes(format))return new Intl.NumberFormat('en-US',{minimumFractionDigits:format==='integer'?0:2,maximumFractionDigits:format==='integer'?0:2}).format(value);}return String(value);}
