import {render} from './renderer.mjs';

export const TEXT_DOCUMENT_LIMITS = Object.freeze({characters:1000000,bytes:4<<20,paragraphs:20000,timeout_ms:300000});

// Whole-document jobs submit committed text through this route. Rich model tool
// arguments keep their own smaller limit; neither route silently drops content.
export function prepareTextDocument(input) {
 if(!input||typeof input!=='object'||Array.isArray(input)||Object.keys(input).some(k=>!['format','title','text'].includes(k)))throw Error('Unsupported text document field');
 if(!['docx','pdf','hwp','hwpx'].includes(input.format))throw Error('Text document format must be docx, pdf, hwp or hwpx');
 if(input.title!==undefined&&(typeof input.title!=='string'||!input.title.trim()||input.title.length>240||input.title.includes('\0')))throw Error('Invalid text document title');
 if(typeof input.text!=='string'||!input.text.trim()||input.text.includes('\0'))throw Error('Text document body is required');
 if(Buffer.byteLength(input.text,'utf8')>TEXT_DOCUMENT_LIMITS.bytes)throw Error('Text document exceeds 4 MiB');
 let count=0;for(const _ of input.text)if(++count>TEXT_DOCUMENT_LIMITS.characters)throw Error('Text document exceeds 1000000 characters');
 const paragraphs=[];
 for(const paragraph of input.text.split('\n\n')) {
  if(!paragraph)continue;
  paragraphs.push(paragraph);
  if(paragraphs.length>TEXT_DOCUMENT_LIMITS.paragraphs)throw Error('Text document exceeds 20000 paragraphs');
 }
 return {format:input.format,title:input.title??'',sections:[{paragraphs}]};
}

export function renderTextDocument(input,dir) {
 const normalized=prepareTextDocument(input);
 // JSON uses UTF-16 length; astral characters count twice, plus paragraph keys.
 // Layout engines paginate long paragraphs themselves. A transport limit must
 // not create new paragraphs in the middle of a sentence or dialogue.
 // A text export may already contain its title. Only add a heading when the
 // caller explicitly supplies one; ordinary rich document tools still require it.
 return render(normalized,dir,{allowEmptyTitle:true,contentLimit:2*TEXT_DOCUMENT_LIMITS.characters+6*TEXT_DOCUMENT_LIMITS.paragraphs+1024,paragraphLimit:2*TEXT_DOCUMENT_LIMITS.characters});
}
