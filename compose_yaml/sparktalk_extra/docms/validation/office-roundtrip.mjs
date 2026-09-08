import fs from 'node:fs/promises';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {createRequire} from 'node:module';
import {render} from '/app/renderer.mjs';
const require=createRequire('/app/package.json');const ExcelJS=require('exceljs');
process.env.XDG_CACHE_HOME='/tmp/office-cache';process.env.DCONF_PROFILE='/dev/null';
const root=process.argv[2]||'/tmp/office-review';await fs.mkdir(root,{recursive:true});
const report={title:'한글 문서 검증',sections:[{heading:'생성 결과',paragraphs:['같은 내용으로 Office 원본과 PDF를 생성합니다. English 123'],table:[['항목','설명'],...Array.from({length:80},(_,i)=>[String(i+1),'한글 표 내용'])]}]};
const samples=[{...report,format:'docx',stem:'report'},{...report,format:'pdf',stem:'report_pdf'},
 {format:'pptx',stem:'slides',title:'한글 발표자료',slides:[{title:'문서 생성',bullets:['한글 발표 내용 English 123','DOCX · PPTX · XLSX · PDF']},{title:'검증 결과',bullets:['두 번째 슬라이드','원본과 별도 PDF의 내용 일치 확인']}]},
 {format:'xlsx',stem:'sales',title:'매출 집계',sheets:[{name:'매출',columns:[{title:'날짜',format:'date'},{title:'항목'},{title:'금액',format:'currency'},{title:'세금',format:'currency'}],rows:[[{date:'2026-09-07'},'상품 A',100,{formula:'C2*0.1'}],[{date:'2026-09-08'},'상품 B',200,{formula:'C3*0.1'}],[null,'합계',{formula:'SUM(C2:C3)'},{formula:'SUM(D2:D3)'}]]},{name:'요약',columns:[{title:'항목'},{title:'결과'}],rows:[['총액',{formula:"'매출'!C4"}],['평균',{formula:"AVERAGE('매출'!C2:C3)"}],['달성',{formula:'OR(B2>200,FALSE)'}],['판정',{formula:'IF(B4,"충족","미달")'}],['문자 숫자',{formula:'IF(TRUE,"123","456")'}],['빈 문자열',{formula:'IF(TRUE,"","x")'}]]}]}];
await fs.writeFile(path.join(root,'inputs.json'),JSON.stringify(samples,null,2));
await fs.writeFile(path.join(root,'environment.json'),JSON.stringify({date:new Date().toISOString(),image:process.env.DOCUMENT_IMAGE_ID||'not recorded',endpoint:process.env.DOCUMENTS_ENDPOINT||'in-process',node:process.version,packages:JSON.parse(await fs.readFile('/app/package.json','utf8')).dependencies,libreoffice:execFileSync('libreoffice',['--version'],{encoding:'utf8'}).trim()},null,2));
for(const sample of samples){
 const {stem,...input}=sample;const dir=path.join(root,stem);await fs.mkdir(dir,{recursive:true});
 let result;
 if(process.env.DOCUMENTS_ENDPOINT){const response=await fetch(process.env.DOCUMENTS_ENDPOINT+'/v1/documents',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(input),signal:AbortSignal.timeout(100000)});result=await response.json();if(!response.ok)throw Error(JSON.stringify(result));}
 else result=await render(input,dir);
 if(result.warning)throw Error(result.warning);
 for(const file of result.files){const data=Buffer.from(file.data,'base64'),ext=path.extname(file.name);await fs.writeFile(path.join(dir,file.name),data);await fs.writeFile(path.join(root,stem+(ext==='.pdf'&&input.format!=='pdf'?'_view':'')+ext),data);}
 if(input.format==='pdf')continue;
 const ref=path.join(dir,'reference');await fs.mkdir(ref,{recursive:true});const source=path.join(dir,'document.'+input.format);
 execFileSync('xvfb-run',['-a','python3',new URL('./reopen-office.py',import.meta.url).pathname,source,ref],{stdio:'inherit',timeout:90000});
 console.log(input.format+': generated and independently reopened/saved');
}
execFileSync('python3',[new URL('./verify-office.py',import.meta.url).pathname,root],{stdio:'inherit',timeout:120000});
