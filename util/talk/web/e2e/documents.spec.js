import {test,expect} from '@playwright/test';
import fs from 'node:fs/promises';
import {createServer} from 'node:http';

for (const format of ['pdf','xlsx','hwp','hwpx']) test(`generated ${format} files persist and PDF preview opens in a modal`,async({page,request})=>{
 let calls=0;let sawTool=false;
 const hancom=['hwp','hwpx'].includes(format);
 const source=hancom?await fs.readFile(new URL(`../../internal/server/testdata/document.${format}`,import.meta.url)):null;
 const documents=createServer(async(req,res)=>{for await(const _ of req){};res.setHeader('Content-Type','application/json');res.end(JSON.stringify({...(hancom?{text:'한글 후속 참조 본문',page_count:1}:{}),files:[...(hancom?[{name:'document.'+format,mime:format==='hwp'?'application/x-hwp':'application/vnd.hancom.hwpx',data:source.toString('base64')}]:[]),...(format==='xlsx'?[{name:'document.xlsx',mime:'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',data:'UEsDBBQAAAAIAESxJ13HHBc8CgAAAAgAAAATAAAAW0NvbnRlbnRfVHlwZXNdLnhtbLMJqSxILda3AwBQSwMEFAAAAAgARLEnXc6emBMNAAAACwAAAA8AAAB4bC93b3JrYm9vay54bWyzKc8vyk7Kz8/WtwMAUEsBAhQDFAAAAAgARLEnXcccFzwKAAAACAAAABMAAAAAAAAAAAAAAIABAAAAAFtDb250ZW50X1R5cGVzXS54bWxQSwECFAMUAAAACABEsSddzp6YEw0AAAALAAAADwAAAAAAAAAAAAAAgAE7AAAAeGwvd29ya2Jvb2sueG1sUEsFBgAAAAACAAIAfgAAAHUAAAAAAA=='}]:[]),{name:'document.pdf',mime:'application/pdf',data:Buffer.from('%PDF-1.4\nTest transport fixture').toString('base64')}]}));});
 const model=createServer(async(req,res)=>{
  if(req.method!=='POST'){res.end(JSON.stringify({data:[{id:'test-model',context_length:65536}]}));return;}
  let raw='';for await(const b of req)raw+=b;const body=JSON.parse(raw);
  if(!body.stream){res.end(JSON.stringify({choices:[{message:{content:'Document test'},finish_reason:'stop'}]}));return;}
  calls++;sawTool ||=body.tools?.some(x=>x.function.name==='document_generate');
  const delta=calls===1?{tool_calls:[{index:0,id:'document-call',type:'function',function:{name:'document_generate',arguments:JSON.stringify({format,title:'보고서',filename:'report',sections:[{paragraphs:['본문']}]})}}]}:{content:'문서를 생성했습니다.'};
  res.setHeader('Content-Type','text/event-stream');res.end('data: '+JSON.stringify({choices:[{delta,finish_reason:calls===1?'tool_calls':'stop'}]})+'\n\ndata: [DONE]\n\n');
 });
 await Promise.all([new Promise(r=>documents.listen(0,'127.0.0.1',r)),new Promise(r=>model.listen(0,'127.0.0.1',r))]);
 const original=await(await request.get('/api/config')).json();let session;
 try{
  const cfg=structuredClone(original);cfg.model.endpoint=`http://127.0.0.1:${model.address().port}`;cfg.context.window_tokens=65536;cfg.extra.documents_enabled=true;cfg.extra.documents_endpoint=`http://127.0.0.1:${documents.address().port}`;
  expect((await request.put('/api/config',{data:cfg})).ok()).toBeTruthy();
  session=await(await request.post('/api/sessions',{data:{title:'Document test'}})).json();
  const response=await request.post('/api/chat',{data:{session_id:session.id,content:'PDF 파일을 만들어 줘.',model:'test-model',tools_enabled:false}});
  expect(await response.text()).toContain('report.'+format);expect(sawTool).toBeTruthy();
  const history=await(await request.get(`/api/sessions/${session.id}/messages`)).json();expect(history.at(-1).attachments).toHaveLength(format==='pdf'?1:2);expect(history.at(-1).status).toBe('completed');
  if(format!=='pdf'){const original=history.at(-1).attachments[0];expect(original.name).toBe('report.'+format);expect((await request.get(original.url)).ok()).toBeTruthy();}
  const file=history.at(-1).attachments.at(-1);expect((await request.get(file.url)).ok()).toBeTruthy();
  await page.goto('/');await page.reload();await page.getByRole('button',{name:'PDF 보기',exact:true}).click();
  await expect(page.getByRole('dialog')).toBeVisible();await expect(page.locator('iframe[title="PDF 보기"]')).toHaveAttribute('src',file.url);
  await page.keyboard.press('Escape');await expect(page.getByRole('dialog')).not.toBeVisible();
  await page.setViewportSize({width:390,height:760});const sidebarClose=page.getByRole('button',{name:'사이드바 닫기',exact:true}).first();if(await sidebarClose.isVisible())await sidebarClose.click();await page.getByRole('button',{name:'PDF 보기',exact:true}).click();expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBe(390);
  await page.getByRole('button',{name:'문서 미리보기 닫기',exact:true}).click();
 }finally{
  if(session)await request.delete(`/api/sessions/${session.id}`);await request.put('/api/config',{data:original});
  for(const s of [model,documents]){s.closeAllConnections();await new Promise(r=>s.close(r));}
 }
});
