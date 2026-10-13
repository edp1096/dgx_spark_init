import {test,expect} from '@playwright/test';
import {createServer} from 'node:http';
import {zipSync,unzipSync,strToU8,strFromU8} from 'fflate';

test('whole document background translation preserves offline illustrations',async({page,request})=>{
 let attachment,sourceId,calls=0;
 const backend=createServer(async(req,res)=>{
  if(req.method!=='POST'){res.end(JSON.stringify({data:[{id:'test-model',context_length:65536}]}));return;}
  let raw='';for await(const b of req)raw+=b;const body=JSON.parse(raw);
  if(!body.stream){res.end(JSON.stringify({choices:[{message:{content:'문서 번역'},finish_reason:'stop'}]}));return;}
  let delta,finish='tool_calls';
  if(body.response_format?.type==='json_schema'||body.tools?.some(t=>t.function.name==='document_result')){
   const input=JSON.parse(body.messages.at(-1).content.split('Reference data (JSON):\n')[1]);
   delta={tool_calls:[{index:0,id:'result',type:'function',function:{name:'document_result',arguments:JSON.stringify({entries:input.map(x=>({id:x.id,text:'번역 '+x.text}))})}}]};
  }else{
   calls++;
   if(calls===1)delta={tool_calls:[{index:0,id:'read',type:'function',function:{name:'attachment_read',arguments:JSON.stringify({attachment_id:attachment.id})}}]};
   else if(calls===2){sourceId=JSON.parse(body.messages.findLast(m=>m.role==='tool').content).source_id;delta={tool_calls:[{index:0,id:'task',type:'function',function:{name:'document_task',arguments:JSON.stringify({action:'start',kind:'translate',sources:[sourceId],instructions:'전체를 한국어로 번역'})}}]};}
   else{delta={content:'전권 번역 작업을 시작했습니다.'};finish='stop';}
  }
  res.setHeader('Content-Type','text/event-stream');res.end('data: '+JSON.stringify({choices:[{delta,finish_reason:finish}],usage:{prompt_tokens:100,completion_tokens:100,total_tokens:200},timings:{prompt_n:100,prompt_ms:200,predicted_n:100,predicted_ms:2000}})+'\n\ndata: [DONE]\n\n');
 });
 await new Promise(r=>backend.listen(0,'127.0.0.1',r));const original=await(await request.get('/api/config')).json();let session;
 try{
  const cfg=structuredClone(original);cfg.model.endpoint=`http://127.0.0.1:${backend.address().port}`;cfg.model.model_type='generic';cfg.context.window_tokens=65536;cfg.context.output_reserve=8192;cfg.tools.enabled=true;
  expect((await request.put('/api/config',{data:cfg})).ok()).toBeTruthy();session=await(await request.post('/api/sessions',{data:{title:'삽화 포함 번역'}})).json();
  const html='<html><body><h2 id="first">Chapter I</h2><p>'+('The rabbit ran into the garden. '.repeat(190))+'</p><figure><img src="images/rabbit.png" alt="Rabbit"><figcaption>A rabbit</figcaption></figure><h2 id="last">Chapter II</h2><p>FINAL_SENTENCE</p><a href="#first">Return</a></body></html>';
  const data=zipSync({'index.html':strToU8(html),'images/rabbit.png':Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==','base64')});
  attachment=await(await request.post('/api/files',{multipart:{file:{name:'book.zip',mimeType:'application/zip',buffer:Buffer.from(data)}}})).json();
  const response=await request.post('/api/chat',{data:{session_id:session.id,content:'이 책 전체를 한국어로 번역하고 삽화를 유지해라.',model:'test-model',tools_enabled:true,attachments:[attachment]}});expect(await response.text()).toContain('번역 작업을 시작했습니다.');expect(sourceId).toBeTruthy();
  let run;await expect.poll(async()=>{const runs=await(await request.get(`/api/sessions/${session.id}/workflows`)).json();run=runs.find(x=>x.document);return run?.status;}).toBe('completed');expect(run.document.completed).toBe(run.document.total);expect(run.document.total).toBeGreaterThan(1);
  await page.goto('/');await expect(page.locator('.document-task .task-status')).toHaveText('완료');await page.reload();await expect(page.locator('.document-task .task-status')).toHaveText('완료');
  expect(run.document.last_performance.pp).toBe(500);expect(run.document.last_performance.tg).toBe(50);
  await expect(page.locator('.document-performance-last')).toContainText('최근 호출');
  await expect(page.locator('.document-performance-last')).toContainText('500 tok/s');
  await expect(page.locator('.document-performance-last')).toContainText('50 tok/s');
  await expect(page.locator('.document-performance-current')).toHaveCount(0);
  const zip=await request.get(`/api/document-tasks/${run.id}/output.zip?session_id=${session.id}`);expect(zip.ok()).toBeTruthy();const files=unzipSync(new Uint8Array(await zip.body()));const result=strFromU8(files['index.html']);expect(result).toContain('번역 FINAL_SENTENCE');expect(result).toContain('<figcaption>');expect(result).toContain('href="#d1-first"');const image=result.match(/src="(images\/[^"]+)"/)[1];expect(files[image]?.length).toBeGreaterThan(0);
  await page.locator('.task-details > summary').click();await page.getByRole('button',{name:'구간 결과',exact:true}).click();await expect(page.locator('.document-task')).toContainText('구간 1');
  const reviewed=structuredClone(run);reviewed.status='needs_review';reviewed.document.warnings=['glossary spelling requires review: Holmes → 홈즈'];
  await page.route(`**/api/sessions/${session.id}/workflows`,route=>route.fulfill({json:[reviewed]}));
  await page.reload();const card=page.locator('.document-task');
  await expect(card.locator('.task-heading strong')).toHaveText('문서 번역');await expect(card.locator('.task-status')).toHaveText('완료 · 확인 필요');
  await expect(page.locator('.workflow-runs > summary')).toContainText('완료 · 확인 필요');
  await expect(card.locator('.document-timing')).toContainText('완료');
  await expect(card.getByRole('button',{name:'재개',exact:true})).toHaveCount(0);
  await expect(card.getByRole('button',{name:'취소',exact:true})).toHaveCount(0);
  await expect(card.getByRole('link',{name:/^결과 받기/})).toBeVisible();
  await expect(card).toContainText('확인할 항목 1개');
  reviewed.document.image_total=1;reviewed.document.images_ready=0;reviewed.document.warnings.push('unexpectedly short translation: p1');
  await page.reload();await card.locator('.task-warnings > summary').click();await expect(card.getByRole('button',{name:'삽화 다시 받기',exact:true})).toBeVisible();
  await expect(card.getByRole('button',{name:'누락 의심 구간 재번역',exact:true})).toBeVisible();
  await expect(card.getByRole('button',{name:'재개',exact:true})).toHaveCount(0);
  reviewed.document.pipeline='reviewed_v1';reviewed.document.terminal=true;reviewed.document.quality_status='needs_review';reviewed.document.reviewed=2;reviewed.document.review_total=3;reviewed.document.unresolved=1;reviewed.document.finished_at='2026-10-13T00:01:02Z';reviewed.document.publication={files:[{name:'output.zip'}]};
  await page.reload();await expect(card).toContainText('검수 2/3문단');await expect(card.getByRole('button',{name:'새 개정판으로 재검수'})).toBeVisible();await expect(card.getByRole('button',{name:'재개',exact:true})).toHaveCount(0);
  reviewed.document.terminal=false;reviewed.document.publication=null;reviewed.status='conversion_failed';reviewed.document.phase='convert';
  await page.reload();await expect(card.getByRole('button',{name:'파일 변환 재시도'})).toBeVisible();await expect(card.getByRole('link',{name:/^결과 받기/})).toHaveCount(0);
  reviewed.document.draft_publication={files:[{name:'output.zip'},{name:'output.txt'}]};
  await page.reload();await expect(card.getByRole('link',{name:/^본문 결과 받기/})).toHaveAttribute('href',`/api/document-tasks/${run.id}/draft/output.zip?session_id=${session.id}`);await expect(card.getByRole('button',{name:'파일 변환 재시도'})).toBeVisible();
  delete reviewed.document.draft_publication;
  delete reviewed.document.pipeline;delete reviewed.document.terminal;delete reviewed.document.publication;reviewed.status='needs_review';
  reviewed.current=0;reviewed.document.phase='process';reviewed.document.completed=0;
  await page.reload();await expect(card.locator('.task-status')).toHaveText('검토 필요');
  await expect(card.getByRole('button',{name:'재개',exact:true})).toBeVisible();
  await expect(card.getByRole('button',{name:'취소',exact:true})).toBeVisible();
  const other=await(await request.post('/api/sessions',{data:{title:'격리'}})).json();expect((await request.get(`/api/document-tasks/${run.id}/output.zip?session_id=${other.id}`)).status()).toBe(404);await request.delete(`/api/sessions/${other.id}`);
  const paused=structuredClone(run);paused.status='paused';paused.current=0;paused.document.max_calls=paused.document.calls;paused.document.error='task execution budget reached';
  let renewed=false;
  await page.route(`**/api/sessions/${session.id}/workflows`,route=>route.fulfill({json:[paused]}));
  await page.route(`**/api/document-tasks/${run.id}/resume?*`,route=>{renewed=route.request().postDataJSON().extend_budget===true;paused.status='running';return route.fulfill({json:{status:'running'}});});
  await page.reload();await page.getByRole('button',{name:'한도 연장 후 재개',exact:true}).click();expect(renewed).toBeTruthy();
  paused.document.performance={live:true,tg:46.5,tg_estimated:true,ttft:12,ttft_estimated:true,calls:1};
  await page.reload();await expect(page.locator('.document-performance-current')).toContainText('≈ 46.5 tok/s');
  await expect(page.locator('.document-performance-current .metric').first()).toContainText('—');
  await expect(page.locator('.document-performance-last')).toContainText('직전 호출');
  await expect(page.locator('.document-performance-last')).toContainText('500 tok/s');
 }finally{if(session)await request.delete(`/api/sessions/${session.id}`);await request.put('/api/config',{data:original});backend.closeAllConnections();await new Promise(r=>backend.close(r));}
});
