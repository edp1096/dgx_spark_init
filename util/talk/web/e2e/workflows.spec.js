import {test,expect} from '@playwright/test';
import {createServer} from 'node:http';

test('procedure editor, ordered execution, persistent pause and resume',async({page,request})=>{
 const calls=[];let blocked=false;
 const backend=createServer(async(req,res)=>{
  if(req.method!=='POST'){res.setHeader('Content-Type','application/json');res.end(JSON.stringify({data:[{id:'test-model',context_length:65536}]}));return;}
  let raw='';for await(const b of req)raw+=b;const body=JSON.parse(raw);
  if(!body.stream){res.end(JSON.stringify({choices:[{message:{content:'Procedure test'},finish_reason:'stop'}]}));return;}
  calls.push(body);const text=JSON.stringify(body.messages);const stage=Number(text.match(/Current stage (\d)\/3/)?.[1]||0);
  let status='completed';if(stage===2&&!blocked){blocked=true;status='blocked';}
  const report={status,summary:`Stage ${stage} deliverable`,handoff:`stage ${stage} handoff`,evidence:[]};
  res.setHeader('Content-Type','text/event-stream');res.end('data: '+JSON.stringify({choices:[{delta:{tool_calls:[{index:0,id:'report',type:'function',function:{name:'workflow_report',arguments:JSON.stringify(report)}}]},finish_reason:'tool_calls'}]})+'\n\ndata: [DONE]\n\n');
 });await new Promise(resolve=>backend.listen(0,'127.0.0.1',resolve));
 const original=await(await request.get('/api/config')).json();let session;
 try{
  const cfg=structuredClone(original);cfg.model.endpoint=`http://127.0.0.1:${backend.address().port}`;cfg.context.window_tokens=65536;cfg.tools.skills_enabled=true;
  expect((await request.put('/api/config',{data:cfg})).ok()).toBeTruthy();
  session=await(await request.post('/api/sessions',{data:{title:'Procedure test'}})).json();
  await page.goto('/');await page.getByRole('button',{name:'▤ 라이브러리',exact:true}).click();await page.getByRole('button',{name:'작업 절차',exact:true}).click();
  await page.getByRole('button',{name:'document-production',exact:true}).click();await page.getByRole('button',{name:'복사해서 수정',exact:true}).click();
  await page.getByLabel('절차 이름',{exact:true}).fill('procedure-e2e');await page.getByLabel('절차 설명',{exact:true}).fill('문서 작성 실사용 검사');
  await page.getByRole('button',{name:'단계 2 위로',exact:true}).click();await expect(page.getByLabel('단계 1 이름',{exact:true})).toHaveValue('작성');
  await page.getByRole('button',{name:'단계 1 아래로',exact:true}).click();await expect(page.getByLabel('단계 1 이름',{exact:true})).toHaveValue('구성');
  await page.setViewportSize({width:390,height:760});expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBe(390);await page.screenshot({path:'/tmp/talk-workflow-editor-mobile.png'});
  await page.getByRole('button',{name:'저장',exact:true}).click();await expect(page.getByRole('button',{name:'procedure-e2e',exact:true})).toBeVisible();
  await page.setViewportSize({width:1280,height:800});await page.getByRole('button',{name:'대화로 돌아가기',exact:true}).click();
  await page.locator('.composer textarea').fill('문서 작성해 줘.');await page.getByRole('button',{name:'입력 도구 열기',exact:true}).click();await page.getByRole('button',{name:'스킬·작업 절차',exact:true}).click();
  await page.getByText('작업 절차 선택 · 실행 순서 미리보기',{exact:true}).click();await page.getByRole('button',{name:/^procedure-e2e/}).click();await page.getByRole('button',{name:'이 순서로 적용',exact:true}).click();
  await expect(page.locator('.composer textarea')).toHaveValue(/^@workflow:procedure-e2e/);await page.getByRole('button',{name:'메시지 전송',exact:true}).click();
  await expect.poll(async()=>{const runs=await(await request.get(`/api/sessions/${session.id}/workflows`)).json();return runs[0]?.status;}).toBe('paused');
  await page.reload();await page.locator('.workflow-runs > summary').click();await page.locator('.workflow-runs > details > summary').click();await expect(page.getByRole('button',{name:'이 작업 이어하기',exact:true})).toBeVisible();
  await page.getByRole('button',{name:'이 작업 이어하기',exact:true}).click();await page.getByRole('button',{name:'메시지 전송',exact:true}).click();
  await expect.poll(async()=>{const runs=await(await request.get(`/api/sessions/${session.id}/workflows`)).json();return runs[0]?.status;}).toBe('completed');
  expect(calls.length).toBe(4);expect(calls[2].messages.some(m=>String(m.content).includes('Stage 1 deliverable'))).toBeTruthy();
  await page.screenshot({path:'/tmp/talk-workflow-completed.png'});
  const runs=await(await request.get(`/api/sessions/${session.id}/workflows`)).json();expect(runs[0].steps[1].history).toHaveLength(1);expect(runs[0].skills).toBeUndefined();
 }finally{if(session)await request.delete(`/api/sessions/${session.id}`);await request.delete('/api/workflows/procedure-e2e');await request.put('/api/config',{data:original});backend.closeAllConnections();await new Promise(resolve=>backend.close(resolve));}
});
