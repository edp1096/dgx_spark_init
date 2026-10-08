import {test,expect} from '@playwright/test';
import {createServer} from 'node:http';

test('truncated code is marked incomplete after reload and retained for continuation',async({page,request})=>{
 const calls=[];
 const backend=createServer(async(req,res)=>{
  if(req.method!=='POST'){res.setHeader('Content-Type','application/json');res.end(JSON.stringify({data:[{id:'test-model',context_length:65536}]}));return;}
  let raw='';for await(const part of req)raw+=part;
  const body=JSON.parse(raw);
  if(!body.stream){res.end(JSON.stringify({choices:[{message:{content:'Code test'},finish_reason:'stop'}]}));return;}
  calls.push(body);
  res.setHeader('Content-Type','text/event-stream');
  res.end('data: '+JSON.stringify({choices:[{delta:{content:calls.length<=3?'```go\nfunc main() {\n':'continued code'},finish_reason:calls.length<=3?'length':'stop'}],usage:{prompt_tokens:40,completion_tokens:512,total_tokens:552}})+'\n\ndata: [DONE]\n\n');
 });
 await new Promise(resolve=>backend.listen(0,'127.0.0.1',resolve));
 const original=await(await request.get('/api/config')).json();
 let session;
 try{
  const cfg=structuredClone(original);cfg.model.endpoint=`http://127.0.0.1:${backend.address().port}`;cfg.context.window_tokens=65536;cfg.context.output_reserve=512;
  expect((await request.put('/api/config',{data:cfg})).ok()).toBeTruthy();
  session=await(await request.post('/api/sessions',{data:{title:'Code limit test'}})).json();
  const response=await request.post('/api/chat',{data:{session_id:session.id,content:'write code',model:'test-model',tools_enabled:false}});
  expect(await response.text()).toContain('finish_reason=length');
  const messages=await(await request.get(`/api/sessions/${session.id}/messages`)).json();
  expect(messages.at(-1).status).toBe('failed');expect(messages.at(-1).content).toContain('func main() {');
  await page.goto('/');await expect(page.getByText('불완전한 답변',{exact:true})).toBeVisible();
  await page.reload();await expect(page.getByText(/출력 토큰 한도에 도달해/).first()).toBeVisible();
  await request.post('/api/chat',{data:{session_id:session.id,content:'continue',model:'test-model',tools_enabled:false}});
  expect(calls.length).toBe(4);
  expect(calls[3].messages.some(m=>m.role==='assistant'&&String(m.content).includes('func main() {'))).toBeTruthy();
  expect(calls[0].max_completion_tokens).toBe(512);
 }finally{
  if(session)await request.delete(`/api/sessions/${session.id}`);
  await request.put('/api/config',{data:original});
  backend.closeAllConnections();await new Promise(resolve=>backend.close(resolve));
 }
});
