import {test,expect} from '@playwright/test';
import {createServer} from 'node:http';
test('speaker transcripts persist and display readable labels',async({page,request})=>{
 let received='';let diarCalls=0;
 const backend=createServer(async(req,res)=>{
  let raw='';for await(const b of req)raw+=b;
  if(req.url.startsWith('/v1/audio/extract')){res.end('wav');return;}
  res.setHeader('Content-Type','application/json');
  if(req.url==='/v1/audio/diarizations'){diarCalls++;res.end(JSON.stringify({segments:[{start:0,end:1,speaker:4},{start:1,end:2,speaker:7}]}));return;}
  if(req.url==='/v1/audio/transcriptions'){res.end(JSON.stringify({text:'안녕하세요 반갑습니다',words:[{word:'안녕하세요',start:0,end:1},{word:'반갑습니다',start:1,end:2}]}));return;}
  if(req.method!=='POST'){res.end(JSON.stringify({data:[{id:'test-model'}]}));return;}
  const body=JSON.parse(raw);if(!body.stream){res.end(JSON.stringify({choices:[{message:{content:'음성 테스트'}}]}));return;}
  received=JSON.stringify(body.messages);res.setHeader('Content-Type','text/event-stream');res.end('data: '+JSON.stringify({choices:[{delta:{content:'두 화자가 인사했습니다.'},finish_reason:'stop'}]})+'\n\ndata: [DONE]\n\n');
 });await new Promise(r=>backend.listen(0,'127.0.0.1',r));
 const original=await(await request.get('/api/config')).json();let session;
 try{
  const cfg=structuredClone(original),endpoint=`http://127.0.0.1:${backend.address().port}`;
  cfg.model.endpoint=endpoint;cfg.asr.enabled=true;cfg.asr.diarization=true;cfg.asr.endpoint=endpoint;cfg.asr.ffmpeg_endpoint=endpoint;cfg.extra.media_endpoint=endpoint;
  expect((await request.put('/api/config',{data:cfg})).ok()).toBeTruthy();
  const file=await(await request.post('/api/files',{multipart:{file:{name:'meeting.mp3',mimeType:'audio/mpeg',buffer:Buffer.from('ID3test')}}})).json();
  session=await(await request.post('/api/sessions',{data:{title:'화자 검증'}})).json();
  const response=await request.post('/api/chat',{data:{session_id:session.id,content:'녹음을 요약해',model:'test-model',tools_enabled:false,attachments:[file]}});
  expect(await response.text()).toContain('event: done');expect(received).toContain('화자 1: 안녕하세요');expect(received).toContain('화자 2: 반갑습니다');expect(diarCalls).toBe(1);
  await page.goto('/');await page.getByRole('button',{name:'전사 보기',exact:true}).click();
  await expect(page.locator('.transcript strong',{hasText:'화자 1'})).toBeVisible();await expect(page.locator('.transcript strong',{hasText:'화자 2'})).toBeVisible();
  await page.reload();await page.getByRole('button',{name:'전사 보기',exact:true}).click();await expect(page.locator('.transcript .content')).toContainText('반갑습니다');expect(diarCalls).toBe(1);
  await page.setViewportSize({width:390,height:760});expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBe(390);
 }finally{if(session)await request.delete(`/api/sessions/${session.id}`);await request.put('/api/config',{data:original});backend.closeAllConnections();await new Promise(r=>backend.close(r));}
});
