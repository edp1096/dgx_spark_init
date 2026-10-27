import {expect,test} from '@playwright/test';
import {createServer} from 'node:http';

if(process.env.SPARKTALK_CHROMIUM_PATH)test.use({launchOptions:{executablePath:process.env.SPARKTALK_CHROMIUM_PATH,args:['--no-sandbox','--disable-gpu','--disable-dev-shm-usage']}});

test('photo-to-video keeps one job across reload and publishes only its reviewed result',async({page,request})=>{
 test.skip(!process.env.SPARKTALK_E2E_MINIMAL_CATALOG,'requires isolated managed catalog');
 let completed=false,created=0,rounds=0,job,sourceBytes;
 const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=','base64');
 const worker=createServer(async(req,res)=>{
  res.setHeader('Content-Type','application/json');
  if(req.url==='/health'||req.url==='/v1/runtime/memory'){res.end('{"status":"ok","busy":false}');return;}
  if(req.method==='POST'&&req.url==='/v1/clips'){
   let raw='';for await(const part of req)raw+=part;
   const data=JSON.parse(raw);created++;sourceBytes=data.source_image;
   job={id:'abcdef0123456789abcdef0123456789',session_id:data.session_id,status:'queued',stage:'대기 중',progress:0,created:Date.now()/1000,output_ready:false};
   res.end(JSON.stringify(job));return;
  }
  if(job&&req.url==='/v1/clips/'+job.id){
   res.end(JSON.stringify({...job,status:completed?'completed':'running',stage:completed?'검사·수정 완료':'원본과 편집 결과의 손·얼굴·구도 검사',progress:completed?100:35,output_ready:completed,report:completed?{passed:true,plan:{title:'만화 사진 영상'}}:undefined}));return;
  }
  if(req.url.endsWith('/files/final.mp4')){
   if(!completed){res.statusCode=409;res.end('{}');return;}
   res.setHeader('Content-Type','video/mp4');res.end(Buffer.from('\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isommp42'));return;
  }
  res.statusCode=404;res.end('{}');
 });
 const model=createServer(async(req,res)=>{
  let raw='';for await(const part of req)raw+=part;
  const data=JSON.parse(raw||'{}');
  if(!data.stream){res.setHeader('Content-Type','application/json');res.end(JSON.stringify({data:[{id:'test-model'}],choices:[{message:{content:'사진 영상 제작 검증'}}]}));return;}
  if(String(data.messages?.[0]?.content).includes('photo-to-video intent router')){
   const input=JSON.parse(data.messages.at(-1).content);
   const content=JSON.stringify({action:'produce',source_image_id:input.available_images[0].id,job_id:'',brief:input.latest_user_request,resolution:'480p',evidence:input.latest_user_request});
   res.setHeader('Content-Type','text/event-stream');res.end('data: '+JSON.stringify({choices:[{delta:{content}}]})+'\n\ndata: [DONE]\n\n');return;
  }
  const source=JSON.stringify(data.messages).match(/id=([a-f0-9]{32})/)?.[1];
  res.setHeader('Content-Type','text/event-stream');
  const delta=rounds++===0?{tool_calls:[{index:0,id:'clip-call',type:'function',function:{name:'clip_produce',arguments:JSON.stringify({source_image_id:source||'missing',brief:'이 사진 만화처럼 바꾸고 손 흔들게 해줘'})}}]}:{content:'검사한 영상을 첨부했습니다.'};
  res.end('data: '+JSON.stringify({choices:[{delta}]})+'\n\ndata: [DONE]\n\n');
 });
 await Promise.all([new Promise(r=>worker.listen(0,'127.0.0.1',r)),new Promise(r=>model.listen(0,'127.0.0.1',r))]);
 const original=await(await request.get('/api/config')).json();let session;
 try{
  const workerURL=`http://127.0.0.1:${worker.address().port}`,modelURL=`http://127.0.0.1:${model.address().port}`;
  const cfg=structuredClone(original);cfg.model.endpoint=modelURL;cfg.model.default_model='test-model';cfg.model.model_type='qwen3.8';cfg.asr.enabled=false;cfg.tts.enabled=false;cfg.tools.enabled=false;
  cfg.runtime.mode='managed';cfg.runtime.auto_start=false;cfg.runtime.bundle='clip-chat-test';cfg.runtime.active_bundle='clip-chat-test';
  cfg.runtime.catalog={hosts:{local:{}},components:[
   {id:'chat-test-llm',name:'Test model',role:'llm',controller:'external',endpoint:modelURL,health_url:modelURL+'/health',model:'test-model'},
   {id:'extra-shorts',name:'Production worker',role:'tool',controller:'external',compose_asset:'compose.extra-shorts.yaml',endpoint:workerURL,health_url:workerURL+'/health'},
   {id:'qwim-mmh3',name:'Test image/video',role:'image',controller:'external',compose_asset:'compose.qwim-mmh3.yaml',endpoint:workerURL,health_url:workerURL+'/health',model:'qwen-image-2.1-uc-nvfp4'},
  ],bundles:[{id:'clip-chat-test',name:'Photo production test',model_id:'test-model',model_type:'qwen3.8',context_tokens:65536,components:['chat-test-llm','qwim-mmh3']}]};
  const saved=await request.put('/api/config',{data:cfg});expect(saved.ok(),await saved.text()).toBeTruthy();
  session=await(await request.post('/api/sessions',{data:{title:'사진 영상 자동 제작 검증'}})).json();
  await page.goto('/');
  await expect(page.locator('header.chat-header')).toContainText('사진 영상 자동 제작 검증');
  await page.locator('.composer input.media-input').setInputFiles({name:'source.png',mimeType:'image/png',buffer:png});
  await expect(page.getByRole('button',{name:'source.png 첨부 제거',exact:true})).toBeVisible();
  await page.locator('.composer textarea').fill('이 사진 만화처럼 바꾸고 손 흔들게 해줘');await page.locator('.composer textarea').press('Enter');
  await expect(page.locator('.tool-trace summary')).toContainText('손·얼굴·구도 검사 · 35%',{timeout:20000});
  expect(sourceBytes).toBe('data:image/png;base64,'+png.toString('base64'));
  await expect(page.locator('.messages video')).toHaveCount(0);
  await page.reload();await expect(page.locator('.tool-trace summary')).toContainText('손·얼굴·구도 검사 · 35%',{timeout:10000});
  expect(created).toBe(1);completed=true;
  await expect(page.locator('.messages')).toContainText('검수한 영상을 첨부했습니다.',{timeout:15000});
  await expect(page.locator('.messages video')).toHaveCount(1);expect(created).toBe(1);
  const messages=await(await request.get(`/api/sessions/${session.id}/messages`)).json();
  expect(messages.flatMap(x=>x.attachments||[]).filter(x=>x.mime==='video/mp4')).toHaveLength(1);
 }finally{
  completed=true;
  if(session)await request.delete(`/api/sessions/${session.id}`);
  await request.put('/api/config',{data:original});
  worker.closeAllConnections();model.closeAllConnections();
  await Promise.all([new Promise(r=>worker.close(r)),new Promise(r=>model.close(r))]);
 }
});
