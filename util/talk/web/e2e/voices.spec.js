import {expect,test} from '@playwright/test';
import {createServer} from 'node:http';
if(process.env.SPARKTALK_CHROMIUM_PATH)test.use({launchOptions:{executablePath:process.env.SPARKTALK_CHROMIUM_PATH,args:['--no-sandbox','--disable-gpu','--disable-dev-shm-usage']}});

test('voice catalog has nine playable samples; reload and selection do not create a video',async({page,request})=>{
 test.skip(!process.env.SPARKTALK_E2E_MINIMAL_CATALOG,'requires isolated catalog');
 const ids=['sohee','uncle_fu','vivian','serena','dylan','eric','ryan','aiden','ono_anna'];
 let selected='',unexpected=0;
 const wav=Buffer.alloc(48044);wav.write('RIFF');wav.writeUInt32LE(48036,4);wav.write('WAVEfmt ',8);wav.writeUInt32LE(16,16);wav.writeUInt16LE(1,20);wav.writeUInt16LE(1,22);wav.writeUInt32LE(24000,24);wav.writeUInt32LE(48000,28);wav.writeUInt16LE(2,32);wav.writeUInt16LE(16,34);wav.write('data',36);wav.writeUInt32LE(48000,40);
 const worker=createServer(async(req,res)=>{
  res.setHeader('Content-Type','application/json');
  if(req.url==='/health'||req.url==='/v1/runtime/memory'){res.end('{"status":"ok","busy":false}');return;}
  if(req.url.endsWith('/preview.wav')){res.setHeader('Content-Type','audio/wav');res.end(wav);return;}
  if(req.url==='/v1/voices'){
   let raw='';for await(const part of req)raw+=part;const data=JSON.parse(raw);
   if(data.action==='select'){selected=data.voice_id;res.end(JSON.stringify({selected_voice:selected}));return;}
   expect(data.action).toBe('list');res.end(JSON.stringify({voices:ids.map(id=>({id,name:id,description:'검증 음성',native_language:'한국어'})),selected_voice:selected}));return;
  }
  unexpected++;res.statusCode=404;res.end('{}');
 });
 const model=createServer(async(req,res)=>{
  let raw='';for await(const p of req)raw+=p;const data=JSON.parse(raw||'{}');
  if(!data.stream){res.setHeader('Content-Type','application/json');res.end(JSON.stringify({data:[{id:'test-model'}],choices:[{message:{content:'음성 목록 검증'}}]}));return;}
  const latest=data.messages.filter(m=>m.role==='user').at(-1)?.content;
  const args=String(latest).includes('선택')?{action:'select',voice_id:'ryan'}:{action:'list'};
  res.setHeader('Content-Type','text/event-stream');res.end('data: '+JSON.stringify({choices:[{delta:{tool_calls:[{index:0,id:'voices',type:'function',function:{name:'shorts_voices',arguments:JSON.stringify(args)}}]}}]})+'\n\ndata: [DONE]\n\n');
 });
 await Promise.all([new Promise(r=>worker.listen(0,'127.0.0.1',r)),new Promise(r=>model.listen(0,'127.0.0.1',r))]);
 const original=await(await request.get('/api/config')).json();let session;
 try{
  const workerURL=`http://127.0.0.1:${worker.address().port}`,modelURL=`http://127.0.0.1:${model.address().port}`;
  const cfg=structuredClone(original);cfg.model.endpoint=modelURL;cfg.model.default_model='test-model';cfg.model.model_type='qwen3.8';cfg.asr.enabled=false;cfg.tts.enabled=false;cfg.tools.enabled=false;cfg.runtime.mode='managed';cfg.runtime.auto_start=false;cfg.runtime.bundle='shorts-chat-test';cfg.runtime.active_bundle='shorts-chat-test';
  cfg.runtime.catalog={hosts:{local:{}},components:[{id:'chat-test-llm',name:'Test model',role:'llm',controller:'external',endpoint:modelURL,health_url:modelURL+'/health',model:'test-model'},{id:'extra-shorts',name:'Worker',role:'tool',controller:'external',compose_asset:'compose.extra-shorts.yaml',endpoint:workerURL,health_url:workerURL+'/health'}],bundles:[{id:'shorts-chat-test',name:'Test',model_id:'test-model',model_type:'qwen3.8',context_tokens:65536,components:['chat-test-llm']}]};
  expect((await request.put('/api/config',{data:cfg})).ok()).toBeTruthy();session=await(await request.post('/api/sessions',{data:{title:'음성 목록 검증'}})).json();
  await page.goto('/');await expect(page.locator('header.chat-header')).toContainText('음성 목록 검증');
  await page.locator('.composer textarea').fill('영상 음성 목록 보여줘');await page.locator('.composer textarea').press('Enter');
  await expect(page.locator('.messages')).toContainText('9개입니다',{timeout:15000});await expect(page.locator('.messages audio')).toHaveCount(9);
  await page.locator('.messages audio').first().evaluate(el=>el.play());await expect.poll(()=>page.locator('.messages audio').first().evaluate(el=>el.currentTime)).toBeGreaterThan(0);
  await page.reload();await expect(page.locator('.messages audio')).toHaveCount(9);
  await page.locator('.composer textarea').fill('Ryan을 기본 음성으로 선택해');await page.locator('.composer textarea').press('Enter');
  await expect(page.locator('.messages')).toContainText('ryan로 선택했습니다',{timeout:15000});expect(selected).toBe('ryan');expect(unexpected).toBe(0);await expect(page.locator('.messages video')).toHaveCount(0);
 }finally{await request.put('/api/config',{data:original});if(session?.id)await request.delete('/api/sessions/'+session.id);await Promise.all([new Promise(r=>worker.close(r)),new Promise(r=>model.close(r))]);}
});
