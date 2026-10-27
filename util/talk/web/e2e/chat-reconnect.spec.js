import {expect,test} from '@playwright/test';
import {createServer} from 'node:http';

if(process.env.SPARKTALK_CHROMIUM_PATH)test.use({launchOptions:{executablePath:process.env.SPARKTALK_CHROMIUM_PATH,args:['--no-sandbox','--disable-gpu','--disable-dev-shm-usage']}});

test('reload and closed tabs preserve work; explicit stop still cancels the server task',async({page,context,request})=>{
  const generations=[];
  const model=createServer(async(req,res)=>{
    let raw='';for await(const part of req)raw+=part;
    const data=JSON.parse(raw||'{}');
    if(!data.stream){res.setHeader('Content-Type','application/json');res.end(JSON.stringify({data:[{id:'test-model'}],choices:[{message:{content:'재연결 검증'}}]}));return;}
    let release;const gate=new Promise(resolve=>release=resolve);
    const generation={release,canceled:false,finished:false};generations.push(generation);
    res.on('close',()=>{if(!generation.finished)generation.canceled=true;});
    res.setHeader('Content-Type','text/event-stream');res.write('data: '+JSON.stringify({choices:[{delta:{content:'먼저 생성한 문장. '}}]})+'\n\n');
    await gate;
    generation.finished=true;
    if(!res.destroyed)res.end('data: '+JSON.stringify({choices:[{delta:{content:'끝까지 생성한 결과.'},finish_reason:'stop'}]})+'\n\ndata: [DONE]\n\n');
  });
  await new Promise(resolve=>model.listen(0,'127.0.0.1',resolve));
  const original=await(await request.get('/api/config')).json();let session,returned;
  try{
    const cfg=structuredClone(original);cfg.model.endpoint=`http://127.0.0.1:${model.address().port}`;cfg.model.default_model='test-model';cfg.model.model_type='qwen3.8';cfg.tools.enabled=false;cfg.asr.enabled=false;cfg.tts.enabled=false;
    expect((await request.put('/api/config',{data:cfg})).ok()).toBeTruthy();
    session=await(await request.post('/api/sessions',{data:{title:'브라우저 이탈 복구 검증'}})).json();
    await page.goto('/');await page.locator('.composer textarea').fill('연결이 끊겨도 계속 작성해');await page.locator('.composer textarea').press('Enter');
    await expect(page.locator('.messages')).toContainText('먼저 생성한 문장.');
    await page.reload();
    await expect(page.locator('.messages')).toContainText('먼저 생성한 문장.');
    await expect(page.getByRole('button',{name:'응답 중지',exact:true})).toBeVisible();
    expect(generations).toHaveLength(1);expect(generations[0].canceled).toBe(false);
    await page.close();generations[0].release();
    await expect.poll(async()=>{
      const messages=await(await request.get(`/api/sessions/${session.id}/messages`)).json();return messages.at(-1)?.content;
    }).toBe('먼저 생성한 문장. 끝까지 생성한 결과.');
    returned=await context.newPage();await returned.goto('/');
    await expect(returned.locator('.messages')).toContainText('끝까지 생성한 결과.');
    await expect(returned.getByRole('button',{name:'응답 중지',exact:true})).toHaveCount(0);
    expect(generations).toHaveLength(1);
    await returned.locator('.composer textarea').fill('이번에는 중지 버튼으로 취소할 작업');await returned.locator('.composer textarea').press('Enter');
    await expect.poll(()=>generations.length).toBe(2);
    await returned.reload();await expect(returned.getByRole('button',{name:'응답 중지',exact:true})).toBeVisible();
    await returned.getByRole('button',{name:'응답 중지',exact:true}).click();
    await expect.poll(()=>generations[1].canceled).toBe(true);
    await expect.poll(async()=>{
      const messages=await(await request.get(`/api/sessions/${session.id}/messages`)).json();return messages.at(-1)?.status;
    }).toBe('cancelled');
    expect(generations).toHaveLength(2);
  }finally{
    for(const generation of generations)generation.release();
    if(returned)await returned.close();
    if(session)await request.delete(`/api/sessions/${session.id}`);
    await request.put('/api/config',{data:original});model.closeAllConnections();await new Promise(resolve=>model.close(resolve));
  }
});
