import { expect, test } from '@playwright/test';
import { createServer } from 'node:http';

if (process.env.SPARKTALK_CHROMIUM_PATH) test.use({ launchOptions: { executablePath: process.env.SPARKTALK_CHROMIUM_PATH, args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'] } });

test('natural language creates a short with inline progress and one attachment, without a separate screen', async ({ page, request }) => {
  test.skip(!process.env.SPARKTALK_E2E_MINIMAL_CATALOG, 'requires the isolated managed-mode catalog');
  let completed = false, created = 0, rounds = 0, job;
  const worker = createServer(async (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    if (req.url === '/health') { res.end('{"status":"ok"}'); return; }
    if (req.url.endsWith('/files/final.mp4')) {
      res.setHeader('Content-Type', 'video/mp4');
      res.end(Buffer.from('\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isommp42')); return;
    }
    if (req.method === 'POST' && req.url === '/v1/projects') {
      let raw = ''; for await (const part of req) raw += part;
      const data = JSON.parse(raw);
      created++;
      job = { id: 'abcdef0123456789abcdef0123456789', session_id: data.session_id, title: data.title, status: 'queued', stage: '대기 중', progress: 0, created: Date.now()/1000, output_ready: false };
      res.end(JSON.stringify(job)); return;
    }
    if (job && req.url === '/v1/projects/' + job.id) {
      res.end(JSON.stringify({ ...job, status: completed ? 'completed' : 'running', stage: completed ? '완료' : 'Fish 음성 준비·합성', progress: completed ? 100 : 15, output_ready: completed })); return;
    }
    res.statusCode = 404; res.end('{}');
  });
  const model = createServer(async (req, res) => {
    let raw=''; for await (const part of req) raw += part;
    const data=JSON.parse(raw || '{}');
    if ((data.messages || []).some(message => message.role !== 'user' && Array.isArray(message.content) && message.content.some(part => part.type === 'video_url'))) {
      res.statusCode=400;res.end(JSON.stringify({error:{message:'video_url parts are supported only in user messages'}}));return;
    }
    if (!data.stream) { res.setHeader('Content-Type','application/json'); res.end(JSON.stringify({data:[{id:'test-model'}],choices:[{message:{content:'쇼츠 대화 검증'}}]}));return; }
    if(String(data.messages?.[0]?.content).includes('narrated-video execution intent router')){
      const input=JSON.parse(data.messages.at(-1).content);
      res.setHeader('Content-Type','text/event-stream');
      const content=JSON.stringify({action:'create',title:'캐시 원리',brief:input.latest_user_request,duration_seconds:60,bgm:true,voice_id:'',job_id:'',evidence:'쇼츠로 만들어줘'});
      res.end('data: '+JSON.stringify({choices:[{delta:{content}}]})+'\n\ndata: [DONE]\n\n');return;
    }
    res.setHeader('Content-Type','text/event-stream');
    rounds++; const delta = {content:'다음 주제로 메모리 계층을 추천합니다.'};
    res.write('data: '+JSON.stringify({choices:[{delta}]})+'\n\n'); res.end('data: [DONE]\n\n');
  });
  await Promise.all([new Promise(r=>worker.listen(0,'127.0.0.1',r)),new Promise(r=>model.listen(0,'127.0.0.1',r))]);
  const workerURL=`http://127.0.0.1:${worker.address().port}`, modelURL=`http://127.0.0.1:${model.address().port}`;
  const original=await (await request.get('/api/config')).json();
  let session;
  try {
    const cfg=structuredClone(original);
    cfg.model.endpoint=modelURL; cfg.model.default_model='test-model'; cfg.model.model_type='qwen3.8';
    cfg.asr.enabled=false; cfg.tts.enabled=false; cfg.tools.enabled=false;
    cfg.runtime.mode='managed'; cfg.runtime.auto_start=false; cfg.runtime.bundle='shorts-chat-test'; cfg.runtime.active_bundle='shorts-chat-test';
    cfg.runtime.catalog={hosts:{local:{}},components:[
      {id:'chat-test-llm',name:'검증 모델',role:'llm',controller:'external',endpoint:modelURL,health_url:modelURL+'/health',model:'test-model'},
      {id:'extra-shorts',name:'Extra Shorts',role:'tool',controller:'external',compose_asset:'compose.extra-shorts.yaml',endpoint:workerURL,health_url:workerURL+'/health'},
      {id:'extra-fish',name:'Fish',role:'tts',controller:'external',compose_asset:'compose.extra-fish.yaml',endpoint:'http://127.0.0.1:9',health_url:'http://127.0.0.1:9/health'},
    ],bundles:[{id:'shorts-chat-test',name:'쇼츠 대화 검증',model_id:'test-model',model_type:'qwen3.8',context_tokens:65536,components:['chat-test-llm']}]};
    const saved=await request.put('/api/config',{data:cfg}); expect(saved.ok(),await saved.text()).toBeTruthy();
    session=await (await request.post('/api/sessions',{data:{title:'쇼츠 자연어 경로 검증'}})).json();
    await page.goto('/');
    await expect(page.locator('header.chat-header .header-actions').getByRole('button',{name:/쇼츠/})).toHaveCount(0);
    await page.getByRole('button',{name:'대화 도구',exact:true}).click();
    await expect(page.getByRole('dialog',{name:'대화 도구 설정'}).getByRole('button',{name:/쇼츠/})).toHaveCount(0);
    await page.keyboard.press('Escape');
    await page.locator('.composer textarea').fill('캐시 원리를 한국어 쇼츠로 만들어줘. 브금도 넣어줘.');
    await page.locator('.composer textarea').press('Enter');
    await expect(page.locator('.tool-trace summary')).toContainText('Fish 음성 준비·합성 · 15%',{timeout:20000});
    await page.reload();
    await expect(page.locator('.tool-trace summary')).toContainText('Fish 음성 준비·합성 · 15%',{timeout:10000});
    expect(created).toBe(1);
    await expect(page.getByRole('dialog',{name:'쇼츠 만들기'})).toHaveCount(0);
    await page.locator('.tool-trace summary').click();
    await expect(page.getByRole('progressbar',{name:'현재 생성 단계 진행률'})).toHaveAttribute('value','15');
    await page.screenshot({path:'test-results/shorts-inline-progress.png'});
    completed=true;
    await expect(page.locator('.messages')).toContainText('검수한 영상을 첨부했습니다.',{timeout:15000});
    await expect(page.locator('.messages video')).toHaveCount(1);
    expect(created).toBe(1);
    const messages=await (await request.get(`/api/sessions/${session.id}/messages`)).json();
    expect(messages.flatMap(m=>m.attachments || []).filter(a=>a.mime==='video/mp4')).toHaveLength(1);
    await page.locator('.composer textarea').fill('이전 영상은 그대로 두고 다음 주제만 추천해줘.');
    await page.locator('.composer textarea').press('Enter');
    await expect.poll(()=>rounds).toBe(1);
    await expect(page.locator('.messages')).not.toContainText('video_url parts are supported only in user messages');
  } finally {
    completed=true;
    if(session) await request.delete(`/api/sessions/${session.id}`);
    await request.put('/api/config',{data:original});
    worker.closeAllConnections(); model.closeAllConnections();
    await Promise.all([new Promise(r=>worker.close(r)),new Promise(r=>model.close(r))]);
  }
});
