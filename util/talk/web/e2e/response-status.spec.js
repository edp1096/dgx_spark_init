import { test, expect } from '@playwright/test';
import { createServer } from 'node:http';

test('shows status immediately and while waiting for the first model token', async ({ page, request }) => {
  let releaseRequest, releaseToken, releaseAnswer;
  const requestGate = new Promise(resolve => releaseRequest = resolve);
  const tokenGate = new Promise(resolve => releaseToken = resolve);
  const answerGate = new Promise(resolve => releaseAnswer = resolve);
  const model = createServer(async (req, res) => {
    let raw = ''; for await (const chunk of req) raw += chunk;
    const data = JSON.parse(raw || '{}');
    if (!data.stream) { res.setHeader('Content-Type','application/json'); res.end(JSON.stringify({data:[{id:'test-model'}],choices:[{message:{content:'초기 상태 검증'}}]})); return; }
    await tokenGate;
    if (res.destroyed) return;
    res.setHeader('Content-Type','text/event-stream');
    res.write('data: '+JSON.stringify({choices:[{delta:{reasoning_content:'생각이 시작됐습니다.'}}]})+'\n\n');
    await answerGate;
    if (!res.destroyed) res.end('data: '+JSON.stringify({choices:[{delta:{content:'답변 완료'},finish_reason:'stop'}]})+'\n\ndata: [DONE]\n\n');
  });
  await new Promise(resolve => model.listen(0,'127.0.0.1',resolve));
  const original = await (await request.get('/api/config')).json();
  let session;
  try {
    const cfg=structuredClone(original); cfg.model.endpoint=`http://127.0.0.1:${model.address().port}`; cfg.model.default_model='test-model'; cfg.model.model_type='qwen3.8'; cfg.tools.enabled=false; cfg.asr.enabled=false; cfg.tts.enabled=false;
    expect((await request.put('/api/config',{data:cfg})).ok()).toBeTruthy();
    session=await(await request.post('/api/sessions',{data:{title:'초기 상태 검증'}})).json();
    await page.route('**/api/chat', async route => { await requestGate; await route.continue(); });
    await page.goto('/');
    await page.locator('.composer textarea').fill('첫 출력 전 상태를 확인해');
    await page.locator('.composer textarea').press('Enter');
    await expect(page.locator('.response-stage')).toHaveText('요청 전송 중');
    releaseRequest();
    await expect(page.locator('.response-stage')).toHaveText('모델 첫 응답 대기 중');
    await expect(page.locator('.messages')).not.toContainText('답변 완료');
    await page.reload();
    await expect(page.locator('.response-stage')).toHaveText('모델 첫 응답 대기 중');
    releaseToken();
    await expect(page.getByRole('button',{name:'생각 과정',exact:false})).toBeVisible();
    await expect(page.locator('.response-stage')).toHaveCount(0);
    releaseAnswer();
    await expect(page.locator('.messages')).toContainText('답변 완료');
    await expect(page.getByRole('button',{name:'응답 중지',exact:true})).toHaveCount(0);
  } finally {
    releaseRequest(); releaseToken(); releaseAnswer();
    if (session) await request.delete(`/api/sessions/${session.id}`);
    await request.put('/api/config',{data:original});
    model.closeAllConnections(); await new Promise(resolve=>model.close(resolve));
  }
});
