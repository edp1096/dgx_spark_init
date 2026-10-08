import { test, expect } from '@playwright/test';
import { createServer } from 'node:http';

test('additional input interrupts inference, survives navigation/reload, and does not duplicate', async ({ page, request }) => {
  // Match LAN HTTP browsers, where randomUUID is unavailable.
  await page.addInitScript(() => Object.defineProperty(crypto, 'randomUUID', { value: undefined }));
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  let calls = 0, oldClosed = false;
  let resume;
  const resumed = new Promise(resolve => { resume = resolve; });
  const backend = createServer(async (req, res) => {
    if (req.method !== 'POST') { res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify({data:[{id:'test-model',context_length:32768}]})); return; }
    let raw=''; for await (const part of req) raw+=part;
    const body=JSON.parse(raw);
    if (!body.stream) { res.end(JSON.stringify({choices:[{message:{content:'추가 입력 시험'},finish_reason:'stop'}]})); return; }
    calls++;
    res.setHeader('Content-Type','text/event-stream');
    if (calls===1) {
      res.on('close',()=>{oldClosed=true;});
      res.write('data: '+JSON.stringify({choices:[{delta:{reasoning_content:'첫 추론 중'},finish_reason:null}]})+'\n\n');
      return;
    }
    expect(JSON.stringify(body.messages)).toContain('원래 작업');
    expect(JSON.stringify(body.messages)).toContain('한국어로 바꿔');
    await resumed;
    res.end('data: '+JSON.stringify({choices:[{delta:{content:'추가 지시 반영 완료'},finish_reason:'stop'}]})+'\n\ndata: [DONE]\n\n');
  });
  await new Promise(resolve=>backend.listen(0,'127.0.0.1',resolve));
  const original=await(await request.get('/api/config')).json();
  let session, other;
  try {
    const cfg=structuredClone(original); cfg.model.endpoint=`http://127.0.0.1:${backend.address().port}`;cfg.context.enabled=false;cfg.context.window_tokens=32768;cfg.tools.enabled=false;
    expect((await request.put('/api/config',{data:cfg})).ok()).toBeTruthy();
    other=await(await request.post('/api/sessions',{data:{title:'별도 대화방'}})).json();
    session=await(await request.post('/api/sessions',{data:{title:'추가 입력 시험'}})).json();
    await page.goto('/');
    const input=page.locator('.composer textarea');
    await input.fill('원래 작업');await page.getByRole('button',{name:'메시지 전송',exact:true}).click();
    await expect(page.getByRole('button',{name:'응답 중지',exact:true})).toBeVisible();
    await expect(input).toBeEnabled();
    await input.fill('한국어로 바꿔');await page.getByRole('button',{name:'추가 지시 전송',exact:true}).click();
    await expect(input).toHaveValue('');
    expect(pageErrors).toEqual([]);
    await expect(page.getByText('한국어로 바꿔',{exact:true})).toBeVisible();
    await expect.poll(()=>calls).toBe(2);await expect.poll(()=>oldClosed).toBe(true);
    await page.getByText('별도 대화방',{exact:true}).first().click();
    await expect(page.getByText('한국어로 바꿔',{exact:true})).toHaveCount(0);
    await page.getByText('추가 입력 시험',{exact:true}).first().click();
    await expect(page.getByText('한국어로 바꿔',{exact:true})).toBeVisible();
    resume();
    await expect(page.getByText('추가 지시 반영 완료',{exact:true})).toBeVisible();
    await expect(page.getByRole('button',{name:'응답 중지',exact:true})).toHaveCount(0);
    await page.reload();
    await expect(page.getByText('한국어로 바꿔',{exact:true})).toHaveCount(1);
    const messages=await(await request.get(`/api/sessions/${session.id}/messages`)).json();
    expect(messages[0].turn_inputs).toHaveLength(1);expect(messages[1].content).toBe('추가 지시 반영 완료');
  } finally {
    resume();backend.closeAllConnections();await new Promise(resolve=>backend.close(resolve));
    if(session)await request.delete(`/api/sessions/${session.id}`);
    if(other)await request.delete(`/api/sessions/${other.id}`);
    await request.put('/api/config',{data:original});
  }
});
