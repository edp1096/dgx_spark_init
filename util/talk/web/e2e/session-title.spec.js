import { test, expect } from '@playwright/test';
import { createServer } from 'node:http';

test('model tool updates the current title during streaming and preserves manual names', async ({ page, request }) => {
  let finishAnswer;
  const answerReady = new Promise(resolve => { finishAnswer = resolve; });
  let sawTitleTool = false;
  const backend = createServer(async (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    if (req.method !== 'POST') {
      res.end(JSON.stringify({ data: [{ id: 'test-model', context_length: 32768 }] })); return;
    }
    let raw = ''; for await (const part of req) raw += part;
    const body = JSON.parse(raw);
    if (!body.stream) { res.end(JSON.stringify({ choices: [{ message: { content: '늦은 제목' } }] })); return; }
    sawTitleTool = (body.tools || []).some(tool => tool.function.name === 'session_title');
    res.setHeader('Content-Type', 'text/event-stream');
    const lastUser = body.messages.findLastIndex(message => message.role === 'user');
    const userText = body.messages[lastUser].content;
    const explicitRename = userText === '대화방 제목을 261009 주요뉴스로 바꿔라.';
    if (!body.messages.slice(lastUser + 1).some(message => message.role === 'tool')) {
      expect(sawTitleTool).toBe(true);
      const args = explicitRename ? { title: '261009 주요뉴스', user_request: userText } : { title: 'GPU 의미 검색' };
      res.end('data: ' + JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'title-one', type: 'function', function: { name: 'session_title', arguments: JSON.stringify(args) } }] }, finish_reason: 'tool_calls' }] }) + '\n\ndata: [DONE]\n\n');
    } else {
      await answerReady;
      res.end('data: ' + JSON.stringify({ choices: [{ delta: { content: explicitRename ? '대화방 제목을 변경했습니다.' : '27입니다.' }, finish_reason: 'stop' }] }) + '\n\ndata: [DONE]\n\n');
    }
  });
  await new Promise(resolve => backend.listen(0, '127.0.0.1', resolve));
  const original = await (await request.get('/api/config')).json();
  let current, other;
  try {
    const cfg = structuredClone(original);
    cfg.model.endpoint = `http://127.0.0.1:${backend.address().port}`;
    cfg.context.enabled = false; cfg.tools.enabled = false;
    expect((await request.put('/api/config', { data: cfg })).ok()).toBeTruthy();
    other = await (await request.post('/api/sessions', { data: { title: '별도 대화방' } })).json();
    current = await (await request.post('/api/sessions', { data: { title: '새 대화' } })).json();
    await page.goto('/');
    await page.locator('.composer textarea').fill('12+15를 계산해줘');
    await page.getByRole('button', { name: '메시지 전송', exact: true }).click();
    // The second model response is blocked: the title must arrive through its own SSE event.
    await expect(page.locator('.chat-title')).toContainText('GPU 의미 검색');
    await expect(page.locator('.session-select').filter({ hasText: /^GPU 의미 검색$/ })).toHaveCount(1);
    await expect(page.getByRole('button', { name: '응답 중지', exact: true })).toBeVisible();
    await expect(page.locator('.session-select').filter({ hasText: /^별도 대화방$/ })).toHaveCount(1);
    finishAnswer();
    await expect(page.getByText('27입니다.', { exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: '응답 중지', exact: true })).toHaveCount(0);
    await page.locator('.chat-title').click();
    await page.getByRole('textbox', { name: '대화 제목', exact: true }).fill('내가 정한 제목');
    await page.getByRole('textbox', { name: '대화 제목', exact: true }).press('Enter');
    await expect(page.locator('.chat-title')).toContainText('내가 정한 제목');
    await page.reload();
    await expect(page.locator('.chat-title')).toContainText('내가 정한 제목');
    await page.locator('.composer textarea').fill('대화방 제목을 261009 주요뉴스로 바꿔라.');
    await page.getByRole('button', { name: '메시지 전송', exact: true }).click();
    await expect(page.locator('.chat-title')).toContainText('261009 주요뉴스');
    await expect(page.getByText('대화방 제목을 변경했습니다.', { exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: '응답 중지', exact: true })).toHaveCount(0);
    await page.reload();
    await expect(page.locator('.chat-title')).toContainText('261009 주요뉴스');
    expect(sawTitleTool).toBe(true);
  } finally {
    finishAnswer(); backend.closeAllConnections(); await new Promise(resolve => backend.close(resolve));
    if (current) await request.delete(`/api/sessions/${current.id}`);
    if (other) await request.delete(`/api/sessions/${other.id}`);
    await request.put('/api/config', { data: original });
  }
});
