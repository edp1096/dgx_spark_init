import { test, expect } from '@playwright/test';
import { createServer } from 'node:http';

test('model moves the current chat into a folder during streaming and can ungroup it', async ({ page, request }) => {
  let finishAnswer;
  const answerReady = new Promise(resolve => { finishAnswer = resolve; });
  let destination;
  const backend = createServer(async (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    if (req.method !== 'POST') { res.end(JSON.stringify({ data: [{ id: 'test-model', context_length: 32768 }] })); return; }
    let raw = ''; for await (const part of req) raw += part;
    const body = JSON.parse(raw);
    if (!body.stream) { res.end(JSON.stringify({ choices: [{ message: { content: '대화' } }] })); return; }
    expect(body.tools.some(tool => tool.function.name === 'session_folder')).toBe(true);
    const lastUser = body.messages.findLastIndex(message => message.role === 'user');
    const userText = body.messages[lastUser].content;
    const ungroup = userText.includes('빼라');
    res.setHeader('Content-Type', 'text/event-stream');
    if (!body.messages.slice(lastUser + 1).some(message => message.role === 'tool')) {
      const args = ungroup ? { action: 'ungroup', user_request: userText } : { action: 'move', group_id: destination.id, user_request: userText };
      res.end('data: ' + JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'folder-one', type: 'function', function: { name: 'session_folder', arguments: JSON.stringify(args) } }] }, finish_reason: 'tool_calls' }] }) + '\n\ndata: [DONE]\n\n');
    } else {
      await answerReady;
      res.end('data: ' + JSON.stringify({ choices: [{ delta: { content: ungroup ? '폴더에서 뺐습니다.' : '작업 폴더로 옮겼습니다.' }, finish_reason: 'stop' }] }) + '\n\ndata: [DONE]\n\n');
    }
  });
  await new Promise(resolve => backend.listen(0, '127.0.0.1', resolve));
  const original = await (await request.get('/api/config')).json();
  let current, other;
  try {
    const cfg = structuredClone(original); cfg.model.endpoint = `http://127.0.0.1:${backend.address().port}`;
    cfg.context.enabled = false; cfg.tools.enabled = false;
    expect((await request.put('/api/config', { data: cfg })).ok()).toBeTruthy();
    destination = await (await request.post('/api/groups', { data: { name: '작업' } })).json();
    other = await (await request.post('/api/sessions', { data: { title: '다른 대화방' } })).json();
    current = await (await request.post('/api/sessions', { data: { title: '이동할 대화방' } })).json();
    await request.patch(`/api/sessions/${current.id}`, { data: { title: current.title } });
    await page.goto('/');
    const folder = page.locator('.chat-group').filter({ has: page.locator('.group-toggle').filter({ hasText: '작업' }) });
    await folder.locator('.group-toggle').click(); // Initially collapse the destination.
    await page.locator('.composer textarea').fill('이 대화방을 작업 폴더로 옮겨라.');
    await page.getByRole('button', { name: '메시지 전송', exact: true }).click();
    // The final answer is blocked: the folder must update via its SSE event.
    await expect(folder.locator('.session-select')).toHaveText('이동할 대화방');
    await expect(folder.locator('.group-toggle')).toHaveAttribute('aria-expanded', 'true');
    await expect(page.locator('.chat-title')).toContainText('이동할 대화방');
    await expect(page.getByRole('button', { name: '응답 중지', exact: true })).toBeVisible();
    const sessions = await (await request.get('/api/sessions')).json();
    expect(sessions.find(item => item.id === current.id).group_id).toBe(destination.id);
    expect(sessions.find(item => item.id === other.id).group_id).toBe('');
    finishAnswer();
    await expect(page.getByText('작업 폴더로 옮겼습니다.', { exact: true })).toBeVisible();
    await page.reload();
    await expect(folder.locator('.session-select')).toHaveText('이동할 대화방');
    await page.locator('.chat-group.ungrouped .group-toggle').click();
    await page.locator('.composer textarea').fill('이 대화방을 폴더에서 빼라.');
    await page.getByRole('button', { name: '메시지 전송', exact: true }).click();
    await expect(page.getByText('폴더에서 뺐습니다.', { exact: true })).toBeVisible();
    await expect(folder.locator('.session-select')).toHaveCount(0);
    await expect(page.locator('.chat-group.ungrouped .session-select').filter({ hasText: '이동할 대화방' })).toHaveCount(1);
    await page.reload();
    await expect(page.locator('.chat-group.ungrouped .session-select').filter({ hasText: '이동할 대화방' })).toHaveCount(1);
  } finally {
    finishAnswer(); backend.closeAllConnections(); await new Promise(resolve => backend.close(resolve));
    if (current) await request.delete(`/api/sessions/${current.id}`);
    if (other) await request.delete(`/api/sessions/${other.id}`);
    if (destination) await request.delete(`/api/groups/${destination.id}`);
    await request.put('/api/config', { data: original });
  }
});
