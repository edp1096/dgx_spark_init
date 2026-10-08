import { test, expect } from '@playwright/test';
import { createServer } from 'node:http';

for (const position of ['middle', 'bottom']) {
  test(`keeps the ${position} reading position when a streamed answer is saved`, async ({ page, request }) => {
    const answer = Array.from({ length: 100 }, (_, i) => `문단 ${i}: 완료 전후에도 읽던 위치가 유지되어야 합니다.`).join('\n\n');
    let finish;
    const gate = new Promise(resolve => { finish = resolve; });
    const backend = createServer(async (req, res) => {
      if (req.method !== 'POST') { res.end(JSON.stringify({ data: [{ id: 'test-model' }] })); return; }
      let raw = ''; for await (const part of req) raw += part;
      if (!JSON.parse(raw).stream) { res.end(JSON.stringify({ choices: [{ message: { content: 'Scroll test' }, finish_reason: 'stop' }] })); return; }
      res.setHeader('Content-Type', 'text/event-stream');
      res.write('data: ' + JSON.stringify({ choices: [{ delta: { content: answer }, finish_reason: null }] }) + '\n\n');
      await gate;
      res.end('data: ' + JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }] }) + '\n\ndata: [DONE]\n\n');
    });
    await new Promise(resolve => backend.listen(0, '127.0.0.1', resolve));
    const original = await (await request.get('/api/config')).json();
    let session;
    try {
      const cfg = structuredClone(original); cfg.model.endpoint = `http://127.0.0.1:${backend.address().port}`;
      expect((await request.put('/api/config', { data: cfg })).ok()).toBeTruthy();
      session = await (await request.post('/api/sessions', { data: { title: 'Scroll position' } })).json();
      await page.goto('/');
      await page.locator('textarea').first().fill('긴 답변을 작성하세요.');
      await page.locator('textarea').first().press('Enter');
      const pane = page.locator('.messages');
      await expect(pane).toContainText('문단 99:');
      await pane.evaluate((el, position) => {
        el.style.scrollBehavior = 'auto';
        el.scrollTop = position === 'middle' ? el.scrollHeight / 2 : el.scrollHeight;
      }, position);
      await pane.locator('article').last().evaluate(el => { el.dataset.scrollTestIdentity = 'retained'; });
      const before = await pane.evaluate(el => el.scrollTop);
      await pane.evaluate(el => { el.style.scrollBehavior = ''; });
      finish();
      await expect(pane.locator('article[data-message-id]:not([data-message-id=""])')).toHaveCount(2);
      await expect(pane.locator('article').last()).toHaveAttribute('data-scroll-test-identity', 'retained');
      await expect.poll(async () => pane.evaluate((el, args) => args.position === 'middle' ? Math.abs(el.scrollTop - args.before) : el.scrollHeight - el.scrollTop - el.clientHeight, { position, before })).toBeLessThan(5);
    } finally {
      finish();
      if (session) await request.delete(`/api/sessions/${session.id}`);
      await request.put('/api/config', { data: original });
      backend.closeAllConnections(); await new Promise(resolve => backend.close(resolve));
    }
  });
}
