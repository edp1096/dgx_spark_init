import { expect, test } from '@playwright/test';
import { createServer } from 'node:http';

test('keeps video despite a decoder failure, a later cover, completion and reload', async ({ page, request }) => {
  let finish;
  const gate = new Promise(resolve => { finish = resolve; });
  let rounds = 0;
  const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=', 'base64');
  const video = Buffer.concat([Buffer.from([0, 0, 0, 12]), Buffer.from('ftypisomvideo')]);
  const media = createServer(async (req, res) => {
    if (req.method !== 'POST') { res.end(JSON.stringify({status:'ok'})); return; }
    if (req.url === '/v1/video/frames') { res.statusCode = 500; res.end('decoder unavailable'); return; }
    let raw = ''; for await (const part of req) raw += part;
    const poster = JSON.parse(raw).url.endsWith('/poster');
    res.setHeader('Content-Type', poster ? 'image/png' : 'video/mp4');
    res.setHeader('Content-Disposition', `attachment; filename="${poster ? 'cover.png' : 'clip.mp4'}"`);
    res.end(poster ? png : video);
  });
  const model = createServer(async (req, res) => {
    if (req.method !== 'POST') { res.end(JSON.stringify({ data: [{ id: 'test-model' }] })); return; }
    let raw = ''; for await (const part of req) raw += part;
    if (!JSON.parse(raw).stream) { res.end(JSON.stringify({ choices: [{ message: { content: '미디어 보존 검사' } }] })); return; }
    res.setHeader('Content-Type', 'text/event-stream');
    rounds++;
    if (rounds <= 2) {
      const url = 'https://example.org/' + (rounds === 1 ? 'video' : 'poster');
      res.write('data: ' + JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, id: String(rounds), type: 'function', function: { name: 'media_import', arguments: JSON.stringify({ url }) } }] } }] }) + '\n\n');
    } else {
      res.write('data: ' + JSON.stringify({ choices: [{ delta: { content: '첨부 확인 중' } }] }) + '\n\n');
      await gate;
      res.write('data: ' + JSON.stringify({ choices: [{ delta: { content: '\n\n확인 완료' }, finish_reason: 'stop' }] }) + '\n\n');
    }
    res.end('data: [DONE]\n\n');
  });
  await Promise.all([new Promise(r => media.listen(0, '127.0.0.1', r)), new Promise(r => model.listen(0, '127.0.0.1', r))]);
  const original = await (await request.get('/api/config')).json();
  let session;
  try {
    const config = structuredClone(original);
    config.model.endpoint = `http://127.0.0.1:${model.address().port}`;
    config.model.model_type = 'qwen3.5';
    config.model.video_inputs = { [config.model.endpoint + '\ntest-model']: 'frames' };
    config.asr.enabled = false;
    config.asr.ffmpeg_endpoint = `http://127.0.0.1:${media.address().port}`;
    config.extra.media_endpoint = config.asr.ffmpeg_endpoint;
    config.tools.media_import_enabled = true;
    config.tools.max_rounds = 4;
    expect((await request.put('/api/config', { data: config })).ok()).toBeTruthy();
    session = await (await request.post('/api/sessions', { data: { title: '미디어 보존 검사' } })).json();
    await page.goto('/');
    await page.locator('.composer textarea').fill('https://example.org/video 와 https://example.org/poster 를 가져와 확인해.');
    await page.locator('.composer textarea').press('Enter');
    const player = page.locator('.messages article.mine video');
    await expect(player).toHaveCount(1);
    await expect(page.locator('.messages article.mine .media-image')).toHaveCount(1);
    await expect(page.locator('.messages')).toContainText('첨부 확인 중');
    const source = await player.getAttribute('src');
    await player.evaluate(node => { node.dataset.identity = 'original-video'; });
    finish();
    await expect(page.locator('.messages')).toContainText('확인 완료');
    await expect(page.locator('.messages article.mine')).not.toHaveAttribute('data-message-id', '');
    await expect(player).toHaveAttribute('src', source);
    await expect(player).toHaveAttribute('data-identity', 'original-video');
    const messages = await (await request.get(`/api/sessions/${session.id}/messages`)).json();
    expect(messages[0].attachments.map(a => a.mime)).toEqual(['video/mp4', 'image/png']);
    await page.reload();
    await expect(player).toHaveAttribute('src', source);
    await expect(page.locator('.messages article.mine .media-image')).toHaveCount(1);
  } finally {
    finish();
    if (session) await request.delete(`/api/sessions/${session.id}`);
    await request.put('/api/config', { data: original });
    media.closeAllConnections(); model.closeAllConnections();
    await Promise.all([new Promise(r => media.close(r)), new Promise(r => model.close(r))]);
  }
});
