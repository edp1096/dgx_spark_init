import { expect, test } from '@playwright/test';
import { createServer } from 'node:http';

test('shows separate download bars and ETA before the media response completes', async ({ page, request }) => {
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  let id = '', rounds = 0;
  const media = createServer(async (req, res) => {
    if (req.url.startsWith('/v1/source/progress?')) {
      if (!id || new URL(req.url, 'http://localhost').searchParams.get('id') !== id) { res.statusCode = 404; res.end(); return; }
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify({ stage: 'downloading', streams: [
        { id: 'v', kind: 'video', status: 'finished', downloaded_bytes: 67108864, total_bytes: 67108864 },
        { id: 'a', kind: 'audio', status: 'downloading', downloaded_bytes: 12582912, total_bytes: 25165824, speed_bps: 65536, eta_seconds: 192 },
      ] }));
      return;
    }
    if (req.url === '/v1/source/download') {
      id = req.headers['x-media-download-id'];
      await gate;
      res.setHeader('Content-Type', 'video/mp4');
      res.setHeader('Content-Disposition', 'attachment; filename="clip.mp4"');
      res.end(Buffer.concat([Buffer.from([0, 0, 0, 12]), Buffer.from('ftypisomvideo')]));
      return;
    }
    res.statusCode = 500; res.end('decoder unavailable');
  });
  const model = createServer(async (req, res) => {
    let raw = ''; for await (const part of req) raw += part;
    if (req.method !== 'POST' || !JSON.parse(raw || '{}').stream) { res.end(JSON.stringify({ data: [{ id: 'test-model' }], choices: [{ message: { content: '다운로드 표시 검사' } }] })); return; }
    res.setHeader('Content-Type', 'text/event-stream');
    const delta = rounds++ === 0 ? { tool_calls: [{ index: 0, id: 'download', type: 'function', function: { name: 'media_import', arguments: '{"url":"https://example.org/video"}' } }] } : { content: '검사 완료' };
    res.write('data: ' + JSON.stringify({ choices: [{ delta }] }) + '\n\n');
    res.end('data: [DONE]\n\n');
  });
  await Promise.all([new Promise(r => media.listen(0, '127.0.0.1', r)), new Promise(r => model.listen(0, '127.0.0.1', r))]);
  const original = await (await request.get('/api/config')).json();
  let session;
  try {
    const config = structuredClone(original);
    config.model.endpoint = `http://127.0.0.1:${model.address().port}`;
    config.model.model_type = 'qwen3.8';
    config.model.video_inputs = { [config.model.endpoint + '\ntest-model']: 'frames' };
    config.asr.enabled = false;
    config.asr.ffmpeg_endpoint = `http://127.0.0.1:${media.address().port}`;
    config.extra.media_endpoint = config.asr.ffmpeg_endpoint;
    config.tools.enabled = true;
    config.tools.media_import_enabled = true;
    expect((await request.put('/api/config', { data: config })).ok()).toBeTruthy();
    session = await (await request.post('/api/sessions', { data: { title: '다운로드 표시 검사' } })).json();
    await page.goto('/');
    await page.locator('.composer textarea').fill('https://example.org/video 영상 요약');
    await page.locator('.composer textarea').press('Enter');
    const progress = page.getByRole('status', { name: '미디어 다운로드 진행 상태' });
    await expect(progress).toContainText('영상 · 완료 · 64.0 MiB');
    await expect(progress).toContainText('음성 · 50.0%');
    await expect(progress).toContainText('64.0 KiB/s');
    await expect(progress).toContainText('현재 파일 약 3분 12초 남음');
    await expect(page.getByRole('progressbar', { name: '영상 다운로드 진행률' })).toHaveAttribute('value', '100');
    await expect(page.getByRole('progressbar', { name: '음성 다운로드 진행률' })).toHaveAttribute('value', '50');
    await expect(page.locator('.tool-trace')).not.toHaveAttribute('open');
    await page.screenshot({ path: 'test-results/media-download-progress.png' });
    release();
    await expect(page.locator('.messages')).toContainText('검사 완료');
    await expect(progress).toHaveCount(0);
    await expect(page.locator('.messages article.mine video')).toHaveCount(1);
  } finally {
    release();
    if (session) await request.delete(`/api/sessions/${session.id}`);
    await request.put('/api/config', { data: original });
    media.closeAllConnections(); model.closeAllConnections();
    await Promise.all([new Promise(r => media.close(r)), new Promise(r => model.close(r))]);
  }
});
