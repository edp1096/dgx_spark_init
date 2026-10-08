import { expect, test } from '@playwright/test';

test('plays streamed PCM at 1.3 with real Web Audio and unchanged pitch', async ({ page }) => {
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.addInitScript(() => {
    window.speechBuffers = [];
    const Original = window.AudioContext;
    window.AudioContext = class extends Original {
      createBuffer(...args) {
        const buffer = super.createBuffer(...args);
        window.speechBuffers.push(buffer);
        return buffer;
      }
    };
  });
  const id = 'speak-rate-session';
  const now = new Date().toISOString();
  await page.route('**/api/groups', (route) => route.fulfill({ json: [] }));
  await page.route('**/api/models', (route) => route.fulfill({ json: ['test-model'] }));
  await page.route('**/api/sessions', (route) => route.fulfill({ json: [{ id, title: '속도 확인', model: 'test-model', created_at: now, updated_at: now }] }));
  await page.route(`**/api/sessions/${id}/messages`, (route) => route.fulfill({ json: [
    { id: 1, session_id: id, role: 'user', status: 'completed', content: '읽어 주세요.', created_at: now },
    { id: 2, session_id: id, role: 'assistant', status: 'completed', content: '말하기 속도를 확인합니다.', created_at: now },
  ] }));
  await page.route(`**/api/sessions/${id}/context`, (route) => route.fulfill({ json: { enabled: true, segments: [] } }));
  await page.route(`**/api/sessions/${id}/ssh-grants`, (route) => route.fulfill({ json: [] }));
  const pcm = Buffer.alloc(2 * 24000 * 2);
  for (let i = 0; i < 48000; i += 1) pcm.writeInt16LE(Math.round(10000 * Math.sin(2 * Math.PI * 220 * i / 24000)), i * 2);
  await page.route('**/api/tts/speech', (route) => route.fulfill({
    body: pcm,
    headers: { 'Content-Type': 'audio/pcm', 'X-Audio-Sample-Rate': '24000', 'X-Audio-Speak-Rate': '1.3' },
  }));
  await page.goto('/');
  const reply = page.locator('article[data-message-id="2"]');
  await reply.getByRole('button', { name: '🔊 읽기', exact: true }).click();
  await expect(reply.getByRole('button', { name: '■ 정지', exact: true })).toBeVisible();
  await expect(reply.getByRole('button', { name: '🔊 읽기', exact: true })).toBeVisible();
  const result = await page.evaluate(() => {
    const frames = window.speechBuffers.reduce((total, buffer) => total + buffer.length, 0);
    const mono = new Float32Array(frames);
    let offset = 0;
    for (const buffer of window.speechBuffers) { mono.set(buffer.getChannelData(0), offset); offset += buffer.length; }
    const crossings = [];
    for (let i = 4000; i < frames - 4000; i += 1) if (mono[i - 1] <= 0 && mono[i] > 0) crossings.push(i);
    return { frames, frequency: (crossings.length - 1) * 24000 / (crossings.at(-1) - crossings[0]) };
  });
  expect(result.frames).toBe(Math.round(48000 / 1.3));
  expect(Math.abs(result.frequency - 220)).toBeLessThan(3);
  expect(errors).toEqual([]);
});
