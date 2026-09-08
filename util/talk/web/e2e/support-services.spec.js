import { expect, test } from '@playwright/test';

test('manages all support services independently and saves deployment bindings', async ({ page, request }) => {
  let config = await (await request.get('/api/config')).json();
  config.runtime.mode = 'managed';
  config.runtime.bundle = 'flash-next';
  config.asr.enabled = false;
  config.tools.media_import_enabled = true;
  config.extra.documents_enabled = true;
  const keys = ['media', 'collector', 'documents', 'ssh'];
  const services = keys.map(key => ({
    ...config.runtime.catalog.components.find(c => c.id === `extra-${key}`),
    key, description: key, installed: key === 'documents' ? 'ready' : 'missing',
    version: key === 'documents' ? '0.4.0' : '0.1.0',
    enabled: true, status: key === 'documents' ? 'running' : 'missing',
    health: key === 'documents' ? 'online' : 'offline',
  }));
  const actions = [];
  let saved;
  await page.route('**/api/config', async route => {
    if (route.request().method() === 'PUT') { saved = route.request().postDataJSON(); config = saved; await route.fulfill({ json: { config, restart_required: false } }); }
    else await route.fulfill({ json: config });
  });
  await page.route('**/api/support', route => route.fulfill({ json: { services, managed: true, bundle_id: 'flash-next', operation: { state: 'complete' } } }));
  await page.route('**/api/runtime/components/*/*', async route => {
    const parts = new URL(route.request().url()).pathname.split('/');
    const id = parts.at(-2), action = parts.at(-1);
    actions.push({ id, action });
    const row = services.find(c => c.id === id);
    if (action === 'prepare') row.installed = 'ready';
    if (action === 'start') { row.status = 'running'; row.health = 'online'; }
    if (action === 'stop') { row.status = 'exited'; row.health = 'offline'; }
    await route.fulfill({ json: { operation: { state: 'complete' } } });
  });
  await page.goto('/');
  await page.locator('.settings-button').click();
  await page.getByRole('tab', { name: '시스템' }).click();
  await page.getByRole('button', { name: '지원 서비스', exact: true }).click();
  const media = page.getByRole('group', { name: '지원 서비스 media', exact: true });
  const docs = page.getByRole('group', { name: '지원 서비스 documents', exact: true });
  for (const key of keys) await expect(page.getByRole('group', { name: `지원 서비스 ${key}`, exact: true })).toBeVisible();
  await expect(media.getByRole('checkbox')).toBeChecked();
  await expect(media).toContainText('이미지 준비 필요');
  await expect(media).toContainText('상태: 정지');
  await expect(docs).toContainText('준비 완료');
  await docs.getByRole('button', { name: '이미지 준비', exact: true }).click();
  await expect(docs).toContainText('준비 완료');
  expect(actions.at(-1)).toEqual({ id: 'extra-documents', action: 'prepare' });
  await media.getByRole('button', { name: '이미지 준비', exact: true }).click();
  await expect(media).toContainText('이미지 준비됨');
  await media.getByRole('button', { name: '시작', exact: true }).click();
  await expect(media).toContainText('준비 완료');
  await expect(media.getByLabel('실행 호스트')).toBeDisabled();
  await media.getByRole('button', { name: '중지', exact: true }).click();
  await expect(media).toContainText('상태: 정지');
  await media.getByLabel('실행 호스트').selectOption('worker');
  await media.getByLabel('서버 포트').fill('18690');
  await media.getByLabel('API 주소').fill('http://192.168.100.60:18690');
  await media.getByLabel('API 주소').blur();
  await expect(media.getByRole('button', { name: '시작', exact: true })).toBeDisabled();
  await docs.getByRole('checkbox').uncheck();
  await page.getByRole('button', { name: '저장', exact: true }).click();
  await expect.poll(() => saved?.extra?.documents_enabled).toBe(false);
  const binding = saved.runtime.catalog.bundles.find(b => b.id === 'flash-next').bindings['extra-media'];
  expect(binding).toMatchObject({ host: 'worker', endpoint: 'http://192.168.100.60:18690', health_url: 'http://192.168.100.60:18690/health', port: 18690, auto_address: false });
});
