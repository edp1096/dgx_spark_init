import { expect, test } from '@playwright/test';

test('configures physical key replicas without losing them on settings save', async ({ page }) => {
  let saved;
  let extraOnline = true;
  let state = { hosts: [], peers: {}, available_hosts: ['local', 'worker'], report: { replicas: [], authority_host: '' } };
  await page.route('**/api/config', async route => {
    if (route.request().method() === 'PUT') {
      saved = route.request().postDataJSON();
      await route.fulfill({ json: { config: saved, restart_required: false } });
    } else {
      const response = await route.fetch();
      const config = await response.json();
      config.extra.ssh_enabled = true;
      config.runtime.key_store_hosts = state.hosts;
      config.runtime.key_store_peers = state.peers;
      await route.fulfill({ json: config });
    }
  });
  await page.route('**/api/health', route => route.fulfill({ json: { status: 'ok', extra: { ssh: { status: extraOnline ? 'ok' : 'offline' } } } }));
  await page.route('**/api/ssh/keys', route => route.fulfill({ json: [] }));
  await page.route('**/api/ssh/key-store', async route => {
    if (route.request().method() === 'POST') {
      const req = route.request().postDataJSON();
      if (req.action === 'configure') {
        state.hosts = req.hosts;
        state.peers = { local: { address: '192.0.2.61' }, worker: { address: '192.0.2.60' } };
        state.report = { authority_host: 'local', replicas: state.hosts.map(host => ({ host, manifest: { epoch: 1, version: 2, keys: {} } })) };
      }
      if (req.action === 'handoff') state.report.authority_host = req.target;
      await route.fulfill({ json: state.report });
    } else await route.fulfill({ json: state });
  });
  await page.goto('/');
  await page.locator('.settings-button').click();
  await page.getByRole('tab', { name: '기능', exact: true }).click();
  await page.getByRole('button', { name: 'SSH·키', exact: true }).click();
  const sync = page.locator('.key-sync');
  await sync.locator('summary').first().click();
  await sync.getByLabel('local', { exact: true }).check();
  await sync.getByLabel('worker', { exact: true }).check();
  await sync.getByRole('button', { name: '선택한 호스트에 동기화 설정' }).click();
  await expect(sync.locator('.replicas > div')).toHaveCount(2);
  await expect(sync).toContainText('192.0.2.61');
  await sync.getByLabel('관리 권한 이전 대상').selectOption('worker');
  await sync.getByRole('button', { name: '관리 권한 이전', exact: true }).click();
  await expect(sync.locator('summary').first()).toContainText('관리: worker');
  await page.setViewportSize({ width: 390, height: 850 });
  expect(await sync.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
  await page.getByRole('button', { name: '저장', exact: true }).click();
  await expect.poll(() => saved?.runtime?.key_store_hosts).toEqual(['local', 'worker']);
  expect(saved.runtime.key_store_peers.local.address).toBe('192.0.2.61');
  extraOnline = false;
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.reload();
  await page.locator('.settings-button').click();
  await page.getByRole('tab', { name: '기능', exact: true }).click();
  await page.getByRole('button', { name: 'SSH·키', exact: true }).click();
  await expect(page.locator('.key-sync')).toBeVisible();
});
