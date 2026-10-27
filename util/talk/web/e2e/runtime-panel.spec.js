import { expectControlValue } from './select-helpers.js';
import { expect, test } from '@playwright/test';

const runtimeSnapshot = {
  selected_bundle: 'flash-next',
  bundles: [
    { id: 'qwen27', name: 'Qwen 27B 세트', memory_gib: 61.4, components: ['qwen27', 'flux2'] },
    { id: 'flash-next', name: 'Flash-Next 세트', description: '64K 문맥과 Flash-Next를 사용하는 고성능 세트', memory_gib: 99.1, components: ['flash-next', 'flux2'] },
    { id: 'ornith35', name: 'Ornith 세트', memory_gib: 67.4, components: ['ornith35', 'flux2'] },
  ],
  components: [
    { id: 'flash-next', name: 'Qwen3.8 Flash-Next', role: 'llm', model: 'Qwen3.8-Flash-Next', status: 'running', health: 'online', gpu_memory_gib: 90.7 },
    { id: 'flux2', name: 'FLUX.2 Klein 4B', role: 'image', status: 'running', health: 'online', gpu_memory_gib: 6.5 },
  ],
  memory: { total_gib: 121.6, used_gib: 108.1, available_gib: 13.5, free_gib: 3.2 },
  operation: {},
  docker: 'online',
};

async function routeManagedRuntime(page, snapshot) {
  await page.route('**/api/config', async (route) => {
    const response = await route.fetch();
    const config = await response.json();
    config.runtime = { ...config.runtime, mode: 'managed' };
    await route.fulfill({ response, json: config });
  });
  await page.route('**/api/runtime', (route) => route.fulfill({ json: snapshot }));
}

test('shows managed set, memory, engines, and set controls from the connection button', async ({ page }) => {
  await routeManagedRuntime(page, runtimeSnapshot);
  await page.goto('/');
  await page.locator('.status').click();

  const panel = page.getByRole('region', { name: 'DGX Spark 운영' });
  await expect(panel).toBeVisible();
  await expect(panel.getByRole('strong')).toHaveText('Flash-Next 세트');
  await expect(panel.getByText('시스템 전체 통합메모리', { exact: true })).toBeVisible();
  await expect(panel.getByText('시스템 가용 13.5 GiB · 즉시 여유 3.2 GiB', { exact: true })).toBeVisible();
  await expect(panel.getByText('Qwen3.8 Flash-Next', { exact: true })).toBeVisible();
  await expectControlValue(panel.getByRole('combobox', { name: '전환할 AI 세트' }), 'flash-next');
  await expect(panel.getByRole('button', { name: '실행 중' })).toBeDisabled();
});

test('shows runtime-specific startup stages without duplicating shard updates', async ({ page }) => {
  const running = structuredClone(runtimeSnapshot);
  running.components[0].health = 'starting';
  running.components[0].phase = 'Flash Next 체크포인트 적재';
  running.operation = {
    action: 'start',
    bundle_id: 'flash-next',
    component_id: 'flash-next',
    state: 'running',
    phase: 'Flash Next 체크포인트 적재',
    detail: '118/206 샤드 · SSD에서 통합메모리로 읽는 중',
    progress: 0.49,
    eta: '00:20',
    started_at: new Date(Date.now() - 120_000).toISOString(),
    steps: [
      { phase: '기동 계획 준비', state: 'complete', started_at: new Date().toISOString() },
      { component_id: 'flash-next', phase: 'vLLM 엔진 구성', state: 'complete', started_at: new Date().toISOString() },
      { component_id: 'flash-next', phase: 'Flash Next 체크포인트 적재', detail: '118/206 샤드 · SSD에서 통합메모리로 읽는 중', state: 'current', started_at: new Date().toISOString() },
    ],
  };
  await routeManagedRuntime(page, running);
  await page.goto('/');
  await page.locator('.status').click();

  const panel = page.getByRole('region', { name: 'DGX Spark 운영' });
  await expect(panel.getByText('Flash Next 체크포인트 적재', { exact: true }).first()).toBeVisible();
  await expect(panel.getByText('118/206 샤드 · SSD에서 통합메모리로 읽는 중').first()).toBeVisible();
  await expect(panel.getByRole('list', { name: '진행 단계' })).toContainText('vLLM 엔진 구성');
  await expect(panel.getByText('예상 00:20 남음', { exact: true })).toBeVisible();
  await expect(panel.getByRole('button', { name: '기동 중' })).toBeDisabled();
});

test('keeps managed model controls reachable by touch scrolling on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 300 });
  await routeManagedRuntime(page, runtimeSnapshot);
  await page.goto('/');
  await page.locator('.status').click();

  const drawer = page.getByRole('dialog', { name: 'DGX Spark 운영 상태' });
  const stop = drawer.getByRole('button', { name: '모델 세트 중지', exact: true });

  await expect(drawer).toHaveCSS('overflow-y', 'auto');
  expect(await drawer.evaluate((node) => node.scrollHeight)).toBeGreaterThan(
    await drawer.evaluate((node) => node.clientHeight),
  );
  await stop.scrollIntoViewIfNeeded();
  await expect(stop).toBeVisible();
  expect(await drawer.evaluate((node) => node.scrollTop)).toBeGreaterThan(0);
});

test('shared Extra occupancy is visible on mobile without changing model startup readiness', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 640 });
  const snapshot = structuredClone(runtimeSnapshot);
  snapshot.support_services = [
    { id: 'extra-media', name: 'Extra Media', role: 'media', host: 'local', controller: 'compose', status: 'running', health: 'online', memory_measured: true, resident_memory_gib: 40 / 1024, host_memory_gib: 40 / 1024, gpu_memory_gib: 0, memory_gib: .1 },
    { id: 'extra-collector', name: 'Extra Collector', role: 'collector', host: 'worker', controller: 'compose', status: 'running', health: 'online', memory_measured: false, resident_memory_gib: 0, memory_gib: .3 },
    { id: 'extra-documents', name: 'Extra Documents', role: 'documents', host: 'local', controller: 'compose', status: 'exited', health: 'offline', memory_measured: false, memory_gib: .4 },
  ];
  await routeManagedRuntime(page, snapshot);
  await page.goto('/');
  await page.locator('.status').click();
  const panel = page.getByRole('region', { name: 'DGX Spark 운영' });
  const extras = panel.getByRole('region', { name: 'Extra 서비스' });
  await expect(extras.getByRole('group')).toHaveCount(3);
  const media = extras.getByRole('group', { name: 'Extra Media', exact: true });
  await expect(media).toContainText('점유 40.0 MiB');
  await expect(media).toContainText('예산 102.4 MiB');
  await expect(extras.getByRole('group', { name: 'Extra Collector', exact: true })).toContainText('점유 측정 불가');
  await expect(extras.getByRole('group', { name: 'Extra Documents', exact: true })).toContainText('점유 0.0 MiB');
  await expect(panel.getByRole('button', { name: '실행 중', exact: true })).toBeDisabled();
  const drawer = page.getByRole('dialog', { name: 'DGX Spark 운영 상태' });
  expect(await drawer.evaluate(node => node.scrollWidth <= node.clientWidth)).toBe(true);
});

test('groups music and speech models together and keeps tools in Extra', async ({ page }) => {
  const snapshot = structuredClone(runtimeSnapshot);
  snapshot.support_services = [
    { id: 'extra-fish', name: 'Fish Drama3 KoEn160', role: 'tts', status: 'exited' },
    { id: 'extra-music-ace', name: 'ACE-Step 1.5', role: 'tool', status: 'exited' },
    { id: 'extra-ssh', name: 'Extra SSH', role: 'tool', status: 'running', health: 'online' },
  ];
  await routeManagedRuntime(page, snapshot);
  await page.goto('/');
  await page.locator('.status').click();
  const models = page.getByRole('group', { name: '모델', exact: true });
  await expect(models.getByRole('group', { name: 'ACE-Step 1.5', exact: true })).toContainText('음악 생성');
  await expect(models.getByRole('group', { name: 'ACE-Step 1.5', exact: true })).toContainText('요청 시 실행');
  await expect(models.getByRole('group', { name: 'Fish Drama3 KoEn160', exact: true })).toContainText('음성 합성');
  const extra = page.getByRole('region', { name: 'Extra 서비스', exact: true });
  await expect(extra).toContainText('Extra SSH');
  await expect(extra).not.toContainText('ACE-Step');
  await expect(extra).not.toContainText('Fish');
});

for (const bundleID of ['flash-next-fp4-local', 'flash-next-tp2', 'glm53-worker-extra', 'ds41']) {
  test(`excluded audio stays hidden for ${bundleID}`, async ({ page }) => {
    const snapshot = structuredClone(runtimeSnapshot);
    snapshot.selected_bundle = bundleID;
    snapshot.bundles = [{ ...snapshot.bundles[1], id: bundleID, disabled_support: ['extra-fish', 'extra-music-ace'] }];
    snapshot.support_services = [
      { id: 'extra-fish', name: 'Fish Drama3 KoEn160', role: 'tts', status: 'exited' },
      { id: 'extra-music-ace', name: 'ACE-Step 1.5', role: 'tool', status: 'exited' },
      { id: 'extra-ssh', name: 'Extra SSH', role: 'tool', status: 'running', health: 'online' },
    ];
    await routeManagedRuntime(page, snapshot);
    await page.goto('/');
    await page.locator('.status').click();
    const panel = page.getByRole('region', { name: 'DGX Spark 운영' });
    await expect(panel.getByText('메모리 확보를 위해 Fish 음성·ACE-Step 음악 생성 제외')).toBeVisible();
    await expect(panel.getByRole('group', { name: 'ACE-Step 1.5', exact: true })).toHaveCount(0);
    await expect(panel.getByRole('group', { name: 'Fish Drama3 KoEn160', exact: true })).toHaveCount(0);
    await expect(panel.getByRole('region', { name: 'Extra 서비스' })).toContainText('Extra SSH');
  });
}


test('keeps EmbeddingGemma in models despite its internal tool role', async ({ page }) => {
  const snapshot = structuredClone(runtimeSnapshot);
  snapshot.bundles[1].components.push('extra-embedding');
  snapshot.components.push({ id: 'extra-embedding', name: 'EmbeddingGemma 2', role: 'tool', status: 'exited' });
  snapshot.support_services = [{ id: 'extra-ssh', name: 'Extra SSH', role: 'tool', status: 'running', health: 'online' }];
  await routeManagedRuntime(page, snapshot);
  await page.goto('/');
  await page.locator('.status').click();
  await expect(page.getByRole('group', { name: '모델', exact: true }).getByRole('group', { name: 'EmbeddingGemma 2', exact: true })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Extra 서비스', exact: true })).not.toContainText('EmbeddingGemma');
});

test('a stopped set shows Extra workers stopped without online lights', async ({ page }) => {
  const snapshot = structuredClone(runtimeSnapshot);
  snapshot.bundles[1].workload_swap = true;
  snapshot.components = snapshot.components.map(c => ({ ...c, status: 'exited', health: 'offline' }));
  snapshot.support_services = [
    { id: 'extra-shorts', name: 'Extra Shorts', role: 'tool', status: 'exited', health: 'offline' },
    { id: 'extra-fish', name: 'Fish Drama3 KoEn160', role: 'tts', status: 'exited', health: 'offline' },
  ];
  await routeManagedRuntime(page, snapshot);
  await page.goto('/');
  await page.locator('.status').click();
  const panel = page.getByRole('region', { name: 'DGX Spark 운영' });
  await expect(panel.getByRole('group', { name: 'Extra Shorts', exact: true })).toContainText('중지됨');
  await expect(panel.getByRole('group', { name: 'Fish Drama3 KoEn160', exact: true })).toContainText('중지됨');
  await expect(panel.locator('.runtime-component > span.online')).toHaveCount(0);
});

test('model status stays running during embedding work and distinguishes startup', async ({ page }) => {
  const snapshot = structuredClone(runtimeSnapshot);
  snapshot.operation = { action: 'workload', state: 'running', bundle_id: 'flash-next', component_id: 'extra-embedding', phase: '임베딩 작업' };
  await routeManagedRuntime(page, snapshot);
  await page.goto('/');
  await expect(page.locator('.chat-header .status')).toHaveText('● 실행 중');
  await expect(page.locator('.chat-header .status')).not.toHaveClass(/starting/);
  snapshot.operation = { action: 'start', state: 'running', bundle_id: 'flash-next', component_id: 'flash-next', phase: '모델 적재' };
  snapshot.components[0].health = 'starting';
  await page.reload();
  await expect(page.locator('.chat-header .status')).toHaveText('● 기동 중');
  snapshot.operation = { action: 'stop', state: 'running', bundle_id: 'flash-next' };
  await page.reload();
  await expect(page.locator('.chat-header .status')).toHaveText('● 중지 중');
  snapshot.operation = { action: 'stop', state: 'complete', bundle_id: 'flash-next' };
  snapshot.components[0].health = 'offline'; snapshot.components[0].status = 'exited';
  await page.reload();
  await expect(page.locator('.chat-header .status')).toHaveText('● 중지됨');
});
