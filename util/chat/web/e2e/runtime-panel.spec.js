import { expect, test } from '@playwright/test';

const runtimeSnapshot = {
  selected_bundle: 'flash-next',
  bundles: [
    { id: 'qwen27', name: 'Qwen 27B 세트', memory_gib: 61.4, components: ['qwen27', 'flux2'] },
    { id: 'flash-next', name: 'Flash-Next 세트', description: '64K 문맥과 Flash-Next를 사용하는 고성능 세트', memory_gib: 99.1, components: ['flash-next', 'flux2'] },
    { id: 'gemma', name: 'Gemma 세트', memory_gib: 67.4, components: ['gemma31', 'flux2'] },
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
  await expect(panel.getByText('Flash-Next 세트', { exact: true })).toBeVisible();
  await expect(panel.getByText('통합메모리', { exact: true })).toBeVisible();
  await expect(panel.getByText('시스템 가용 13.5 GiB · 즉시 여유 3.2 GiB', { exact: true })).toBeVisible();
  await expect(panel.getByText('Qwen3.8 Flash-Next', { exact: true })).toBeVisible();
  await expect(panel.getByRole('combobox', { name: '전환할 AI 세트' })).toHaveValue('flash-next');
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
  await expect(panel.getByRole('list', { name: '기동 단계' })).toContainText('vLLM 엔진 구성');
  await expect(panel.getByText('예상 00:20 남음', { exact: true })).toBeVisible();
  await expect(panel.getByRole('button', { name: '기동 중' })).toBeDisabled();
});

test('keeps managed model controls reachable by touch scrolling on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 500 });
  await routeManagedRuntime(page, runtimeSnapshot);
  await page.goto('/');
  await page.getByRole('button', { name: '모델 및 대화 설정' }).click();

  const drawer = page.getByRole('dialog', { name: '모델 및 대화 설정' });
  await drawer.locator('.drawer-status > button').click();
  const stop = drawer.getByRole('button', { name: '중지' });

  await expect(drawer).toHaveCSS('overflow-y', 'auto');
  expect(await drawer.evaluate((node) => node.scrollHeight)).toBeGreaterThan(
    await drawer.evaluate((node) => node.clientHeight),
  );
  await stop.scrollIntoViewIfNeeded();
  await expect(stop).toBeVisible();
  expect(await drawer.evaluate((node) => node.scrollTop)).toBeGreaterThan(0);
});
