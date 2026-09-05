import { expect, test } from '@playwright/test';

test('edits and persists a set endpoint without resetting it to localhost', async ({ page, request }) => {
  const original = await (await request.get('/api/config')).json();
  try {
    await page.goto('/');
    await page.locator('.settings-button').click();
    await page.getByRole('tab', { name: '시스템' }).click();
  await page.getByRole('button', { name: 'AI 세트', exact: true }).click();
    const editor = page.locator('.set-editor');
    await expect(editor).toBeVisible();
    await editor.getByLabel('편집할 세트').selectOption('qwen27');
    await editor.getByRole('button', { name: '세트 복제', exact: true }).click();
    await editor.getByLabel('세트 이름', { exact: true }).fill('원격 주소 테스트');
    await editor.getByText('모델·세트 상세 설정', { exact: true }).click();
    await editor.getByLabel('세트 ID', { exact: true }).fill('editable-remote');
    await editor.getByLabel('세트 ID', { exact: true }).blur();
    await expect(editor.getByLabel('편집할 세트')).toHaveValue('editable-remote');
    const service = editor.locator('.service-card').filter({ hasText: 'Qwen3.8 27B' }).first();
    await service.locator(':scope > summary').click();
    await service.getByLabel('API 주소', { exact: true }).fill('http://127.0.0.1:18000');
    await page.getByRole('button', { name: '저장', exact: true }).click();
    await expect.poll(async () => {
      const config = await (await request.get('/api/config')).json();
      return config.runtime.catalog.bundles.find(b => b.id === 'editable-remote')?.bindings?.qwen27?.endpoint;
    }).toBe('http://127.0.0.1:18000');
    const config = await (await request.get('/api/config')).json();
    expect(config.runtime.catalog.bundles.some(b => b.name === '원격 주소 테스트')).toBeTruthy();
    expect(config.runtime.catalog.components.find(c => c.id === 'qwen27').endpoint).toBe(original.runtime.catalog.components.find(c => c.id === 'qwen27').endpoint);
  } finally { await request.put('/api/config', { data: original }); }
});

test('imports YAML and reports invalid references without replacing the editor', async ({ page }) => {
  await page.goto('/');
  await page.locator('.settings-button').click();
  await page.getByRole('tab', { name: '시스템' }).click();
  await page.getByRole('button', { name: 'AI 세트', exact: true }).click();
  const editor = page.locator('.set-editor');
  await editor.getByText('JSON / YAML 가져오기·내보내기', { exact: true }).click();
  await editor.getByLabel('전체 세트 정의').fill(`hosts:
  local: {}
components:
  - id: remote-llm
    name: Remote LLM
    role: llm
    host: local
    controller: external
    endpoint: http://127.0.0.1:9
    health_url: http://127.0.0.1:9/health
bundles:
  - id: imported
    name: YAML 세트
    model_type: generic
    model_id: model
    components: [remote-llm]
`);
  await editor.getByRole('button', { name: '검증 후 불러오기' }).click();
  await expect(editor.getByLabel('세트 이름', { exact: true })).toHaveValue('YAML 세트');
  await editor.getByLabel('전체 세트 정의').fill('hosts: {}\ncomponents: []\nbundles: [{id: bad, name: bad, components: [missing]}]');
  await editor.getByRole('button', { name: '검증 후 불러오기' }).click();
  await expect(editor.getByRole('status')).toContainText('unknown');
  await expect(editor.getByLabel('세트 이름', { exact: true })).toHaveValue('YAML 세트');
});


test('shows only selected set members and preserves edits across system sections', async ({ page, request }) => {
  const config = await (await request.get('/api/config')).json();
  await page.goto('/');
  await page.locator('.settings-button').click();
  await page.getByRole('tab', { name: '시스템' }).click();
  await expect(page.locator('.set-editor')).not.toBeVisible();
  await page.getByRole('button', { name: 'AI 세트', exact: true }).click();
  const editor = page.locator('.set-editor');
  await expect(editor.getByLabel('편집할 세트')).toHaveValue(config.runtime.bundle);
  await editor.getByLabel('편집할 세트').selectOption('glm53-worker-extra');
  await expect(editor.locator('.service-card')).toHaveCount(4);
  const card = editor.locator('.service-card').filter({ hasText: 'Extra Collector' });
  await card.locator(':scope > summary').click();
  await expect(card.getByLabel('공통 Compose 레시피')).not.toBeVisible();
  await card.getByLabel('API 주소', { exact: true }).fill('http://worker:18695');
  await page.getByRole('button', { name: '앱·저장소', exact: true }).click();
  await expect(page.getByLabel('SQLite 파일')).toBeVisible();
  await page.getByRole('button', { name: 'AI 세트', exact: true }).click();
  await expect(card.getByLabel('API 주소', { exact: true })).toHaveValue('http://worker:18695');
  await page.setViewportSize({ width: 390, height: 700 });
  await expect(page.getByRole('button', { name: '저장', exact: true })).toBeInViewport();
  expect(await page.locator('.settings-modal').evaluate(el => el.scrollWidth <= el.clientWidth)).toBeTruthy();
});

test('uses one Extra definition with independent local and worker bindings', async ({ page }, testInfo) => {
  await page.goto('/');
  await page.locator('.settings-button').click();
  await page.getByRole('tab', { name: '시스템' }).click();
  await page.getByRole('button', { name: 'AI 세트', exact: true }).click();
  const editor = page.locator('.set-editor');
  await editor.getByLabel('편집할 세트').selectOption('glm53-worker-extra');
  let collector = editor.locator('.service-card').filter({ hasText: 'Extra Collector' });
  await collector.locator(':scope > summary').click();
  await expect(collector.getByRole('combobox', { name: '실행 호스트', exact: true })).toHaveValue('worker');
  await collector.getByLabel('API 주소', { exact: true }).fill('http://worker:19695');
  await page.route('**/api/runtime/probe', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'worker-probe-result' }) }));
  await collector.getByRole('button', { name: 'API 연결 시험', exact: true }).click();
  await expect(collector.getByRole('status')).toHaveText('worker-probe-result');
  await collector.scrollIntoViewIfNeeded();
  await page.screenshot({ path: testInfo.outputPath('glm-binding.png') });
  await editor.getByLabel('편집할 세트').selectOption('qwen27');
  collector = editor.locator('.service-card').filter({ hasText: 'Extra Collector' });
  // Keyed shared cards may remain open when switching sets.
  if ((await collector.getAttribute('open')) === null) await collector.locator(':scope > summary').click();
  await expect(collector.getByRole('status')).toHaveCount(0);
  await expect(collector.getByRole('combobox', { name: '실행 호스트', exact: true })).toHaveValue('local');
  await expect(collector.getByLabel('API 주소', { exact: true })).toHaveValue('http://127.0.0.1:8695');
  await collector.getByRole('combobox', { name: '실행 호스트', exact: true }).selectOption('worker');
  await collector.getByLabel('API 주소', { exact: true }).fill('http://worker:20695');
  await editor.getByLabel('편집할 세트').selectOption('glm53-worker-extra');
  await expect(collector.getByLabel('API 주소', { exact: true })).toHaveValue('http://worker:19695');
  await collector.getByRole('button', { name: '기본 배치로 되돌리기', exact: true }).click();
  await expect(collector.getByRole('combobox', { name: '실행 호스트', exact: true })).toHaveValue('local');
  await expect(collector.getByLabel('API 주소', { exact: true })).toHaveValue('http://127.0.0.1:8695');
  await editor.getByLabel('편집할 세트').selectOption('qwen27');
  await expect(collector.getByLabel('API 주소', { exact: true })).toHaveValue('http://worker:20695');
  await editor.getByRole('button', { name: '서비스 선택', exact: true }).click();
  const picker = editor.locator('.service-picker');
  await expect(picker.getByRole('checkbox', { name: /^Extra Collector/ })).toHaveCount(1);
  await expect(picker.getByText(/Worker Extra/)).toHaveCount(0);
});
