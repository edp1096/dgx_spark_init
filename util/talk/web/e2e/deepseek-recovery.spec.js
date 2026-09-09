import { expect, test } from '@playwright/test';

test('persists DeepSeek recovery options independently in the selected set', async ({ page, request }) => {
  const original = await (await request.get('/api/config')).json();
  try {
    await page.goto('/');
    await page.locator('.settings-button').click();
    await page.getByRole('tab', { name: '시스템' }).click();
    await page.getByRole('button', { name: 'AI 세트', exact: true }).click();
    const editor = page.locator('.set-editor');
    await editor.getByLabel('편집할 세트').selectOption('ds4fve');
    const card = editor.locator('.service-card').filter({ hasText: 'DeepSeek V4' }).first();
    await card.locator(':scope > summary').click();
    await card.getByText('실행·포트 상세 설정', { exact: true }).click();
    await expect(card.getByLabel('도구 호출 형식 복구')).toHaveValue('1');
    await expect(card.getByLabel('반복 요청 캐시 보완')).toHaveValue('0');
    await card.getByLabel('도구 호출 형식 복구').selectOption('0');
    await card.getByLabel('반복 요청 캐시 보완').selectOption('1');
    await page.getByRole('button', { name: '저장', exact: true }).click();
    await expect.poll(async () => {
      const config = await (await request.get('/api/config')).json();
      return config.runtime.catalog.bundles.find(item => item.id === 'ds4fve')?.bindings?.ds4fve?.runtime_options;
    }).toMatchObject({ DSPARK_ENABLE_DSML_RECOVERY: '0', DSPARK_ENABLE_DSPARK_SWA_PREFIX: '1' });
    const saved = await (await request.get('/api/config')).json();
    const bundle = saved.runtime.catalog.bundles.find(item => item.id === 'ds4fve');
    const component = saved.runtime.catalog.components.find(item => item.id === 'ds4fve');
    const options = { ...component.runtime_options, ...bundle.bindings?.ds4fve?.runtime_options };
    expect(options.DSPARK_ENABLE_DSML_RECOVERY).toBe('0');
    expect(options.DSPARK_ENABLE_DSPARK_SWA_PREFIX).toBe('1');
    await page.reload();
    await page.locator('.settings-button').click();
    await page.getByRole('tab', { name: '시스템' }).click();
    await page.getByRole('button', { name: 'AI 세트', exact: true }).click();
    await editor.getByLabel('편집할 세트').selectOption('glm53-worker-extra');
    await expect(editor.getByLabel('도구 호출 형식 복구')).toHaveCount(0);
  } finally {
    await request.put('/api/config', { data: original });
  }
});
