import { expect, test } from '@playwright/test';

test('EXL3 offers native effort in chat and saves the default level', async ({ page }) => {
  let initial = true;
  await page.route('**/api/config', async route => {
    if (route.request().method() !== 'GET') return route.continue();
    const response = await route.fetch();
    const config = await response.json();
    config.model.model_type = 'qwen3.8-exl3';
    if (initial) config.model.reasoning_effort = 'on';
    await route.fulfill({ json: config });
  });
  await page.goto('/');
  const slider = page.locator('.model-controls').getByRole('slider', { name: 'Reasoning effort' });
  await expect(slider).toHaveValue('3');
  for (const [index, label] of ['꺼짐', '낮음', '중간', '매우 높음'].entries()) {
    await slider.fill(String(index));
    await expect(slider).toHaveAttribute('aria-valuetext', label);
  }
  await page.locator('.settings-button').click();
  const defaults = page.getByLabel('기본 reasoning effort', { exact: true });
  await expect(defaults.locator('option')).toHaveText(['꺼짐', '낮음', '중간', '매우 높음']);
  await defaults.selectOption('low');
  const saved = page.waitForResponse(response => response.url().endsWith('/api/config') && response.request().method() === 'PUT');
  await page.getByRole('button', { name: '저장', exact: true }).click();
  const response = await saved;
  expect(response.ok()).toBeTruthy();
  expect((await response.json()).config.model.reasoning_effort).toBe('low');
  initial = false;
  await page.reload();
  await expect(slider).toHaveValue('1');
});
