import { test, expect } from '@playwright/test';

test('long current AI set fits mobile width and selected options have a tint', async ({ page }) => {
  await page.route('**/api/config', async route => {
    const response = await route.fetch();
    const config = await response.json();
    config.runtime.mode = 'managed';
    config.runtime.bundle = 'flash-next-tp2';
    config.model.default_model = 'edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4';
    await route.fulfill({ response, json: config });
  });
  await page.route('**/api/runtime', async route => {
    const response = await route.fetch();
    const runtime = await response.json();
    runtime.selected_bundle = 'flash-next-tp2';
    runtime.bundles = [{ id: 'flash-next-tp2', name: 'Huihui-RadixArk Qwen3.8 Flash-Next TP2', description: '2× DGX Spark · 1M 문맥 · BF16 KV · 이미지·ASR·TTS·Extra', model_id: 'edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4', context_tokens: 1048576 }];
    await route.fulfill({ response, json: runtime });
  });
  await page.goto('/');
  await page.locator('.settings-button').click();
  const card = page.locator('.settings-bundle-card');
  await expect(card).toContainText('1024K context');
  for (const width of [320, 390, 600]) {
    await page.setViewportSize({ width, height: 844 });
    await card.scrollIntoViewIfNeeded();
    const title = await card.locator(':scope > span').boundingBox();
    const detail = await card.locator(':scope > div').boundingBox();
    expect(detail.y).toBeGreaterThanOrEqual(title.y + title.height);
    expect(Math.abs(title.x - detail.x)).toBeLessThan(1);
    expect(await card.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
    expect(await card.locator('b').evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: '/tmp/talk-bundle-mobile.png' });
  const select = page.getByRole('combobox', { name: '기본 reasoning effort', exact: true });
  for (const theme of ['dark', 'light']) {
    await page.evaluate(theme => document.documentElement.dataset.theme = theme, theme);
    await select.click();
    const selected = page.getByRole('option', { selected: true });
    const other = page.getByRole('option', { selected: false }).first();
    expect(await selected.evaluate(el => getComputedStyle(el).backgroundColor)).not.toBe(await other.evaluate(el => getComputedStyle(el).backgroundColor));
    await expect(selected).not.toBeDisabled();
    await page.keyboard.press('Escape');
  }
});
