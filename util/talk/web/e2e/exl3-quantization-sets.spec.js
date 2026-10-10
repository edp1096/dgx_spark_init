import { expect, test } from '@playwright/test';
import { optionTexts, selectOption } from './select-helpers.js';

test('EXL3 3bit and 4bit remain distinct and Q4 precedes NVFP4', async ({ page, request }) => {
  const config = await (await request.get('/api/config')).json();
  const bundles = config.runtime.catalog.bundles;
  const ids = bundles.map(b => b.id);
  const i = ids.indexOf('qwen38fn_exl3');
  expect(ids.slice(i, i + 3)).toEqual(['qwen38fn_exl3', 'qwen38fn_exl3_q4', 'flash-next-radixark']);
  expect(bundles[i].name).toBe('Qwen 3.8 Flash-Next EXL3 3bit');
  expect(bundles[i + 1].name).toBe('Qwen 3.8 Flash-Next EXL3 4bit');
  expect(bundles[i + 1].context_tokens).toBe(1048576);
  await page.route('**/api/config', route => route.fulfill({ json: { ...config, runtime: { ...config.runtime, mode: 'managed' } } }));
  await page.route('**/api/runtime', route => route.fulfill({ json: { selected_bundle: 'qwen38fn_exl3', bundles, components: [], support_services: [], memory: { total_gib: 121.6, available_gib: 117, free_gib: 112 }, hosts: {}, operation: {}, docker: 'online' } }));
  await page.goto('/');
  await page.locator('.connection-menu .status').click();
  const options = await optionTexts(page.getByLabel('전환할 AI 세트', { exact: true }));
  const q3 = options.indexOf('Qwen 3.8 Flash-Next EXL3 3bit');
  expect(options.slice(q3, q3 + 3)).toEqual(['Qwen 3.8 Flash-Next EXL3 3bit', 'Qwen 3.8 Flash-Next EXL3 4bit', 'Qwen 3.8 Flash-Next NVFP4']);
  await page.locator('.connection-menu .status').click();
  await page.locator('.settings-button').click();
  await page.getByRole('tab', { name: '시스템' }).click();
  await page.getByRole('button', { name: '모델 준비', exact: true }).click();
  await selectOption(page.getByLabel('모델', { exact: true }), 'qwen38fn_exl3_q4');
  await expect(page.getByRole('link', { name: /alesha-pro\/Huihui-Qwen3.8-Flash-Next-abliterated-exl3-4bit-hq_h6_ng6/ })).toBeVisible();
});
