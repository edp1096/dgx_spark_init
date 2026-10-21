import { expect, test } from '@playwright/test';
import { optionTexts } from './select-helpers.js';

test('EXL3 3bit remains available and retired 4bit is absent', async ({ page, request }) => {
  const config = await (await request.get('/api/config')).json();
  const bundles = config.runtime.catalog.bundles;
  const ids = bundles.map(b => b.id);
  const i = ids.indexOf('qwen38fn_exl3');
  expect(ids).not.toContain('qwen38fn_exl3_q4');
  expect(ids).not.toContain('flash-next-radixark');
  expect(bundles.find(b => b.id === 'glm53-worker-extra').name).toBe('GLM 5.3 Flash FP4-Local');
  const glm = config.runtime.catalog.components.find(c => c.id === 'glm53');
  expect(glm.controller).toBe('glm53-fp4-cluster');
  expect(glm.runtime_options).toMatchObject({ MTP_TOKENS: '5', MAX_MODEL_LEN: '1048576' });
  expect(ids.slice(i, i + 2)).toEqual(['qwen38fn_exl3', 'flash-next-fp4-local']);
  expect(bundles[i].name).toBe('Qwen 3.8 Flash-Next EXL3 3bit');
  await page.route('**/api/config', route => route.fulfill({ json: { ...config, runtime: { ...config.runtime, mode: 'managed' } } }));
  await page.route('**/api/runtime', route => route.fulfill({ json: { selected_bundle: 'qwen38fn_exl3', bundles, components: [], support_services: [], memory: { total_gib: 121.6, available_gib: 117, free_gib: 112 }, hosts: {}, operation: {}, docker: 'online' } }));
  await page.goto('/');
  await page.locator('.connection-menu .status').click();
  const options = await optionTexts(page.getByLabel('전환할 AI 세트', { exact: true }));
  const q3 = options.indexOf('Qwen 3.8 Flash-Next EXL3 3bit');
  expect(options).not.toContain('Qwen 3.8 Flash-Next EXL3 4bit');
  expect(options.slice(q3, q3 + 2)).toEqual(['Qwen 3.8 Flash-Next EXL3 3bit', 'Qwen 3.8 Flash-Next FP4-Local']);
});
