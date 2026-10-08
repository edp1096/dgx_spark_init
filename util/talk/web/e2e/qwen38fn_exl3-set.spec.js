import { expect, test } from '@playwright/test';
import { expectControlValue, selectOption } from './select-helpers.js';

test('qwen38fn_exl3 is editable with its 1M profile and available in model preparation', async ({ page, request }) => {
  const original = await (await request.get('/api/config')).json();
  const native = original.runtime.catalog.bundles.find(item => item.id === 'qwen38fn_exl3');
  expect(native.context_tokens).toBe(1048576);
  expect(native.model_type).toBe('qwen38fn_exl3');
  try {
    await page.goto('/');
    await page.locator('.settings-button').click();
    await page.getByRole('tab', { name: '시스템' }).click();
    await page.getByRole('button', { name: 'AI 세트', exact: true }).click();
    const editor = page.locator('.set-editor');
    await selectOption(editor.getByLabel('편집할 세트'), native.id);
    await expect(editor.locator('.set-heading').first()).toContainText('1024K 문맥');
    await editor.getByText('모델·세트 상세 설정', { exact: true }).click();
    await expectControlValue(editor.getByLabel('모델 유형', { exact: true }), 'qwen38fn_exl3');
    await expectControlValue(editor.getByLabel('모델 ID', { exact: true }), 'qwen38fn_exl3');
    const saved = page.waitForResponse(response => response.url().endsWith('/api/config') && response.request().method() === 'PUT');
    await page.getByRole('button', { name: '저장', exact: true }).click();
    expect((await saved).ok()).toBeTruthy();
    await expect.poll(async () => (await (await request.get('/api/config')).json()).runtime.catalog.bundles.find(item => item.id === native.id)?.model_type).toBe('qwen38fn_exl3');
    await page.getByRole('button', { name: '모델 준비', exact: true }).click();
    await selectOption(page.getByLabel('모델', { exact: true }), native.id);
    await expect(page.getByRole('link', { name: /alesha-pro\/Huihui-Qwen3.8-Flash-Next-abliterated-exl3-3bit-hq_h6_ng6/ })).toBeVisible();
  } finally {
    await request.put('/api/config', { data: original });
  }
});
