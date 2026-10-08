import { selectOption, controlValue, expectControlValue } from './select-helpers.js';
import { test, expect } from '@playwright/test';

test('context size presets, automatic budgets, saving and responsive fields', async ({ page, request }) => {
  const original = await (await request.get('/api/config')).json();
  try {
    await page.goto('/');
    await page.locator('.settings-button').click();
    const preset = page.getByRole('combobox', { name: '문맥 크기별 출력 프리셋', exact: true });
    const output = page.getByLabel('최대 출력 토큰 (생각 과정 포함)', { exact: true });
    const window = page.getByLabel('모델 context window (0은 백엔드 자동 감지)', { exact: true });
    const choose = label => selectOption(preset, { label });
    const oldWindow = await controlValue(window);
    await choose('256K–1M 이상 문맥 · 출력 64K');
    await expectControlValue(output, '65536');
    await expectControlValue(window, oldWindow);
    await window.fill('1048576');
    await choose('자동 · 현재 문맥 크기에 맞춤');
    await expect(output).toBeDisabled();
    await expectControlValue(output, '65536');
    await window.fill('32768');
    await expectControlValue(output, '8192');
    await page.getByRole('button', { name: '저장', exact: true }).click();
    await expect.poll(async () => (await (await request.get('/api/config')).json()).context.output_auto).toBe(true);
    await page.reload();
    await page.locator('.settings-button').click();
    await expectControlValue(preset, 'auto');
    await expectControlValue(output, '8192');
    await choose('직접 설정');
    await output.fill('12288');
    await expectControlValue(preset, 'custom');
    const rows = page.locator('.context-fields .context-row');
    const a = await rows.nth(0).boundingBox(), b = await rows.nth(1).boundingBox();
    expect(Math.abs(a.y - b.y)).toBeLessThan(2);
    expect(b.x).toBeGreaterThan(a.x);
    await page.screenshot({ path: '/tmp/talk-context-desktop.png' });
    await page.setViewportSize({ width: 390, height: 844 });
    const m = await rows.nth(0).boundingBox(), n = await rows.nth(1).boundingBox();
    expect(n.y).toBeGreaterThan(m.y);
    expect(await page.locator('.context-fields').evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
    await page.screenshot({ path: '/tmp/talk-context-mobile.png' });
  } finally {
    await request.put('/api/config', { data: original });
  }
});


test('preset popover stays outside form layout on desktop and mobile', async ({ page }) => {
  await page.goto('/'); await page.locator('.settings-button').click();
  const preset = page.getByRole('combobox', { name: '문맥 크기별 출력 프리셋', exact: true });
  for (const viewport of [{ width: 1280, height: 720 }, { width: 390, height: 640 }]) {
    await page.setViewportSize(viewport); await preset.scrollIntoViewIfNeeded();
    const before = await controlValue(preset);
    const row = await page.locator('.context-preset').boundingBox();
    const fields = await page.locator('.context-fields').boundingBox();
    await preset.click();
    await expect(preset).toHaveAttribute('aria-expanded', 'true');
    expect(await controlValue(preset)).toBe(before);
    expect(await page.locator('.context-preset').boundingBox()).toEqual(row);
    expect(await page.locator('.context-fields').boundingBox()).toEqual(fields);
    const menu = await page.getByRole('listbox').boundingBox();
    expect(menu.x).toBeGreaterThanOrEqual(0); expect(menu.x + menu.width).toBeLessThanOrEqual(viewport.width);
    await page.keyboard.press('Escape');
  }
});
