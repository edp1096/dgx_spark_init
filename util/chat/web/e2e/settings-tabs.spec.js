import { expect, test } from '@playwright/test';

test('keeps settings values while navigating categorized tabs', async ({ page }) => {
  await page.goto('/');
  await page.locator('.settings-button').click();

  const tabs = page.getByRole('tab');
  await expect(tabs).toHaveCount(5);
  await expect(page.getByRole('tab', { name: '대화' })).toHaveAttribute('aria-selected', 'true');

  const effort = page.getByLabel('기본 reasoning effort', { exact: true });
  await effort.selectOption('xhigh');
  await page.getByRole('tab', { name: '음성' }).click();
  await expect(page.getByText('음성 인식', { exact: true })).toBeVisible();
  const omitParentheticals = page.getByLabel('괄호 속 부연설명 읽지 않기');
  await expect(omitParentheticals).toBeChecked();
  await omitParentheticals.uncheck();
  await page.getByRole('tab', { name: '대화' }).click();
  await expect(effort).toHaveValue('xhigh');
  await page.getByRole('tab', { name: '음성' }).click();
  await expect(omitParentheticals).not.toBeChecked();
  await page.getByRole('tab', { name: '대화' }).click();

  const initialModal = await page.locator('.settings-modal').boundingBox();
  const initialActions = await page.locator('.modal-actions').boundingBox();
  for (const name of ['음성', '기능', '외형', '시스템']) {
    await page.getByRole('tab', { name }).click();
    const modal = await page.locator('.settings-modal').boundingBox();
    const actions = await page.locator('.modal-actions').boundingBox();
    expect(modal?.height).toBe(initialModal?.height);
    expect(actions?.y).toBe(initialActions?.y);
  }
});

test('changes Qwen reasoning with the compact header slider', async ({ page }) => {
  await page.goto('/');
  const effort = page.getByRole('slider', { name: 'Reasoning effort' });
  await expect(effort).toHaveValue('2');
  await effort.fill('3');
  await expect(page.locator('.model-controls .qwen-effort-control output')).toHaveText('XHigh');
  await effort.fill('0');
  await expect(page.locator('.model-controls .qwen-effort-control output')).toHaveText('꺼짐');
});

test('keeps tab navigation and actions reachable on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 600 });
  await page.goto('/');
  await page.getByRole('button', { name: '사이드바 열기 또는 닫기' }).click();
  await page.locator('.settings-button').click();

  const chatTab = page.getByRole('tab', { name: '대화' });
  await chatTab.focus();
  await chatTab.press('End');
  await expect(page.getByRole('tab', { name: '시스템' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('button', { name: '저장' })).toBeVisible();

  const modal = await page.locator('.settings-modal').boundingBox();
  expect((modal?.y ?? -1)).toBeGreaterThanOrEqual(0);
  expect((modal?.y ?? 601) + (modal?.height ?? 0)).toBeLessThanOrEqual(600);
});
