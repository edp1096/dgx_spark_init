import { expect, test } from '@playwright/test';

test('keeps settings values while navigating categorized tabs', async ({ page }) => {
  await page.goto('/');
  await page.locator('.settings-button').click();

  const tabs = page.getByRole('tab');
  await expect(tabs).toHaveCount(5);
  await expect(page.getByRole('tab', { name: '모델' })).toHaveAttribute('aria-selected', 'true');

  const endpoint = page.getByLabel('API endpoint', { exact: true });
  await endpoint.fill('http://example.test:8000');
  await page.getByRole('tab', { name: '음성' }).click();
  await expect(page.getByText('음성 인식', { exact: true })).toBeVisible();
  await page.getByRole('tab', { name: '모델' }).click();
  await expect(endpoint).toHaveValue('http://example.test:8000');

  const initialModal = await page.locator('.settings-modal').boundingBox();
  const initialActions = await page.locator('.modal-actions').boundingBox();
  for (const name of ['음성', '외형', '도구', '앱·저장']) {
    await page.getByRole('tab', { name }).click();
    const modal = await page.locator('.settings-modal').boundingBox();
    const actions = await page.locator('.modal-actions').boundingBox();
    expect(modal?.height).toBe(initialModal?.height);
    expect(actions?.y).toBe(initialActions?.y);
  }
});

test('keeps tab navigation and actions reachable on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 600 });
  await page.goto('/');
  await page.getByRole('button', { name: '사이드바 열기 또는 닫기' }).click();
  await page.locator('.settings-button').click();

  const modelTab = page.getByRole('tab', { name: '모델' });
  await modelTab.focus();
  await modelTab.press('End');
  await expect(page.getByRole('tab', { name: '앱·저장' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('button', { name: '저장' })).toBeVisible();

  const modal = await page.locator('.settings-modal').boundingBox();
  expect((modal?.y ?? -1)).toBeGreaterThanOrEqual(0);
  expect((modal?.y ?? 601) + (modal?.height ?? 0)).toBeLessThanOrEqual(600);
});
