import { expect, test } from '@playwright/test';

test('closes a conversation menu by outside click or Escape', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: '＋ 새 대화' }).click();

  const more = page.getByRole('button', { name: /메뉴$/ }).first();
  await more.click();
  await expect(page.locator('.session-menu')).toBeVisible();
  await expect(more).toHaveAttribute('aria-expanded', 'true');

  await page.locator('header').click();
  await expect(page.locator('.session-menu')).toHaveCount(0);
  await expect(more).toHaveAttribute('aria-expanded', 'false');

  await more.click();
  await expect(page.locator('.session-menu')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.locator('.session-menu')).toHaveCount(0);
});

test('searches every conversation and closes results outside the search box', async ({ page }) => {
  await page.goto('/');
  await page.request.post('/api/sessions', { data: { title: '오로라 검색 실험' } });
  await page.reload();

  const search = page.getByLabel('전체 대화 검색');
  await search.fill('오로라');
  const result = page.locator('.conversation-search-results button').filter({ hasText: '오로라 검색 실험' });
  await expect(result).toBeVisible();
  await result.click();
  await expect(page.locator('.chat-title')).toContainText('오로라 검색 실험');

  await search.fill('검색');
  await expect(page.locator('.conversation-search-results')).toBeVisible();
  await page.locator('header').click();
  await expect(page.locator('.conversation-search-results')).toHaveCount(0);

  await search.fill('오로라');
  await page.getByRole('button', { name: '검색 결과 더보기 →' }).click();
  const modal = page.getByRole('dialog', { name: '전체 대화 검색' });
  await expect(modal).toBeVisible();
  await expect(modal.getByLabel('정렬')).toHaveValue('relevance');
  await expect(modal.getByLabel('범위')).toHaveValue('all');
  await expect(modal.locator('.search-result-card').filter({ hasText: '오로라 검색 실험' })).toBeVisible();

  await page.setViewportSize({ width: 390, height: 600 });
  const box = await modal.boundingBox();
  expect(Math.round(box?.width || 0)).toBe(390);
  expect(Math.round(box?.height || 0)).toBe(600);
  await modal.locator('.search-result-card').filter({ hasText: '오로라 검색 실험' }).click();
  await expect(modal).toHaveCount(0);
  await expect(page.locator('.chat-title')).toContainText('오로라 검색 실험');
});
