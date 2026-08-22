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
