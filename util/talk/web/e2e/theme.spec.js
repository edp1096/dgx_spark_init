import { expect, test } from '@playwright/test';

test('saves light theme and restores system theme', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.goto('/');
  await page.locator('.settings-button').click();
  await page.getByRole('tab', { name: '외형' }).click();

  await page.getByRole('radio', { name: '라이트' }).click();
  await page.getByRole('button', { name: '저장' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await expect(page.locator('html')).toHaveAttribute('data-theme-preference', 'light');

  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');

  await page.locator('.settings-button').click();
  await page.getByRole('tab', { name: '외형' }).click();
  await page.getByRole('radio', { name: '시스템 설정 따름' }).click();
  await page.getByRole('button', { name: '저장' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await expect(page.locator('html')).toHaveAttribute('data-theme-preference', 'system');
});
