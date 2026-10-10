import { expect, test } from '@playwright/test';

test('saves semantic search independently of memory recall and shows index counts', async ({ page }) => {
  await page.goto('/');
  await page.locator('.settings-button').click();
  await page.getByRole('tab', { name: '기억', exact: true }).click();
  const toggle = page.getByRole('checkbox', { name: '키워드 검색에 의미 검색 추가', exact: true });
  await expect(toggle).not.toBeChecked();
  await toggle.check();
  const saved = page.waitForResponse(r => r.url().endsWith('/api/config') && r.request().method() === 'PUT');
  await page.getByRole('button', { name: '저장', exact: true }).click();
  expect((await saved).status()).toBe(200);
  expect((await (await page.request.get('/api/config')).json()).embedding.enabled).toBe(true);
  await page.reload();
  await page.locator('.settings-button').click();
  await page.getByRole('tab', { name: '기억', exact: true }).click();
  await expect(toggle).toBeChecked();
  await page.getByRole('button', { name: '색인 상태 확인', exact: true }).click();
  await expect(page.locator('#settings-panel-memory')).toContainText(/기억 \d+\/\d+ · 대화/);
  await toggle.uncheck();
  await page.getByRole('button', { name: '저장', exact: true }).click();
});
