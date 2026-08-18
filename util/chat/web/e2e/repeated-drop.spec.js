import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect, test } from '@playwright/test';

const fixtureBase64 = readFileSync(resolve(import.meta.dirname, '../public/avatars/spark.png')).toString('base64');

async function dropPNG(page, name) {
  const dataTransfer = await page.evaluateHandle(({ base64, fileName }) => {
    const bytes = Uint8Array.from(atob(base64), (character) => character.charCodeAt(0));
    const transfer = new DataTransfer();
    transfer.items.add(new File([bytes], fileName, { type: 'image/png' }));
    return transfer;
  }, { base64: fixtureBase64, fileName: name });

  await page.dispatchEvent('body', 'dragenter', { dataTransfer });
  await expect(page.locator('.drop-overlay')).toBeVisible();
  await page.dispatchEvent('body', 'dragover', { dataTransfer });
  await page.dispatchEvent('body', 'drop', { dataTransfer });
  await dataTransfer.dispose();
}

test('uploads the first and subsequent file drops independently', async ({ page }) => {
  let uploadRequests = 0;
  page.on('request', (request) => {
    if (request.method() === 'POST' && new URL(request.url()).pathname === '/api/files') uploadRequests++;
  });

  await page.goto('/');
  await page.getByRole('button', { name: '＋ 새 대화' }).click();
  await expect(page.locator('textarea')).toBeEnabled();

  for (const [index, name] of ['first.png', 'second.png', 'third.png'].entries()) {
    await dropPNG(page, name);
    await expect(page.locator('.pending-attachment')).toHaveCount(index + 1);
    await expect(page.locator('.pending-media-name', { hasText: name })).toBeVisible();
  }

  await expect.poll(() => uploadRequests).toBe(3);
  await expect(page.locator('.drop-overlay')).toHaveCount(0);
});
