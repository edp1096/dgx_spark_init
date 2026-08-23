import { expect, test } from '@playwright/test';

test('keeps the composer inside a normal desktop browser viewport', async ({ page }) => {
  for (const viewport of [
    { width: 1366, height: 768 },
    { width: 1280, height: 640 },
  ]) {
    await page.setViewportSize(viewport);
    await page.goto('/');
    await expect(page.locator('.composer')).toBeVisible();

    const shell = await page.locator('.shell').boundingBox();
    const main = await page.locator('main').boundingBox();
    const footer = await page.locator('footer').boundingBox();
    expect(shell?.height).toBe(viewport.height);
    expect(main?.height).toBe(viewport.height);
    expect((footer?.y ?? viewport.height + 1) + (footer?.height ?? 0)).toBeLessThanOrEqual(viewport.height);
  }
});

test('keeps the app header fixed while only the message pane scrolls on mobile resize', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await expect(page.locator('.composer')).toBeVisible();
  await page.waitForTimeout(250);

  await page.locator('.messages').evaluate((messages) => {
    const filler = document.createElement('div');
    filler.style.height = '2400px';
    messages.append(filler);
    messages.scrollTop = messages.scrollHeight;
  });

  await page.evaluate(() => window.scrollTo(0, 1200));
  expect(await page.evaluate(() => window.scrollY)).toBe(0);

  await page.setViewportSize({ width: 390, height: 500 });

  const header = await page.locator('header').boundingBox();
  const footer = await page.locator('footer').boundingBox();
  expect(header?.y).toBe(0);
  expect((footer?.y ?? 501) + (footer?.height ?? 0)).toBeLessThanOrEqual(500);
  expect(await page.locator('.messages').evaluate((messages) => messages.scrollTop)).toBeGreaterThan(0);
  expect(await page.evaluate(() => document.documentElement.scrollHeight)).toBeLessThanOrEqual(500);
});
