import { test, expect } from '@playwright/test';

test('groups model controls, conversation tools, and connection status without overflowing', async ({ page }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  await page.route('**/api/models', route => route.fulfill({ json: ['first-model', 'second-model-with-a-very-long-name'] }));
  await page.goto('/');
  const modelButton = page.getByRole('button', { name: '모델 및 대화 설정', exact: true });
  const toolsButton = page.getByRole('button', { name: '대화 도구', exact: true });
  for (const width of [1280, 390, 320]) {
    await page.setViewportSize({ width, height: 720 });
    if (width < 600 && await page.locator('.sidebar').count()) await page.locator('.sidebar-close').click();
    await expect(page.locator('.chat-header .status')).toBeVisible();
    expect(await page.locator('.chat-header').evaluate(el => el.scrollWidth <= el.clientWidth)).toBeTruthy();
    await modelButton.click();
    const modelPanel = page.getByRole('dialog', { name: '모델 및 대화 설정', exact: true });
    await expect(modelPanel).toBeVisible();
    if (width === 1280) await expect(page.locator('.sidebar')).toBeVisible();
    await modelPanel.getByRole('combobox', { name: '모델 선택' }).selectOption('second-model-with-a-very-long-name');
    await modelPanel.getByRole('slider', { name: 'Reasoning effort' }).fill('3');
    await expect(modelPanel.locator('output')).toHaveText('XHigh');
    expect(await modelPanel.evaluate(el => el.scrollWidth <= el.clientWidth)).toBeTruthy();
    await page.screenshot({ path: `/tmp/sparktalk-header-model-${width}.png` });
    await toolsButton.click();
    const toolsPanel = page.getByRole('dialog', { name: '대화 도구 설정', exact: true });
    await expect(modelPanel).toHaveCount(0);
    await expect(toolsPanel).toBeVisible();
    const web = toolsPanel.getByRole('button', { name: '웹검색 자동 사용', exact: true });
    await expect(web).toHaveAttribute('aria-pressed', 'false'); await web.click();
    await expect(web).toHaveAttribute('aria-pressed', 'true');
    await expect(toolsButton).toHaveAttribute('title', /웹검색 자동/);
    await page.screenshot({ path: `/tmp/sparktalk-header-tools-${width}.png` });
    await toolsPanel.getByRole('button', { name: 'Extra·모델 서비스 상태 보기 →' }).click();
    await expect(toolsPanel).toHaveCount(0);
    await expect(page.getByRole('dialog', { name: 'DGX Spark 운영 상태' })).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.getByRole('dialog', { name: 'DGX Spark 운영 상태' })).toHaveCount(0);
    await modelButton.click();
    await expect(page.getByRole('combobox', { name: '모델 선택' })).toHaveValue('second-model-with-a-very-long-name');
    await expect(page.getByRole('slider', { name: 'Reasoning effort' })).toHaveValue('3');
    await page.keyboard.press('Escape'); await expect(modelButton).toBeFocused();
    await toolsButton.click(); await page.getByRole('button', { name: '웹검색 자동 사용' }).click();
    await page.locator('.chat-header').click({ position: { x: 10, y: 1 } });
    await expect(toolsPanel).toHaveCount(0);
  }
  expect(errors).toEqual([]);
});

test('does not reset a chosen model when the initial health check finishes late', async ({ page }) => {
  let releaseHealth;
  const ready = new Promise(resolve => releaseHealth = resolve);
  await page.route('**/api/models', route => route.fulfill({ json: ['first-model', 'second-model'] }));
  await page.route('**/api/health', async route => {
    await ready;
    await route.fulfill({ json: { status: 'ok', model: 'first-model' } });
  });
  try {
    await page.goto('/');
    await page.getByRole('button', { name: '모델 및 대화 설정', exact: true }).click();
    const selector = page.getByRole('combobox', { name: '모델 선택', exact: true });
    await selector.selectOption('second-model');
    const initialized = page.waitForResponse(response => response.url().endsWith('/api/sessions'));
    releaseHealth(); await initialized;
    await expect(selector).toHaveValue('second-model');
    await expect(page.getByRole('button', { name: '모델 및 대화 설정', exact: true })).toHaveAttribute('title', /second-model/);
  } finally { releaseHealth(); }
});
