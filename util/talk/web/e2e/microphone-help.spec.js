import { test, expect } from '@playwright/test';

for (const browser of [{ name: 'Chrome', protocol: 'chrome', marker: '' }, { name: 'Edge', protocol: 'edge', marker: ' Edg/150.0.0.0' }]) {
test(`${browser.name}: shows one setup address and copies the actual HTTP origin`, async ({ page }) => {
  let closing = false;
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  await page.route('http://sparktalk.test:8088/**', async route => {
    try {
      const url = new URL(route.request().url());
      const response = await route.fetch({ url: `http://127.0.0.1:18585${url.pathname}${url.search}` });
      await route.fulfill({ response });
    } catch (error) { if (!closing) throw error; }
  });
  await page.addInitScript(({ marker }) => {
    if (marker) { const ua = navigator.userAgent; Object.defineProperty(navigator, 'userAgent', { get: () => ua + marker }); }
    window.copiedValues = [];
    const original = document.execCommand.bind(document);
    document.execCommand = (command, ...args) => {
      if (command === 'copy') {
        window.copiedValues.push(document.activeElement.value);
        if (window.rejectCopy) return false;
      }
      return original(command, ...args);
    };
  }, { marker: browser.marker });
  try {
    await page.goto('http://sparktalk.test:8088/?view=chat#draft');
    expect(await page.evaluate(() => isSecureContext)).toBe(false);
    expect(await page.evaluate(() => Boolean(navigator.clipboard))).toBe(false);
    await page.getByRole('button', { name: '대화 도구', exact: true }).click();
    const tools = page.getByRole('dialog', { name: '대화 도구 설정', exact: true });
    await expect(tools.getByRole('button', { name: '마이크 사용 설정 도움말' })).toHaveText('HTTP 마이크 설정 방법');
    await tools.getByRole('button', { name: '마이크 사용 설정 도움말' }).click();
    const guide = page.getByRole('dialog', { name: '마이크 사용 설정', exact: true });
    await expect(guide.getByLabel('등록할 현재 화면 주소', { exact: true })).toHaveValue('http://sparktalk.test:8088');
    await expect(guide.locator('.copy-row')).toHaveCount(2);
    await guide.getByRole('button', { name: `${browser.name} 설정 주소 복사`, exact: true }).click();
    expect(await page.evaluate(() => window.copiedValues.at(-1))).toBe(`${browser.protocol}://flags/#unsafely-treat-insecure-origin-as-secure`);
    await expect(guide.getByRole('status')).toContainText('복사했습니다');
    await guide.getByRole('button', { name: '현재 화면 주소 복사', exact: true }).click();
    expect(await page.evaluate(() => window.copiedValues.at(-1))).toBe('http://sparktalk.test:8088');
    await expect(guide).toContainText('Enabled'); await expect(guide).toContainText('Relaunch');
    await page.evaluate(() => window.rejectCopy = true);
    await guide.getByRole('button', { name: '현재 화면 주소 복사', exact: true }).click();
    await expect(guide.getByRole('status')).toContainText('직접 복사');
    await expect(guide.getByLabel('등록할 현재 화면 주소', { exact: true })).toBeFocused();
    await page.setViewportSize({ width: 390, height: 720 });
    expect(await guide.evaluate(el => el.scrollWidth <= el.clientWidth)).toBeTruthy();
    await page.screenshot({ path: '/tmp/sparktalk-microphone-help-mobile.png' });
    await guide.getByRole('button', { name: '도움말 닫기' }).click();
    await expect(tools).toBeVisible();
    await page.keyboard.press('Escape');
    if (await page.locator('.sidebar').count()) await page.locator('.sidebar-close').click();
    await page.locator('.composer-tools').getByRole('button', { name: '마이크 사용 설정 도움말' }).click();
    await expect(guide).toBeVisible();
    expect(errors).toEqual([]);
  } finally { closing = true; await page.unrouteAll({ behavior: 'ignoreErrors' }); await page.close(); }
});

}

test('offers permission guidance instead of an unnecessary HTTP exception on localhost', async ({ page }) => {
  await page.goto('/'); await page.locator('.settings-button').click();
  await page.getByRole('tab', { name: '음성', exact: true }).click();
  await page.locator('#settings-panel-voice').getByRole('button', { name: '마이크 사용 설정 도움말' }).click();
  const guide = page.getByRole('dialog', { name: '마이크 사용 설정', exact: true });
  await expect(guide).toContainText('별도의 HTTP 예외 설정은 필요하지 않습니다');
  await expect(guide).toContainText('운영체제의 마이크 권한');
  await expect(guide.getByRole('button', { name: 'Chrome 설정 주소 복사' })).toHaveCount(0);
});
