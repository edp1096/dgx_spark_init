import { test, expect } from '@playwright/test';

test('settings help supports keyboard, dismissal, focus return and mobile without losing drafts', async ({ page }) => {
  await page.goto('/'); await page.locator('.settings-button').click();
  const budget = page.getByLabel('최대 출력 토큰 (생각 과정 포함)', { exact: true });
  await budget.fill('12288');
  const info = page.getByRole('button', { name: '지능형 문맥 관리 도움말', exact: true });
  await info.focus(); await info.press('Enter');
  const help = page.getByRole('dialog', { name: '지능형 문맥 관리', exact: true });
  await expect(help).toBeVisible();
  await expect(help).toContainText('입력 예산 = 문맥 한도');
  await expect(help.getByRole('button', { name: '도움말 닫기' })).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(help).not.toBeVisible(); await expect(info).toBeFocused();
  await expect(budget).toHaveValue('12288');
  await info.click(); await page.mouse.click(2, 2);
  await expect(help).not.toBeVisible();
  await page.setViewportSize({ width: 390, height: 600 });
  await info.click();
  expect(await help.evaluate(el => el.scrollWidth <= el.clientWidth)).toBeTruthy();
  await page.screenshot({ path: '/tmp/sparktalk-settings-help-mobile.png' });
  await help.getByRole('button', { name: '도움말 닫기' }).click();
  for (const [tab, name] of [['기억', '기억 회수'], ['음성', '음성 인식'], ['음성', '답변 음성'], ['시스템', '테마'], ['시스템', '모델 연결']]) {
    await page.getByRole('tab', { name: tab, exact: true }).click();
    if (tab === '시스템') await page.getByRole('group', { name: '시스템 설정 분류' }).getByRole('button', { name: name === '테마' ? '외형' : '시작·연결', exact: true }).click();
    await page.getByRole('button', { name: `${name} 도움말`, exact: true }).click();
    const popup = page.getByRole('dialog', { name, exact: true });
    await expect(popup).toBeVisible();
    expect(await popup.evaluate(el => el.scrollWidth <= el.clientWidth)).toBeTruthy();
    await popup.getByRole('button', { name: '도움말 닫기' }).click();
  }
  await page.getByRole('tab', { name: '기억', exact: true }).click();
  await page.screenshot({ path: '/tmp/sparktalk-settings-memory-mobile.png' });
  await page.getByRole('tab', { name: '대화', exact: true }).click();
  await expect(budget).toHaveValue('12288');
  await page.locator('.modal-actions').getByRole('button', { name: '닫기', exact: true }).click();
  await expect(page.locator('dialog.settings-help-dialog')).toHaveCount(0);
});

test('opens settings and saves prompt items on an ordinary HTTP origin', async ({ page, request }) => {
  const original = await (await request.get('/api/config')).json();
  let closing = false;
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  // A non-loopback HTTP origin reproduces a LAN browser's security context.
  // Fulfill every request locally so the test never relies on DNS or a live app.
  await page.route('http://sparktalk.test/**', async route => {
    try {
      const url = new URL(route.request().url());
      const response = await route.fetch({ url: `http://127.0.0.1:18585${url.pathname}${url.search}` });
      await route.fulfill({ response });
    } catch (error) { if (!closing) throw error; }
  });
  try {
    await page.goto('http://sparktalk.test/');
    expect(await page.evaluate(() => isSecureContext)).toBe(false);
    expect(await page.evaluate(() => typeof crypto.randomUUID)).toBe('undefined');
    await page.locator('.settings-button').click();
    await expect(page.getByRole('dialog', { name: '설정', exact: true })).toBeVisible();
    await page.getByRole('button', { name: '지능형 문맥 관리 도움말', exact: true }).click();
    await page.getByRole('dialog', { name: '지능형 문맥 관리', exact: true }).getByRole('button', { name: '도움말 닫기' }).click();
    const ids = await page.locator('dialog.settings-help-dialog h3').evaluateAll(nodes => nodes.map(node => node.id));
    expect(new Set(ids).size).toBe(ids.length);
    await page.getByRole('tab', { name: '프로필', exact: true }).click();
    await page.getByRole('combobox', { name: '프롬프트 작성 방식', exact: true }).selectOption('compose');
    await page.getByText('페르소나·조건 편집', { exact: true }).click();
    await page.getByRole('button', { name: '페르소나 추가', exact: true }).click();
    await page.getByLabel('항목 이름').fill('HTTP 캐릭터');
    await page.getByLabel('항목 내용').fill('차분하게 답한다.');
    await page.getByRole('button', { name: '항목 저장', exact: true }).click();
    await page.getByText('조합 저장·관리', { exact: true }).click();
    page.once('dialog', dialog => dialog.accept('HTTP 조합'));
    await page.getByRole('button', { name: '새 조합 저장', exact: true }).click();
    await page.getByRole('button', { name: '저장', exact: true }).click();
    await expect.poll(async () => (await (await request.get('/api/config')).json()).model.prompt_composer.combinations.some(item => item.name === 'HTTP 조합')).toBeTruthy();
    await page.reload(); await page.locator('.settings-button').click();
    await expect(page.getByRole('dialog', { name: '설정', exact: true })).toBeVisible();
    expect(errors).toEqual([]);
  } finally {
    closing = true;
    await page.unrouteAll({ behavior: 'ignoreErrors' });
    await page.close();
    await request.put('/api/config', { data: original });
  }
});


test('places section help by the heading and field help immediately after its label', async ({ page }) => {
  await page.goto('/'); await page.locator('.settings-button').click();
  await expect(page.locator('legend').getByRole('button', { name: '추론 기본값 도움말', exact: true })).toBeVisible();
  await page.getByRole('tab', { name: '시스템', exact: true }).click();
  await page.getByRole('combobox', { name: '실행 방식', exact: true }).selectOption('managed');
  const field = page.locator('.settings-help-field').filter({ hasText: '기본 AI 세트' });
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 800 });
    const label = await field.locator('label').boundingBox();
    const icon = await field.getByRole('button', { name: '기본 AI 세트 도움말', exact: true }).boundingBox();
    const select = await field.getByRole('combobox').boundingBox();
    expect(icon.x).toBeGreaterThanOrEqual(label.x + label.width);
    expect(icon.x - label.x - label.width).toBeLessThan(12);
    expect(select.x).toBeGreaterThanOrEqual(icon.x + icon.width - 1);
    await page.screenshot({ path: `/tmp/sparktalk-help-placement-${width}.png` });
  }
  await field.getByRole('button', { name: '기본 AI 세트 도움말', exact: true }).click();
  await expect(page.getByRole('dialog', { name: '기본 AI 세트', exact: true })).toContainText('운영 패널');
});
