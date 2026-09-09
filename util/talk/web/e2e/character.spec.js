import { test, expect } from '@playwright/test';

test('keeps character and user profiles separate and preserves shared rules across persona and model changes', async ({ page, request }) => {
  const original = await (await request.get('/api/config')).json();
  try {
    await page.goto('/'); await page.locator('.settings-button').click();
    await page.getByRole('tab', { name: '프로필', exact: true }).click();
    await expect(page.locator('#settings-panel-chat').getByRole('combobox', { name: '프롬프트 작성 방식', exact: true })).toHaveCount(0);
    await page.getByRole('combobox', { name: '프롬프트 작성 방식', exact: true }).selectOption('compose');
    await page.locator('summary').filter({ hasText: /^답변 길이 / }).click();
    await page.getByLabel('간결하게', { exact: true }).check();
    await page.getByRole('tab', { name: '프로필', exact: true }).click();
    await page.getByLabel('AI 이름', { exact: true }).fill('아리아');
    await page.getByLabel('짧은 소개').fill('차분하고 호기심 많은 대화 상대');
    await page.getByRole('combobox', { name: '페르소나', exact: true }).selectOption('teacher');
    await page.getByRole('button', { name: '변경', exact: true }).filter({ visible: true }).click();
    await page.getByRole('button', { name: 'AI 아바타: 고양이', exact: true }).click();
    await page.getByRole('combobox', { name: '페르소나', exact: true }).selectOption('reviewer');
    await page.getByRole('combobox', { name: '프롬프트 작성 방식', exact: true }).scrollIntoViewIfNeeded();
    await page.screenshot({ path: '/tmp/sparktalk-profile-desktop.png' });
    const preview = await page.getByLabel('최종 프롬프트 미리보기').inputValue();
    expect(preview).toContain('아리아'); expect(preview).toContain('[답변 길이]');
    await page.getByRole('button', { name: '내 프로필', exact: true }).click();
    await page.getByLabel('내 이름', { exact: true }).fill('사용자');
    await page.getByRole('button', { name: '변경', exact: true }).filter({ visible: true }).click();
    await page.getByRole('button', { name: '내 아바타: 강아지', exact: true }).click();
    await page.getByRole('button', { name: 'AI 캐릭터', exact: true }).click();
    await expect(page.getByLabel('간결하게', { exact: true })).toBeChecked();
    await expect(page.getByLabel('최종 프롬프트 미리보기')).toHaveValue(preview);
    await page.getByRole('button', { name: '저장', exact: true }).click();
    await expect.poll(async () => (await (await request.get('/api/config')).json()).model.system_prompt).toBe(preview);
    const cfg = await (await request.get('/api/config')).json();
    expect(cfg.appearance.assistant_avatar).toBe('preset:cat'); expect(cfg.appearance.user_avatar).toBe('preset:dog');
    cfg.model.default_model = 'another-model';
    expect((await request.put('/api/config', { data: cfg })).ok()).toBeTruthy();
    await page.reload();
    await expect(page.locator('.brand strong')).toHaveText('아리아');
    await page.locator('.settings-button').click();
    await page.getByRole('tab', { name: '프로필', exact: true }).click();
    await expect(page.getByLabel('AI 이름', { exact: true })).toHaveValue('아리아');
    await expect(page.getByLabel('최종 프롬프트 미리보기')).toHaveValue(preview);
    await expect(page.getByAltText('AI 아바타 미리보기')).toHaveAttribute('src', '/avatars/cat.png');
    await page.getByRole('button', { name: '변경', exact: true }).filter({ visible: true }).click();
    await page.locator('#settings-profile-character input[type=file]').setInputFiles({
      name: 'avatar.png', mimeType: 'image/png',
      buffer: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aS1cAAAAASUVORK5CYII=', 'base64'),
    });
    await expect(page.getByAltText('AI 아바타 미리보기')).toHaveAttribute('src', /\/api\/images\//);
    await page.getByRole('button', { name: '저장', exact: true }).click();
    await expect.poll(async () => (await (await request.get('/api/config')).json()).appearance.assistant_avatar).toMatch(/^\/api\/images\//);
    expect((await (await request.get('/api/config')).json()).appearance.user_avatar).toBe('preset:dog');
    await page.setViewportSize({ width: 390, height: 700 });
    await page.getByRole('combobox', { name: '프롬프트 작성 방식', exact: true }).scrollIntoViewIfNeeded();
    await page.screenshot({ path: '/tmp/sparktalk-profile-mobile.png' });
    expect(await page.locator('.settings-modal').evaluate(el => el.scrollWidth <= el.clientWidth)).toBeTruthy();
  } finally { await request.put('/api/config', { data: original }); }
});
