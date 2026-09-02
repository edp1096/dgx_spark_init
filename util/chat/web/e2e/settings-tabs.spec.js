import { expect, test } from '@playwright/test';

test('keeps settings values while navigating categorized tabs', async ({ page }) => {
  await page.goto('/');
  await page.locator('.settings-button').click();

  const tabs = page.getByRole('tab');
  await expect(tabs).toHaveCount(6);
  await expect(page.getByRole('tab', { name: '대화' })).toHaveAttribute('aria-selected', 'true');

  const effort = page.getByLabel('기본 reasoning effort', { exact: true });
  await effort.selectOption('xhigh');
  await page.getByRole('tab', { name: '음성' }).click();
  await expect(page.getByText('음성 인식', { exact: true })).toBeVisible();
  const omitParentheticals = page.getByLabel('괄호 속 부연설명 읽지 않기');
  await expect(omitParentheticals).toBeChecked();
  await omitParentheticals.uncheck();
  await page.getByRole('tab', { name: '대화' }).click();
  await expect(effort).toHaveValue('xhigh');
  await page.getByRole('tab', { name: '음성' }).click();
  await expect(omitParentheticals).not.toBeChecked();
  await page.getByRole('tab', { name: '대화' }).click();

  const initialModal = await page.locator('.settings-modal').boundingBox();
  const initialActions = await page.locator('.modal-actions').boundingBox();
  for (const name of ['기억', '음성', '기능', '외형', '시스템']) {
    await page.getByRole('tab', { name }).click();
    const modal = await page.locator('.settings-modal').boundingBox();
    const actions = await page.locator('.modal-actions').boundingBox();
    expect(modal?.height).toBe(initialModal?.height);
    expect(actions?.y).toBe(initialActions?.y);
  }
});

test('changes Qwen reasoning with the compact header slider', async ({ page }) => {
  await page.goto('/');
  const effort = page.getByRole('slider', { name: 'Reasoning effort' });
  await expect(effort).toHaveValue('2');
  await effort.fill('3');
  await expect(page.locator('.model-controls .qwen-effort-control output')).toHaveText('XHigh');
  await effort.fill('0');
  await expect(page.locator('.model-controls .qwen-effort-control output')).toHaveText('꺼짐');
});

test('keeps tab navigation and actions reachable on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 600 });
  await page.goto('/');
  await expect(page.locator('.sidebar')).toHaveCount(0);
  await page.getByRole('button', { name: '사이드바 열기 또는 닫기' }).click();
  await expect(page.locator('.sidebar')).toBeVisible();
  await page.locator('.sidebar').evaluate((element) => Promise.all(element.getAnimations().map((animation) => animation.finished)));
  await page.locator('.settings-button').click();

  const chatTab = page.getByRole('tab', { name: '대화' });
  await chatTab.focus();
  await chatTab.press('End');
  await expect(page.getByRole('tab', { name: '시스템' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('button', { name: '저장' })).toBeVisible();

  const modal = await page.locator('.settings-modal').boundingBox();
  expect((modal?.y ?? -1)).toBeGreaterThanOrEqual(0);
  expect((modal?.y ?? 601) + (modal?.height ?? 0)).toBeLessThanOrEqual(600);
});

test('manages persistent memories from settings', async ({ page }) => {
  await page.goto('/');
  await page.locator('.settings-button').click();
  await page.getByRole('tab', { name: '기억' }).click();

  const create = page.locator('fieldset').filter({ has: page.getByText('새 기억', { exact: true }) });
  await create.getByLabel('제목').fill('응답 형식');
  await create.getByLabel('내용').fill('답변은 간결하게 작성');
  await create.getByRole('button', { name: '기억 추가' }).click();

  const item = page.locator('.memory-item').last();
  await expect(item).toBeVisible();
  await expect(item.getByLabel('기억 내용')).toHaveValue('답변은 간결하게 작성');
  await item.getByLabel('기억 내용').fill('답변은 아주 간결하게 작성');
  await item.getByRole('button', { name: '저장' }).click();
  await expect(item.getByLabel('기억 내용')).toHaveValue('답변은 아주 간결하게 작성');
  page.once('dialog', (dialog) => dialog.accept());
  await item.getByRole('button', { name: '삭제' }).click();
  await expect(item).toHaveCount(0);
});
