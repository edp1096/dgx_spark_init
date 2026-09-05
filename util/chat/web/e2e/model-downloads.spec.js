import { expect, test } from '@playwright/test';

test('stores download credentials separately and exposes embedded model preparation', async ({ page, request }) => {
  await page.goto('/');
  await page.locator('.settings-button').click();
  await page.getByRole('tab', { name: '시스템' }).click();
  await page.getByRole('button', { name: '모델 준비', exact: true }).click();
  await page.getByRole('combobox', { name: '모델', exact: true }).selectOption('ds4fve');
  await page.getByRole('combobox', { name: '가중치', exact: true }).selectOption('abliterated');
  await expect(page.getByRole('link', { name: /drowzeys\/keys-DeepSeek/ })).toHaveAttribute('href', 'https://huggingface.co/drowzeys/keys-DeepSeekV4Flash-Vision-EXP-ablit');
  await page.getByRole('combobox', { name: '모델', exact: true }).selectOption('glm53');
  await expect(page.getByRole('link', { name: /lovesenko\/GLM/ })).toBeVisible();
  await page.getByRole('combobox', { name: '가중치', exact: true }).selectOption('official');
  await expect(page.getByRole('link', { name: /lovesenko\/GLM/ })).toHaveCount(0);
  const token = 'hf_ui_test_download_credential';
  try {
    await page.getByLabel('다운로드 토큰', { exact: true }).fill(token);
    await page.getByRole('button', { name: '토큰 저장', exact: true }).click();
    await expect(page.getByText('토큰 등록됨', { exact: true })).toBeVisible();
    await expect(page.getByLabel('새 토큰으로 교체')).toHaveValue('');
    expect(await (await request.get('/api/credentials/huggingface')).json()).toEqual({ configured: true });
    expect(await (await request.get('/api/config')).text()).not.toContain(token);
    await expect(page.getByRole('button', { name: '모델만 준비', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: '전체 준비', exact: true })).toBeVisible();
    await page.getByRole('button', { name: '토큰 삭제', exact: true }).click();
    await expect(page.getByText('등록된 토큰 없음', { exact: true })).toBeVisible();
  } finally { await request.delete('/api/credentials/huggingface'); }
});

test('shows live preparation output and elapsed time without inventing a percentage', async ({ page }) => {
  await page.route('**/api/models/prepare', route => route.fulfill({json: {state:'running',component:'ds4fve',detail:'워커 모델 동기화 시작',logs:['[파일 완료 2/4] 50% (파일 개수 기준)','워커 모델 동기화 시작'],started_at:new Date(Date.now()-65000).toISOString()}}));
  await page.goto('/');
  await page.locator('.settings-button').click();
  await page.getByRole('tab', { name: '시스템' }).click();
  await page.getByRole('button', { name: '모델 준비', exact: true }).click();
  await expect(page.getByRole('progressbar', { name: '모델 준비 진행 중' })).toBeVisible();
  await expect(page.locator('.preparation-progress')).toContainText('경과 1분');
  await expect(page.locator('.preparation-log')).toContainText('[파일 완료 2/4]');
  await expect(page.getByRole('button', {name:'모델만 준비',exact:true})).toBeDisabled();
});
