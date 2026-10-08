import { test, expect } from '@playwright/test';

test('common attachment size is inherited until a type override is enabled', async ({ page, request }) => {
  const original = await (await request.get('/api/config')).json();
  try {
    expect(original.attachments.max_file_mb).toBe(256);
    expect(Object.keys(original.attachments.type_limits_mb || {})).toHaveLength(0);
    await page.goto('/'); await page.locator('.settings-button').click();
    await page.getByRole('tab', { name: '시스템', exact: true }).click();
    await page.getByRole('button', { name: '앱·저장소', exact: true }).click();
    const group=page.locator('fieldset').filter({has:page.locator('legend').filter({hasText:'첨부 파일 제한'})});
    await expect(group.getByLabel('공통 파일 크기 (MiB)')).toHaveValue('256');
    await group.getByText('형식별 크기 설정 (선택)',{exact:true}).click();
    await expect(group.getByLabel('이미지 개별 설정')).not.toBeChecked();
    await group.getByLabel('이미지 개별 설정').check();
    await group.getByLabel('이미지 크기 (MiB)',{exact:true}).fill('32');
    await group.getByLabel('공통 파일 크기 (MiB)').fill('128');
    await expect(group.getByLabel('이미지 크기 (MiB)',{exact:true})).toHaveValue('32');
    await page.getByRole('button',{name:'저장',exact:true}).click();
    await expect.poll(async()=>(await(await request.get('/api/config')).json()).attachments).toEqual({max_file_mb:128,max_files:6,type_limits_mb:{image:32}});
    await page.reload();await page.locator('.settings-button').click();
    await page.getByRole('tab',{name:'시스템',exact:true}).click();
    await page.getByRole('button',{name:'앱·저장소',exact:true}).click();
    await group.getByText('형식별 크기 설정 (선택)',{exact:true}).click();
    await group.getByLabel('이미지 개별 설정').uncheck();
    await page.getByRole('button',{name:'저장',exact:true}).click();
    await expect.poll(async()=>(await(await request.get('/api/config')).json()).attachments.type_limits_mb || {}).toEqual({});
  } finally {await request.put('/api/config',{data:original});}
});
