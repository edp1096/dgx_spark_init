import { test, expect } from '@playwright/test';

test('common API address copies and follows the active model', async ({ page, context, baseURL }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  let model = 'first-model';
  await page.route('**/api/model-api', route => route.fulfill({json:{model:'sparktalk',active_model:model,base_path:'/v1'}}));
  await page.goto('/');
  await page.locator('.settings-button').click();
  await page.getByRole('tab',{name:'시스템',exact:true}).click();
  await expect(page.getByLabel('공통 API 주소',{exact:true})).toHaveValue(`${baseURL}/v1`);
  await expect(page.getByLabel('고정 모델 ID',{exact:true})).toHaveValue('sparktalk');
  await expect(page.getByText('현재 연결 대상:')).toContainText('first-model');
  await page.getByRole('button',{name:'API 주소 복사',exact:true}).click();
  await expect.poll(()=>page.evaluate(()=>navigator.clipboard.readText())).toBe(`${baseURL}/v1`);
  model='second-model';
  await expect(page.getByText('현재 연결 대상:')).toContainText('second-model',{timeout:10000});
  await page.setViewportSize({width:390,height:780});
  await expect(page.getByLabel('공통 API 주소',{exact:true})).toBeVisible();
  expect(await page.getByLabel('공통 API 주소',{exact:true}).evaluate(el=>el.closest('fieldset').scrollWidth<=el.closest('fieldset').clientWidth)).toBeTruthy();
});
