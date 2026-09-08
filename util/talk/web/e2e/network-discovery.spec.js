import { expect, test } from '@playwright/test';

test('discovers physical workers in the host selector without starting services', async ({ page, request }) => {
  const original = await (await request.get('/api/config')).json();
  let requests = 0;
  await page.route('**/api/runtime/network/discover', async route => {
    requests++;
    const { catalog } = route.request().postDataJSON();
    const node = { id:'worker-physical-id', hostname:'spark-worker', address:'192.168.100.60' };
    catalog.network = { enabled:true, head_id:'head-physical-id', worker_id:node.id, nodes:[node] };
    await route.fulfill({ json:{ local:{hostname:'spark-head',address:'192.168.100.61'}, candidates:[node], catalog } });
  });
  await page.goto('/');
  await page.locator('.settings-button').click();
  await page.getByRole('tab', {name:'시스템'}).click();
  await page.getByRole('button', {name:'AI 세트',exact:true}).click();
  const editor=page.locator('.set-editor');
  await editor.getByText('실행 호스트 편집',{exact:true}).click();
  await editor.getByLabel('이 컴퓨터를 헤드로 사용하고 워커 자동 탐색').check();
  await editor.getByRole('button',{name:'워커 탐색 · 주소 맞추기'}).click();
  await expect(editor.getByText('현재 헤드: spark-head',{exact:false})).toBeVisible();
  await expect(editor.getByLabel('워커 장비',{exact:true})).toHaveValue('worker-physical-id');
  expect(requests).toBe(1);
  // Discovery is a preview: unsaved edits must not change running configuration.
  const current=await (await request.get('/api/config')).json();
  expect(current.runtime.catalog).toEqual(original.runtime.catalog);
});
