import { test, expect } from '@playwright/test';

test('a retained chat error can be dismissed without another generation', async ({ page, request }) => {
  let requests = 0;
  await page.route('**/api/chat', route => { requests++; return route.fulfill({status:502,body:'transcribe MiniMax-H3.mp4: ASR returned empty text'}); });
  const session = await (await request.post('/api/sessions', {data:{title:'오류 알림 검증'}})).json();
  try {
  await page.goto('/');
  await page.locator('.composer textarea').fill('오류 알림 확인');
  await page.locator('.composer textarea').press('Enter');
  await expect(page.locator('.chat-error')).toContainText('ASR returned empty text');
  await page.getByRole('button',{name:'오류 알림 닫기',exact:true}).click();
  await expect(page.locator('.chat-error')).toHaveCount(0);
  expect(requests).toBe(1);
  } finally { await request.delete(`/api/sessions/${session.id}`); }
});
