import { expect, test } from '@playwright/test';

test('long user inputs start collapsed, expand intact and collapse from the bottom', async ({ page }) => {
  const sessionId = 'input-collapse';
  const now = new Date().toISOString();
  const content = Array.from({length:80},(_,i)=>`문단 ${i+1}: 원본 입력 내용입니다.`).join('\n\n')+'\n\n마지막 원문 확인';
  const messages = [
    {id:1,role:'user',content:'짧은 입력입니다.'},
    {id:2,role:'assistant',content:'짧은 답변입니다.'},
    {id:3,role:'user',content,turn_inputs:[{id:1,content:'추가 지시\n'.repeat(30)}]},
    {id:4,role:'assistant',content:'답변은 그대로 표시합니다.'},
  ].map(m=>({...m,session_id:sessionId,status:'completed',tool_trace:[],response_variants:[],created_at:now}));
  await page.route('**/api/groups',r=>r.fulfill({json:[]}));
  await page.route('**/api/models',r=>r.fulfill({json:['test-model']}));
  await page.route('**/api/sessions',r=>r.fulfill({json:[{id:sessionId,title:'입력 접기 검증',model:'test-model',created_at:now,updated_at:now}]}));
  await page.route(`**/api/sessions/${sessionId}/messages`,r=>r.fulfill({json:messages}));
  await page.route(`**/api/sessions/${sessionId}/context`,r=>r.fulfill({json:{enabled:true,segments:[]}}));
  await page.route(`**/api/sessions/${sessionId}/ssh-grants`,r=>r.fulfill({json:[]}));
  await page.goto('/');
  const message=page.locator('article[data-message-id="3"]');
  const input=message.locator('.input-disclosure').first();
  await expect(input).not.toHaveAttribute('open','');
  expect((await input.boundingBox()).height).toBeLessThan(180);
  await expect(message.getByText('마지막 원문 확인',{exact:true})).toHaveCount(0);
  await expect(page.locator('article[data-message-id="1"] details')).toHaveCount(0);
  await expect(page.getByText('답변은 그대로 표시합니다.',{exact:true})).toBeVisible();
  await input.locator('summary').focus();await page.keyboard.press('Enter');
  await expect(input).toHaveAttribute('open','');
  await expect(input.locator('.prose')).toContainText('마지막 원문 확인');
  await input.getByRole('button',{name:'↑ 입력 접기',exact:true}).click();
  await expect(input).not.toHaveAttribute('open','');
  await expect(input.locator('summary')).toBeFocused();
  await expect(input.locator('summary')).toBeInViewport();
  await page.setViewportSize({width:390,height:700});
  const closeSidebar=page.getByRole('button',{name:'사이드바 닫기',exact:true}).first();
  if(await closeSidebar.isVisible()) await closeSidebar.click();
  await expect(input).not.toHaveAttribute('open','');
  expect((await input.boundingBox()).height).toBeLessThan(180);
  await input.locator('summary').click();
  await expect(input.locator('.prose')).toContainText('마지막 원문 확인');
  await input.locator('summary').click();
  await message.getByRole('button',{name:'✎ 수정',exact:true}).click();
  await expect(message.locator('textarea')).toHaveValue(content);
  await message.getByRole('button',{name:'취소',exact:true}).click();
  await page.reload();
  await expect(page.locator('article[data-message-id="3"] .input-disclosure').first()).not.toHaveAttribute('open','');
});
