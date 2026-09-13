import {expect,test} from '@playwright/test';

test('opens the matching profile section from message and brand avatars', async ({page}) => {
  const id='avatar-profile',now=new Date().toISOString();
  await page.route('**/api/groups',route=>route.fulfill({json:[]}));
  await page.route('**/api/models',route=>route.fulfill({json:['test-model']}));
  await page.route('**/api/sessions',route=>route.fulfill({json:[{id,title:'Avatar links',model:'test-model',created_at:now,updated_at:now}]}));
  await page.route(`**/api/sessions/${id}/messages`,route=>route.fulfill({json:[
    {id:1,session_id:id,role:'user',content:'질문',status:'completed',created_at:now},
    {id:2,session_id:id,role:'assistant',content:'답변',status:'completed',created_at:now},
  ]}));
  await page.route(`**/api/sessions/${id}/context`,route=>route.fulfill({json:{enabled:true,segments:[]}}));
  await page.route(`**/api/sessions/${id}/ssh-grants`,route=>route.fulfill({json:[]}));
  await page.goto('/');
  for(const width of [1280,390]) {
    await page.setViewportSize({width,height:850});
    for(const [selector,section] of [['article.mine .profile-avatar','내 프로필'],['article:not(.mine) .profile-avatar','AI 캐릭터']]) {
      const avatar=page.locator('.messages').locator(selector);
      await avatar.focus();await avatar.press('Enter');
      await expect(page.getByRole('tab',{name:'프로필',exact:true})).toHaveAttribute('aria-selected','true');
      await expect(page.locator('#settings-panel-profile').getByRole('button',{name:section,exact:true})).toHaveAttribute('aria-pressed','true');
      await page.getByRole('button',{name:'닫기',exact:true}).first().click();
    }
  }
  await page.setViewportSize({width:1280,height:850});
  if (!await page.locator('.brand .profile-avatar').count()) await page.getByRole('button',{name:'사이드바 열기 또는 닫기'}).click();
  await page.locator('.brand .profile-avatar').click();
  await expect(page.getByRole('tab',{name:'프로필',exact:true})).toHaveAttribute('aria-selected','true');
  await expect(page.locator('#settings-panel-profile').getByRole('button',{name:'AI 캐릭터',exact:true})).toHaveAttribute('aria-pressed','true');
  await page.getByRole('button',{name:'닫기',exact:true}).first().click();
  await page.locator('.settings-button').click();
  await expect(page.getByRole('tab',{name:'대화',exact:true})).toHaveAttribute('aria-selected','true');
});
