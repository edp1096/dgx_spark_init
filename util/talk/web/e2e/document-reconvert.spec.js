import {test,expect} from '@playwright/test';

for (const width of [1280,390]) test(`file reconversion keeps downloads and translation timing (${width}px)`,async({page,request})=>{
  await page.setViewportSize({width,height:900});
  const session=await(await request.post('/api/sessions',{data:{title:'인간실격 번역',model:'test-model'}})).json();
  const id=session.id, now=new Date().toISOString();
  const run={id:'conversion-ui-run',session_id:id,status:'completed',current:2,updated:now,definition:{name:'document-translate',steps:[]},conversions:[],
    document:{kind:'translate',model:'test-model',phase:'completed',source_titles:{book:'인간실격'},elapsed_seconds:123,calls:11,completed:20,total:20,formats:['txt','md','html','zip','pdf']}};
  try{
    await page.route('**/api/models',route=>route.fulfill({json:['test-model']}));
    await page.route('**/api/sessions',route=>route.fulfill({json:[{...session,created_at:now,updated_at:now}]}));
    await page.route(`**/api/sessions/${id}/messages`,route=>route.fulfill({json:[]}));
    await page.route(`**/api/sessions/${id}/workflows`,route=>route.fulfill({json:[run]}));
    let calls=0;
    await page.route(`**/api/document-tasks/${run.id}/reexport?*`,route=>{
      expect(route.request().postDataJSON()).toEqual({format:'pdf'});calls++;
      run.conversions.unshift({id:`conversion-${calls}`,format:'pdf',status:'running',started_at:new Date(Date.now()-2000).toISOString()});
      return route.fulfill({status:202,json:{run_id:run.id,conversions:run.conversions}});
    });
    await page.goto('/');const card=page.locator('.document-task');
    await expect(card.locator('.task-status')).toHaveText('완료');
    await card.getByRole('button',{name:'다른 형식'}).click();
    const originalURL=`/api/document-tasks/${run.id}/output.pdf?session_id=${id}`;
    await expect(card.getByRole('link',{name:'PDF 받기',exact:true})).toHaveAttribute('href',originalURL);
    await card.getByRole('button',{name:'PDF 다시 만들기',exact:true}).click();
    await expect(card.locator('.conversion-statuses')).toContainText('PDF 변환 중');
    await expect(card.getByRole('button',{name:'PDF 다시 만들기',exact:true})).toBeDisabled();
    await expect(card.getByRole('link',{name:'PDF 받기',exact:true})).toHaveAttribute('href',originalURL);
    await expect(card.locator('.document-timing')).toContainText('소요 2분 3초');
    await expect(card.locator('.task-status')).toHaveText('완료');
    await page.reload();await expect(card.locator('.conversion-statuses')).toContainText('PDF 변환 중');
    Object.assign(run.conversions[0],{status:'completed',finished_at:now,seconds:4,current:true});
    await expect(card.locator('.conversion-statuses')).toContainText('PDF 변환 완료',{timeout:6000});
    await card.getByRole('button',{name:'다른 형식'}).click();
    const newURL=originalURL;
    await expect(card.getByRole('link',{name:'PDF 받기',exact:true})).toHaveAttribute('href',newURL);
    await expect(card.locator('.conversion-history')).toHaveCount(0);
    await expect(card.getByRole('link',{name:'최초 파일 받기',exact:true})).toHaveCount(0);
    await card.getByRole('button',{name:'PDF 다시 만들기',exact:true}).click();
    await expect(card.locator('.conversion-statuses')).toContainText('PDF 변환 중');
    Object.assign(run.conversions[0],{status:'failed',finished_at:now,seconds:5,error:'문서 변환 서비스에 연결할 수 없습니다.'});
    await expect(card.locator('.conversion-statuses')).toContainText('PDF 변환 실패',{timeout:6000});
    await expect(card.getByRole('link',{name:'PDF 받기',exact:true})).toHaveAttribute('href',newURL);
    await expect(card.getByRole('button',{name:'PDF 다시 만들기',exact:true})).toBeEnabled();
    await expect(card.locator('.document-timing')).toContainText('소요 2분 3초');
    expect(calls).toBe(2);expect(await card.evaluate(el=>el.scrollWidth<=el.clientWidth)).toBeTruthy();
    await page.screenshot({path:`/tmp/talk-document-reconvert-${width}.png`});
  }finally{await request.delete(`/api/sessions/${id}`);}
});
