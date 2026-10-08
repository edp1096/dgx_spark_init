import {test,expect} from '@playwright/test';
import {selectOption,optionTexts,expectControlValue} from './select-helpers.js';
test('set order persists and reaches the runtime menu without changing selection',async({page,request})=>{
 const original=await(await request.get('/api/config')).json();
 try{
  await page.goto('/');await page.locator('.settings-button').click();
  await page.getByRole('tab',{name:'시스템',exact:true}).click();
  await page.getByRole('button',{name:'AI 세트',exact:true}).click();
  const bundles=original.runtime.catalog.bundles;
  const chosen=bundles[1];
  const select=page.getByRole('combobox',{name:'편집할 세트',exact:true});
  await selectOption(select,chosen.id);
  await page.getByRole('button',{name:'표시 순서',exact:true}).click();
  const list=page.getByRole('region',{name:'세트 표시 순서'});
  await list.getByRole('button',{name:chosen.name+' 위로 이동',exact:true}).click();
  await expect(list.getByRole('button',{name:chosen.name+' 위로 이동',exact:true})).toBeDisabled();
  // Drag the first row down, then back up, without changing the edited set.
  const handle=list.getByRole('button',{name:chosen.name+' 순서 이동',exact:true});
  const first=await handle.boundingBox();
  const target=await list.locator('li').nth(2).boundingBox();
  await page.mouse.move(first.x+18,first.y+18);await page.mouse.down();
  await page.mouse.move(target.x+target.width/2,target.y+target.height/2,{steps:5});await page.mouse.up();
  await expect(list.locator('li').nth(2)).toHaveAttribute('data-order-id',chosen.id);
  await list.getByRole('button',{name:chosen.name+' 위로 이동',exact:true}).click();
  await list.getByRole('button',{name:chosen.name+' 위로 이동',exact:true}).click();
  await page.setViewportSize({width:390,height:800});
  expect(await list.evaluate(el=>el.scrollWidth<=el.clientWidth)).toBeTruthy();
  await handle.scrollIntoViewIfNeeded();
  const touchFrom=await handle.boundingBox();
  const touchTo=await list.locator('li').nth(1).boundingBox();
  const cdp=await page.context().newCDPSession(page);
  await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:touchFrom.x+18,y:touchFrom.y+18}]});
  await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:touchTo.x+touchTo.width/2,y:touchTo.y+touchTo.height/2}]});
  await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
  await expect(list.locator('li').nth(1)).toHaveAttribute('data-order-id',chosen.id);
  await list.getByRole('button',{name:chosen.name+' 위로 이동',exact:true}).click();
  await page.getByRole('button',{name:'편집으로 돌아가기',exact:true}).click();
  await expectControlValue(select,chosen.id);
  expect((await optionTexts(select))[0]).toBe(chosen.name);
  await page.getByRole('button',{name:'저장',exact:true}).click();
  await expect.poll(async()=> (await(await request.get('/api/config')).json()).runtime.catalog.bundles[0].id).toBe(chosen.id);
  const saved=await(await request.get('/api/config')).json();expect(saved.runtime.bundle).toBe(original.runtime.bundle);
  await page.route('**/api/config',async route=>{
   const response=await route.fetch();const config=await response.json();
   config.runtime.mode='managed';await route.fulfill({response,json:config});
  });
  await page.route('**/api/runtime',route=>route.fulfill({json:{selected_bundle:saved.runtime.bundle,bundles:saved.runtime.catalog.bundles,components:[],hosts:{},memory:{},operation:{},docker:'online'}}));
  await page.reload();await page.locator('.status').click();
  const runtime=page.getByRole('combobox',{name:'전환할 AI 세트',exact:true});
  await expect(runtime).toBeVisible();
  expect((await optionTexts(runtime))[0].startsWith(chosen.name)).toBeTruthy();
 }finally{await request.put('/api/config',{data:original})}
});
