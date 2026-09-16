import { test, expect } from '@playwright/test';
import { selectOption, controlValue } from './select-helpers.js';

test('settings selections require a separate option click, remain closed, and support keyboard', async ({ page }) => {
  await page.addInitScript(() => { HTMLSelectElement.prototype.showPicker = () => { throw new Error('native picker must not be called'); }; });
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto('/');await page.locator('.settings-button').click();
  async function check(select) {
    await select.scrollIntoViewIfNeeded();const box=await select.boundingBox();const before=await controlValue(select);
    await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.mouse.down();
    await expect(page.getByRole('listbox')).toHaveCount(0);
    await page.mouse.move(box.x+box.width/2,box.y+box.height+25);await page.mouse.up();
    expect(await controlValue(select)).toBe(before);
    for(let i=0;i<3;i++) {
      await select.click();await expect(select).toHaveAttribute('aria-expanded','true');
      await page.keyboard.press('Home');await page.keyboard.press('ArrowDown');await page.keyboard.press('Enter');
      await expect(select).toHaveAttribute('aria-expanded','false');await expect(page.getByRole('listbox')).toHaveCount(0);
    }
    await select.click();await select.click();await expect(select).toHaveAttribute('aria-expanded','false');
    await select.click();await page.keyboard.press('Escape');await expect(select).toBeFocused();
    expect(await select.boundingBox()).toEqual(box);
  }
  await check(page.getByRole('combobox',{name:'기본 reasoning effort',exact:true}));
  await check(page.getByRole('combobox',{name:'문맥 크기별 출력 프리셋',exact:true}));
  await page.getByRole('tab',{name:'시스템',exact:true}).click();
  await check(page.getByRole('combobox',{name:'실행 방식',exact:true}));
  await selectOption(page.getByRole('combobox',{name:'실행 방식',exact:true}),'managed');
  await check(page.getByRole('combobox',{name:'기본 AI 세트',exact:true}));
  expect(errors).toEqual([]);
});
