import { expect } from '@playwright/test';

export async function controlValue(locator) {
  if (await locator.count() > 1) locator = locator.and(locator.page().locator('.select-trigger, input, textarea, select'));
  if (await locator.evaluate(el => el.classList.contains('select-trigger'))) return locator.getAttribute('data-value');
  return locator.inputValue();
}
export async function expectControlValue(locator, value, negate = false) {
  const check = expect.poll(() => controlValue(locator));
  const assertion = negate ? check.not : check;
  if (value instanceof RegExp) await assertion.toMatch(value); else await assertion.toBe(value);
}
export async function selectOption(locator, choice) {
  if (await locator.count() > 1) locator = locator.and(locator.page().locator('.select-trigger, select'));
  let trigger = locator;
  if (await locator.evaluate(el => el.tagName === 'SELECT' && el.nextElementSibling?.classList.contains('select-trigger'))) trigger = locator.locator('xpath=following-sibling::button[1]');
  if (!await trigger.evaluate(el => el.classList.contains('select-trigger'))) return locator.selectOption(choice);
  const option = await trigger.evaluate((el, choice) => {
    const options = Array.from(document.getElementById(el.dataset.selectSource).options);
    const item = typeof choice === 'string' ? options.find(o => o.value === choice) : 'label' in choice ? options.find(o => o.label === choice.label) : 'index' in choice ? options[choice.index] : options.find(o => o.value === choice.value);
    return item ? { index: item.index, value: item.value } : null;
  }, choice);
  if (!option) throw new Error(`Unknown option: ${JSON.stringify(choice)}`);
  await trigger.click();
  const id = await trigger.getAttribute('aria-controls');
  await trigger.page().locator(`#${id} [data-index="${option.index}"]`).click();
  await expectControlValue(trigger, option.value);
  return [option.value];
}

export async function optionTexts(locator) {
  const id = await locator.getAttribute('data-select-source');
  return locator.page().locator(`#${id} option`).allTextContents();
}
