import { test, expect } from '@playwright/test';

test('keeps scalar labels beside their controls throughout settings on desktop and mobile', async ({ page }) => {
  test.setTimeout(60_000);
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto('/'); await page.locator('.settings-button').click();
  const stages = [
    ['대화'], ['프로필', 'AI 캐릭터'], ['프로필', '내 프로필'], ['기억'], ['음성'], ['시스템', '외형'],
    ['기능', '웹·미디어'], ['기능', '이미지'], ['기능', '문서'], ['기능', 'SSH·키'], ['기능', '스킬·기록'],
    ['시스템', '시작·연결'], ['시스템', 'AI 세트'], ['시스템', '지원 서비스'], ['시스템', '모델 준비'], ['시스템', '앱·저장소'],
  ];
  let checked = 0;
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 800 });
    for (const [tab, section] of stages) {
      await page.getByRole('tab', { name: tab, exact: true }).click();
      const panel = page.locator('.settings-tab-panel.active');
      if (section) await panel.locator('.settings-section-navigation').getByRole('button', { name: section, exact: true }).click();
      const result = await panel.evaluate(el => {
        for (const detail of el.querySelectorAll('details')) detail.open = true;
        const issues = []; let count = 0;
        for (const label of el.querySelectorAll('label')) {
          const input = label.querySelector(':scope > input:not([type="checkbox"]):not([type="radio"]):not([type="hidden"]), :scope > select') || (label.htmlFor ? label.control : null);
          if (!input || !input.checkVisibility()) continue;
          const title = label.htmlFor ? label : label.querySelector(':scope > span');
          if (!title) { issues.push(`Missing inline title: ${label.textContent.slice(0, 60)}`); continue; }
          const a = title.getBoundingClientRect(), b = input.getBoundingClientRect();
          if (b.left < a.right - 1 || b.bottom <= a.top || a.bottom <= b.top) issues.push(`Stacked control: ${title.textContent}`);
          count++;
        }
        for (const button of el.querySelectorAll('button')) {
          if (button.checkVisibility() && getComputedStyle(button).borderTopStyle === 'outset') issues.push(`Unstyled button: ${button.textContent}`);
        }
        return { count, issues };
      });
      expect(result.issues, `${width}px ${tab}/${section || ''}`).toEqual([]);
      checked += result.count;
      expect(await page.locator('.settings-modal').evaluate(el => el.scrollWidth <= el.clientWidth)).toBeTruthy();
      if (['대화', '음성', '프로필'].includes(tab) && section !== '내 프로필') {
        await panel.evaluate(el => el.parentElement.scrollTop = 0);
        await page.screenshot({ path: `/tmp/sparktalk-settings-${width}-${tab}.png` });
      }
    }
  }
  expect(checked).toBeGreaterThan(100);
  expect(errors).toEqual([]);
});
