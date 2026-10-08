import { expect, test } from '@playwright/test';

const code = Array.from({ length: 20 }, (_, i) => `  const item${i} = ${i};`).join('\n') + '\n\n\n  // 마지막 줄';
const markdown = '# 제목\n\n**강조**와 [링크](https://example.com).\n\n- 첫 항목\n- 둘째 항목\n\n| 이름 | 값 |\n| --- | --- |\n| A | 42 |\n\n수식 $x^2$\n\n```js\n' + code + '\n```';

async function openReply(page) {
  const id = 'copy-reply-session';
  const now = new Date().toISOString();
  await page.route('**/api/groups', route => route.fulfill({ json: [] }));
  await page.route('**/api/models', route => route.fulfill({ json: ['test-model'] }));
  await page.route('**/api/sessions', route => route.fulfill({ json: [{ id, title: '응답 복사', model: 'test-model', created_at: now, updated_at: now }] }));
  await page.route(`**/api/sessions/${id}/messages`, route => route.fulfill({ json: [
    { id: 1, session_id: id, role: 'user', status: 'completed', content: '질문은 복사하지 않는다.', created_at: now },
    { id: 2, session_id: id, role: 'assistant', status: 'completed', content: markdown + '\n[Historical tool evidence: ssh_exec; archive_id=42; use context_read for original]', reasoning_content: '생각 과정은 복사하지 않는다.', tool_trace: [], created_at: now },
  ] }));
  await page.route(`**/api/sessions/${id}/context`, route => route.fulfill({ json: { enabled: true, segments: [] } }));
  await page.route(`**/api/sessions/${id}/ssh-grants`, route => route.fulfill({ json: [] }));
  await page.goto('/');
  return page.locator('article[data-message-id="2"]');
}

test('copies reply text or Markdown, including folded code without UI or reasoning', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  const reply = await openReply(page);
  await expect(reply.locator('[data-code-card]')).not.toHaveClass(/expanded/);
  await reply.getByRole('button', { name: '일반 복사', exact: true }).click();
  const text = await page.evaluate(() => navigator.clipboard.readText());
  expect(text).toContain('제목\n\n강조와 링크.');
  expect(text).toContain('• 첫 항목\n• 둘째 항목');
  expect(text).toContain('이름\t값\nA\t42');
  expect(text).toContain('수식 x^2');
  expect(text).toContain(code);
  expect(text).not.toMatch(/\*\*|```|Historical tool evidence|생각 과정|질문은|전체 보기|복사|접기/);
  await expect(reply.getByRole('status')).toHaveText('일반 텍스트 복사됨');
  await reply.getByRole('button', { name: 'Markdown 복사', exact: true }).click();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(markdown);
  await expect(reply.getByRole('status')).toHaveText('Markdown 복사됨');
  await reply.locator('[data-code-copy]').click();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(code);
});

test('falls back on HTTP/denied clipboard and restores focus on a narrow screen', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: async () => { throw new Error('Denied'); } } });
    document.execCommand = command => {
      if (command !== 'copy') return false;
      window.copiedText = document.activeElement.value;
      return true;
    };
  });
  const reply = await openReply(page);
  const button = reply.getByRole('button', { name: 'Markdown 복사', exact: true });
  await button.click();
  expect(await page.evaluate(() => window.copiedText)).toBe(markdown);
  await expect(button).toBeFocused();
  await expect(reply.getByRole('status')).toHaveText('Markdown 복사됨');
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
});

test('does not report success when both clipboard methods fail', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined });
    document.execCommand = () => false;
  });
  const reply = await openReply(page);
  await reply.getByRole('button', { name: '일반 복사', exact: true }).click();
  await expect(reply.getByRole('status')).toContainText('복사하지 못했습니다');
  await expect(reply.getByRole('status')).not.toContainText('복사됨');
});
