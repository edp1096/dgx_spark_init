import { expect, test } from '@playwright/test';

test('keeps a bounded message DOM while scrolling in both directions', async ({ page }) => {
  const sessionId = 'long-session';
  const now = new Date().toISOString();
  const messages = Array.from({ length: 120 }, (_, index) => ({
    id: index + 1,
    session_id: sessionId,
    role: index % 2 ? 'assistant' : 'user',
    status: 'completed',
    content: `메시지 ${index + 1}\n\n가상 스크롤 검증을 위한 본문입니다.\n\n두 번째 문단입니다.`,
    reasoning_content: '',
    tool_trace: [],
    response_variants: [],
    created_at: now,
  }));

  await page.route('**/api/groups', (route) => route.fulfill({ json: [] }));
  await page.route('**/api/models', (route) => route.fulfill({ json: ['test-model'] }));
  await page.route('**/api/sessions', (route) => route.fulfill({ json: [{
    id: sessionId, title: '긴 대화', model: 'test-model', reasoning_effort: 'medium', group_id: '', created_at: now, updated_at: now,
  }] }));
  await page.route(`**/api/sessions/${sessionId}/messages`, (route) => route.fulfill({ json: messages }));
  await page.route(`**/api/sessions/${sessionId}/context`, (route) => route.fulfill({ json: { enabled: true, segments: [] } }));
  await page.route(`**/api/sessions/${sessionId}/ssh-grants`, (route) => route.fulfill({ json: [] }));
  await page.goto('/');

  const pane = page.locator('.messages');
  const rendered = pane.locator('article[data-message-index]');
  await expect(rendered).toHaveCount(48);
  const initialFirst = Number(await rendered.first().getAttribute('data-message-index'));
  expect(initialFirst).toBeGreaterThan(0);

  await pane.evaluate((node) => {
    node.style.scrollBehavior = 'auto';
    node.scrollTop = node.querySelector('.message-virtual-spacer')?.getBoundingClientRect().height || 0;
  });
  await expect(rendered).toHaveCount(48);
  await expect.poll(async () => Number(await rendered.first().getAttribute('data-message-index'))).toBeLessThan(initialFirst);
  const earlierFirst = Number(await rendered.first().getAttribute('data-message-index'));

  await pane.evaluate((node) => { node.scrollTop = node.scrollHeight; });
  await expect(rendered).toHaveCount(48);
  await expect.poll(async () => Number(await rendered.first().getAttribute('data-message-index'))).toBeGreaterThan(earlierFirst);
  const laterFirst = Number(await rendered.first().getAttribute('data-message-index'));

  await pane.evaluate((node) => {
    node.scrollTop = node.querySelector('.message-virtual-spacer')?.getBoundingClientRect().height || 0;
  });
  await expect.poll(async () => Number(await rendered.first().getAttribute('data-message-index'))).toBeLessThan(laterFirst);
  await expect(rendered).toHaveCount(48);
});
