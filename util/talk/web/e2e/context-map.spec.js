import { expect, test } from '@playwright/test';

for (const managed of [true, false]) {
  test(`context map separates checkpoint history and estimates (managed=${managed})`, async ({ page }) => {
    const id = 'context-map-test';
    const now = new Date().toISOString();
    await page.route('**/api/models', route => route.fulfill({ json: ['test-model'] }));
    await page.route('**/api/sessions', route => route.fulfill({ json: [{ id, title: '문맥 검증', model: 'test-model', created_at: now, updated_at: now }] }));
    await page.route(`**/api/sessions/${id}/messages`, route => route.fulfill({ json: [{ id: 5, session_id: id, role: 'user', status: 'completed', content: '현재 질문', created_at: now }] }));
    await page.route(`**/api/sessions/${id}/ssh-grants`, route => route.fulfill({ json: [] }));
    await page.route(`**/api/sessions/${id}/context?*`, route => route.fulfill({ json: {
      enabled: managed, managed, preview: true, incomplete: true,
      estimated_tokens: 12000, input_budget: 10000, active_tokens: 7000,
      summary_tokens: managed ? 1000 : 0, recall_tokens: 500, system_tool_tokens: 3500,
      applied_segment_id: managed ? 2 : 0, active_start_message_id: 5, active_end_message_id: 5,
      segments: [{ id: 1, start_message_id: 1, end_message_id: 2, summary: '이전 요약 내용', estimated_tokens: 2000 }, { id: 2, start_message_id: 3, end_message_id: 4, summary: '누적 요약 내용', estimated_tokens: 3000 }],
    } }));
    await page.goto('/');
    await page.locator('.context-rail-toggle').click();
    await expect(page.locator('.context-rail-toggle')).toHaveText('120%');
    await expect(page.getByText('다음 요청 예상 · 저장된 첨부 정보 기준')).toBeVisible();
    await expect(page.getByText(/추정치가 미완료/)).toBeVisible();
    await expect(page.getByText('이전 요약 1 · 미적용')).toBeVisible();
    if (managed) {
      await expect(page.getByText('현재 적용 요약', { exact: true })).toBeVisible();
      await expect(page.getByRole('button', { name: '지금 구간 정리' })).toBeEnabled();
    } else {
      await expect(page.getByText(/자동 관리 꺼짐/)).toBeVisible();
      await expect(page.getByText('현재 적용 요약', { exact: true })).toHaveCount(0);
      await expect(page.getByRole('button', { name: '지금 구간 정리' })).toBeDisabled();
    }
    await page.screenshot({ path: `/tmp/sparktalk-context-map-${managed ? 'on' : 'off'}.png` });
  });
}
