import { expect, test } from '@playwright/test';

for (const width of [1280, 390]) {
  test(`lowercase inference metrics follow response variants (${width}px)`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    const id = 'performance-test', now = new Date().toISOString();
    const measured = { pp: 1000, tg: 20.5, ttft: .42, calls: 1, cached_tokens: 512, elapsed_seconds: 83.2, finished_at: now, finish_status: 'completed' };
    await page.route('**/api/models', route => route.fulfill({ json: ['test-model'] }));
    await page.route('**/api/sessions', route => route.fulfill({ json: [{ id, title: '속도 검증', model: 'test-model', created_at: now, updated_at: now }] }));
    await page.route(`**/api/sessions/${id}/messages`, route => route.fulfill({ json: [
      { id: 1, session_id: id, role: 'user', status: 'completed', content: '질문', created_at: now },
      { id: 2, session_id: id, role: 'assistant', status: 'completed', content: '측정된 답변', performance: measured, created_at: now,
        variants: [{ content: '이전 답변', created_at: now }, { content: '측정된 답변', performance: measured, created_at: now }] },
    ] }));
    await page.route(`**/api/sessions/${id}/ssh-grants`, route => route.fulfill({ json: [] }));
    await page.route(`**/api/sessions/${id}/context?*`, route => route.fulfill({ json: { enabled: false, managed: false, segments: [], recalls: [] } }));
    await page.goto('/');
    const metrics = page.getByRole('group', { name: '모델 속도' });
    await expect(metrics.locator('.metric-label')).toHaveText(['pp', 'tg', 'ttft']);
    await expect(metrics).toContainText('1,000 tok/s');
    await expect(metrics).toContainText('20.5 tok/s');
    await expect(metrics).toContainText('0.42 s');
    await expect(metrics).toContainText('소요 1분 23초');
    await expect(metrics.locator('time')).toHaveAttribute('datetime', now);
    await expect(metrics).toContainText('완료');
    const box = await metrics.boundingBox();
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(width);
    await page.getByRole('button', { name: '이전 답변', exact: true }).click();
    await expect(metrics).toHaveCount(0);
    await page.getByRole('button', { name: '다음 답변', exact: true }).click();
    await expect(metrics).toBeVisible();
    await page.reload();
    await expect(metrics).toContainText('20.5 tok/s');
    await expect(metrics).toContainText('소요 1분 23초');
    await page.screenshot({ path: `/tmp/sparktalk-performance-${width}.png` });
  });
}

test('background document timing distinguishes active reasoning from a canceled task', async ({ page }) => {
  const id = 'document-clock-test', now = new Date().toISOString();
  const run = { id: 'document-run', session_id: id, status: 'running', current: 0, updated: now, definition: { name: 'document-translate', steps: [] },
    document: { kind: 'translate', model: 'test-model', effort: 'xhigh', phase: 'process', activity: 'reasoning', reasoning_chars: 46839, content_chars: 0,
      elapsed_seconds: 83, completed: 0, total: 252, calls: 0, max_calls: 2048, output_tokens: 0, max_output_tokens: 2000000, max_seconds: 43200 } };
  await page.route('**/api/models', route => route.fulfill({ json: ['test-model'] }));
  await page.route('**/api/sessions', route => route.fulfill({ json: [{ id, title: '문서 경과시간', model: 'test-model', created_at: now, updated_at: now }] }));
  await page.route(`**/api/sessions/${id}/messages`, route => route.fulfill({ json: [{ id: 1, session_id: id, role: 'user', status: 'completed', content: '번역', created_at: now }, { id: 2, session_id: id, role: 'assistant', status: 'completed', content: '작업을 시작했습니다.', created_at: now }] }));
  await page.route(`**/api/sessions/${id}/ssh-grants`, route => route.fulfill({ json: [] }));
  await page.route(`**/api/sessions/${id}/context?*`, route => route.fulfill({ json: { enabled: false, managed: false, segments: [], recalls: [] } }));
  await page.route(`**/api/sessions/${id}/workflows`, route => route.fulfill({ json: [run] }));
  await page.goto('/');
  const timing = page.locator('.document-timing');
  await expect(page.locator('.task-activity')).toContainText('모델 추론 중');
  await page.locator('.task-details > summary').click();await expect(page.locator('.task-details')).toContainText('46,839자');
  const before = await timing.textContent();
  await expect.poll(() => timing.textContent()).not.toBe(before);
  run.status = 'canceled'; run.document.elapsed_seconds = 317;
  await page.reload();
  await expect(timing).toContainText('소요 5분 17초');
  await expect(timing).toContainText('취소');
  await expect(timing.locator('time')).toHaveAttribute('datetime', now);
  await expect(page.locator('.task-activity')).toHaveCount(0);
});
