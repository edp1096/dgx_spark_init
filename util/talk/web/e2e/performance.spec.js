import { expect, test } from '@playwright/test';

for (const width of [1280, 390]) {
  test(`lowercase inference metrics follow response variants (${width}px)`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    const id = 'performance-test', now = new Date().toISOString();
    const measured = { pp: 1000, tg: 20.5, ttft: .42, calls: 1, cached_tokens: 512 };
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
    const box = await metrics.boundingBox();
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(width);
    await page.getByRole('button', { name: '이전 답변', exact: true }).click();
    await expect(metrics).toHaveCount(0);
    await page.getByRole('button', { name: '다음 답변', exact: true }).click();
    await expect(metrics).toBeVisible();
    await page.reload();
    await expect(metrics).toContainText('20.5 tok/s');
    await page.screenshot({ path: `/tmp/sparktalk-performance-${width}.png` });
  });
}
