import { test, expect } from '@playwright/test';

test('completed translation exposes correction and reports narrow term checks', async ({ page, request }) => {
  const config = await (await request.get('/api/config')).json();
  const session = await (await request.post('/api/sessions', { data: { title: '표기 교정 시험' } })).json();
  const id = session.id, now = new Date().toISOString();
  const run = { id: 'terms-fixture', session_id: id, status: 'completed', current: 2, updated: now,
    document: { kind: 'translate', pipeline: 'terms_v1', terminal: true, phase: 'completed', source_titles: { source: '번역 문서' },
      completed: 3, total: 3, elapsed_seconds: 52, calls: 4, max_calls: 2048, max_output_tokens: 2000000, max_seconds: 43200,
      quality_status: '', publication: { files: [{ name: 'output.txt' }, { name: 'output.zip' }] },
      terms: { repair: true, status: 'complete', checked: 1, changed: 1, unresolved: 0, layout_blocks: 2, cases: [{ source: 'Perrin', status: 'complete', decisions: [{ canonical: '페린' }], references: [{ id: 'r1', ordinal: 0, status: 'repaired', forms: ['페린단'], evidence: 'Perrin was a cat.', patches: [{ before: '페린단이', after: '페린이' }] }] }] } } };
  run.document.terms.changed = 2;
  run.document.terms.language = { version: 'language_v1', checked: 1, records: [{ id: 'l1', ordinal: 1, status: 'repaired', reason: '원문의 대명사를 한국어로 번역했습니다.', mentions: [{ id: 'm1', text: 'elle는' }], checks: { m1: { status: 'translate', after: '그녀는' } } }] };
  let calls = 0;
  try {
    await page.route('**/api/health', route => route.fulfill({ json: { status: 'ok', model: 'test-model' } }));
    await page.route('**/api/models', route => route.fulfill({ json: ['test-model'] }));
    await page.route('**/api/sessions', route => route.fulfill({ json: [{ ...session, title: '표기 교정 시험' }] }));
    await page.route(`**/api/sessions/${id}/messages`, route => route.fulfill({ json: [] }));
    await page.route(`**/api/sessions/${id}/workflows`, route => route.fulfill({ json: [run] }));
    await page.route(`**/api/document-tasks/terms-fixture/repair_terms?*`, route => { calls++; return route.fulfill({ json: { run_id: 'corrected' } }); });
    const errors = []; page.on('pageerror', e => errors.push(e.message));
    await page.goto('/');
    const card = page.locator('.document-task');
    await expect(card.locator('.task-status')).toHaveText('완료');
    await expect(card.getByRole('button', { name: '재개', exact: true })).toHaveCount(0);
    await expect(card.getByRole('button', { name: '취소', exact: true })).toHaveCount(0);
    await card.getByRole('button', { name: '표기·어휘 교정', exact: true }).click();
    await expect.poll(() => calls).toBe(1);
    await card.locator('.term-results > summary').click();
    await card.locator('.term-results details > summary').click();
    await expect(card.locator('.term-results')).toContainText('페린단이 → 페린이');
    await expect(card.locator('.term-results')).toContainText('Perrin was a cat.');
    await expect(card).toContainText('번역 언어 확인 1/1문단');
    await card.locator('.quality-results > summary').click();
    await expect(card.locator('.quality-results')).toContainText('elle는 → 그녀는');
    for (const width of [390, 1280]) {
      await page.setViewportSize({ width, height: 900 });
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(width);
    }
    expect(errors).toEqual([]);
  } finally { await request.delete(`/api/sessions/${id}`); }
});
