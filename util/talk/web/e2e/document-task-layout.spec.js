import { test, expect } from '@playwright/test';

for (const width of [1280, 390]) for (const theme of ['light', 'dark']) {
  test(`document card prioritizes results and preserves controls (${width}px ${theme})`, async ({ page, request }) => {
    await page.setViewportSize({ width, height: 900 });
    const config = await (await request.get('/api/config')).json();
    config.appearance = { ...config.appearance, theme };
    await page.route('**/api/config', route => route.fulfill({ json: config }));
    const session = await (await request.post('/api/sessions', { data: { title: '셜록홈즈의 모험', model: 'test-model' } })).json();
    const id = session.id, now = new Date().toISOString();
    try {
    await page.route('**/api/health', route => route.fulfill({ json: { status: 'ok', model: 'test-model', endpoint: config.model.endpoint } }));
    const run = { id: 'document-layout-run', session_id: id, status: 'needs_review', current: 2, updated: now, definition: { name: 'document-translate', steps: [] },
      document: { kind: 'translate', model: 'qwen38fn_exl3', effort: 'xhigh', phase: 'completed', instructions: '전문 번역', reasoning_budget_tokens: 8192,
        source_titles: { book: 'The Adventures of Sherlock Holmes | Project Gutenberg' }, completed: 252, total: 252, elapsed_seconds: 23468,
        image_total: 1, images_ready: 1, calls: 89, max_calls: 2048, output_tokens: 1012202, max_output_tokens: 2000000, max_seconds: 43200,
        warnings: ['glossary spelling requires review: Holmes → 홈즈'], glossary: { Holmes: '홈즈' },
        stage_metrics: { review: { calls: 3, seconds: 25, performance: { pp: 133.3333333, tg: 7.5, tg_estimated: true, calls: 2, prompt_tokens: 1300, cached_tokens: 900, output_tokens: 32 } } },
        last_performance: { pp: 450, tg: 50.8, ttft: 4.7, ttft_estimated: true, calls: 1 } } };
    await page.route('**/api/models', route => route.fulfill({ json: ['test-model'] }));
    await page.route('**/api/sessions', route => route.fulfill({ json: [{ id, title: '셜록홈즈의 모험', model: 'test-model', created_at: now, updated_at: now }] }));
    await page.route(`**/api/sessions/${id}/messages`, route => route.fulfill({ json: [{ id: 1, role: 'user', content: '전체 문서를 번역해라.', status: 'completed', created_at: now }, { id: 2, role: 'assistant', content: '번역 작업을 시작했습니다.', status: 'completed', created_at: now }] }));
    await page.route(`**/api/sessions/${id}/ssh-grants`, route => route.fulfill({ json: [] }));
    await page.route(`**/api/sessions/${id}/context?*`, route => route.fulfill({ json: { enabled: false, managed: false, segments: [], recalls: [] } }));
    let polls = 0;
    await page.route(`**/api/sessions/${id}/workflows`, route => { polls++; return route.fulfill({ json: [run] }); });
    const errors = []; page.on('pageerror', e => errors.push(e.message));
    await page.goto('/');
    const card = page.locator('.document-task'), panel = page.locator('.workflow-runs');
    await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
    await expect(card.locator('.task-status')).toHaveText('완료 · 확인 필요');
    await expect(card.locator('.source-title')).toContainText('Sherlock Holmes');
    await expect(card.getByRole('link', { name: /^결과 받기/ })).toBeVisible();
    await expect(card.locator('.document-timing')).toContainText('소요 6시간 31분 8초');
    await expect(card.locator('.document-performance-last')).toContainText('50.8 tok/s');
    await expect(card.getByRole('button', { name: '재개', exact: true })).toHaveCount(0);
    await expect(card.getByRole('button', { name: '취소', exact: true })).toHaveCount(0);
    await expect(card.getByRole('button', { name: '작업 삭제', exact: true })).toBeHidden();
    await expect(card.locator('.task-details dd').filter({ hasText: 'qwen38fn_exl3' })).toBeHidden();
    const primary = await card.getByRole('link', { name: /^결과 받기/ }).boundingBox();
    const panelBox = await panel.boundingBox();
    expect(primary.y + primary.height).toBeLessThanOrEqual(panelBox.y + panelBox.height + 1);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(width);
    expect(await card.evaluate(el => el.scrollWidth <= el.clientWidth)).toBeTruthy();
    await page.screenshot({ path: `/tmp/talk-document-completed-${width}-${theme}.png` });
    await card.getByRole('button', { name: '다른 형식' }).click();
    await expect(card.getByRole('link', { name: '단일 HTML', exact: true })).toBeVisible();
    await expect(card.getByRole('button', { name: 'PDF 만들기', exact: true })).toBeVisible();
    await card.getByRole('button', { name: '다른 형식' }).click();
    await card.locator('.task-warnings > summary').click();
    await expect(card.locator('.task-warnings')).toContainText('용어 표기 확인: Holmes → 홈즈');
    await card.locator('.task-warnings > summary').click();
    await card.locator('.task-details > summary').focus(); await page.keyboard.press('Enter');
    await expect(card.locator('.task-details dd').filter({ hasText: 'qwen38fn_exl3' })).toBeVisible();
    await expect(card.locator('.task-details')).toContainText('검수');
    await expect(card.locator('.task-details')).toContainText('133.3 tok/s');
    await expect(card.locator('.task-details')).toContainText('≈ 7.5 tok/s');
    expect(await card.evaluate(el => el.scrollWidth <= el.clientWidth)).toBeTruthy();
    await card.locator('.task-details > summary').click();
    await card.locator('.task-manage > summary').click();
    await card.getByRole('button', { name: '지침 변경', exact: true }).click();
    await expect(card.getByLabel('새 작업 지침', { exact: true })).toHaveValue('전문 번역');
    await card.getByRole('button', { name: '돌아가기', exact: true }).click();
    await card.getByRole('button', { name: '작업 삭제', exact: true }).click();
    await expect(card.locator('.confirmation')).toContainText('기록과 결과 파일을 삭제');
    await card.getByRole('button', { name: '돌아가기', exact: true }).click();
    await card.locator('.task-manage > summary').click();
    run.status = 'running'; run.current = 0;
    Object.assign(run.document, { phase: 'process', completed: 65, activity: 'reasoning', active_units: [66, 67, 68], current_chapter: 'IV. THE BOSCOMBE VALLEY MYSTERY', reasoning_chars: 12485,
      performance: { tg: 46.5, tg_estimated: true, ttft: 6.4, ttft_estimated: true, live: true, calls: 1 } });
    await page.reload();
    await expect(card.locator('.task-activity')).toContainText('모델 추론 중');
    await expect(card.getByRole('button', { name: '일시정지', exact: true })).toBeVisible();
    await expect(card.locator('.document-performance-current')).toContainText('≈ 46.5 tok/s');
    await expect(card.locator('progress')).toHaveAttribute('value', '65');
    const pause = await card.getByRole('button', { name: '일시정지', exact: true }).boundingBox();
    const activeBox = await panel.boundingBox();
    expect(pause.y + pause.height).toBeLessThanOrEqual(activeBox.y + activeBox.height + 1);
    const rail = await page.locator('.context-rail').boundingBox();
    expect(rail.y + rail.height).toBeLessThan(activeBox.y);
    await page.screenshot({ path: `/tmp/talk-document-running-${width}-${theme}.png` });
    await panel.locator(':scope > summary').click(); const before = polls;
    await expect.poll(() => polls, { timeout: 6000 }).toBeGreaterThan(before);
    await expect(panel).not.toHaveAttribute('open', '');
    await expect(panel.locator(':scope > summary')).toContainText('문서 번역 · 진행 중');
    await panel.locator(':scope > summary').click();
    await expect(card.getByRole('button', { name: '일시정지', exact: true })).toBeVisible();
    expect(errors).toEqual([]);
    await expect(page.locator('main > .error')).toHaveCount(0);
    } finally { await request.delete(`/api/sessions/${id}`); }
  });
}
