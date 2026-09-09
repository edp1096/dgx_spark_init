import { test, expect } from '@playwright/test';
import { createServer } from 'node:http';

test('composes selections, persists a combination and sends the preview as one system prompt', async ({ page, request }) => {
  const calls = [];
  const backend = createServer(async (req, res) => {
    if (req.method !== 'POST') { res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify({ data: [{ id: 'test-model', context_length: 65536 }] })); return; }
    let raw = ''; for await (const part of req) raw += part;
    const body = JSON.parse(raw);
    if (!body.stream) { res.end(JSON.stringify({ choices: [{ message: { content: 'Prompt test' }, finish_reason: 'stop' }] })); return; }
    calls.push(body); res.setHeader('Content-Type', 'text/event-stream');
    res.end('data: ' + JSON.stringify({ choices: [{ delta: { content: '확인했습니다.' }, finish_reason: 'stop' }] }) + '\n\ndata: [DONE]\n\n');
  });
  await new Promise(resolve => backend.listen(0, '127.0.0.1', resolve));
  const original = await (await request.get('/api/config')).json(); let session;
  try {
    const cfg = structuredClone(original); cfg.model.endpoint = `http://127.0.0.1:${backend.address().port}`;
    cfg.context.window_tokens = 65536; cfg.model.system_prompt = ''; cfg.model.system_prompt_preset = '';
    cfg.model.prompt_composer.enabled = false;
    expect((await request.put('/api/config', { data: cfg })).ok()).toBeTruthy();
    await page.goto('/'); await page.locator('.settings-button').click();
    await page.getByRole('tab', { name: '프로필', exact: true }).click();
    await page.getByRole('combobox', { name: '프롬프트 작성 방식', exact: true }).selectOption('compose');
    await page.getByRole('tab', { name: '프로필', exact: true }).click();
    await page.getByRole('combobox', { name: '페르소나', exact: true }).selectOption('colleague');
    await page.locator('summary').filter({ hasText: /^답변 길이 / }).click();
    await page.getByLabel('간결하게', { exact: true }).check();
    await page.getByLabel('자세하게', { exact: true }).check();
    await expect(page.getByLabel('간결하게', { exact: true })).not.toBeChecked();
    await page.getByLabel('간결하게', { exact: true }).check();
    await page.locator('summary').filter({ hasText: /^근거·검증 / }).click();
    await page.getByLabel('사실·추론 구분', { exact: true }).check();
    const preview = await page.getByLabel('최종 프롬프트 미리보기').inputValue();
    expect(preview).toContain('[페르소나]'); expect(preview).toContain('[답변 길이]'); expect(preview).toContain('[근거·검증]');
    await page.getByText('조합 저장·관리', { exact: true }).click();
    page.once('dialog', dialog => dialog.accept('검증 조합'));
    await page.getByRole('button', { name: '새 조합 저장', exact: true }).click();
    await page.getByRole('button', { name: '저장', exact: true }).click();
    await expect.poll(async () => (await (await request.get('/api/config')).json()).model.system_prompt).toBe(preview);
    await page.reload(); await page.locator('.settings-button').click();
    await page.getByRole('tab', { name: '프로필', exact: true }).click();
    await expect(page.getByRole('combobox', { name: '프롬프트 작성 방식', exact: true })).toHaveValue('compose');
    await expect(page.getByLabel('저장한 조합').locator('option:checked')).toHaveText('검증 조합');
    await expect(page.getByLabel('최종 프롬프트 미리보기')).toHaveValue(preview);
    await page.setViewportSize({ width: 390, height: 700 });
    expect(await page.locator('.settings-modal').evaluate(el => el.scrollWidth <= el.clientWidth)).toBeTruthy();
    session = await (await request.post('/api/sessions', { data: { title: 'Prompt composition' } })).json();
    const response = await request.post('/api/chat', { data: { session_id: session.id, content: '확인', model: 'test-model', tools_enabled: false } });
    expect(response.ok()).toBeTruthy(); expect(calls.length).toBeGreaterThan(0);
    const systems = calls[0].messages.filter(m => m.role === 'system'); expect(systems).toHaveLength(1);
    expect(systems[0].content.startsWith(preview)).toBeTruthy();
    const invalid = await (await request.get('/api/config')).json();
    invalid.model.prompt_composer.condition_ids.push('detailed');
    expect((await request.put('/api/config', { data: invalid })).status()).toBe(400);
    expect((await (await request.get('/api/config')).json()).model.system_prompt).toBe(preview);
  } finally {
    if (session) await request.delete(`/api/sessions/${session.id}`);
    await request.put('/api/config', { data: original });
    backend.closeAllConnections(); await new Promise(resolve => backend.close(resolve));
  }
});

test('creates and deletes a custom condition without leaving stale saved references', async ({ page, request }) => {
  const original = await (await request.get('/api/config')).json();
  try {
    await page.goto('/'); await page.locator('.settings-button').click();
    await page.getByRole('tab', { name: '프로필', exact: true }).click();
    await page.getByRole('combobox', { name: '프롬프트 작성 방식', exact: true }).selectOption('compose');
    await page.getByText('페르소나·조건 편집', { exact: true }).click();
    await page.getByRole('button', { name: '조건 추가', exact: true }).click();
    await page.getByLabel('항목 이름').fill('테스트 표기'); await page.getByLabel('항목 내용').fill('테스트 조건을 적용한다.');
    await page.getByRole('button', { name: '항목 저장', exact: true }).click();
    await page.locator('summary').filter({ hasText: /^언어·표기 / }).click();
    await page.getByLabel('테스트 표기', { exact: true }).check();
    await page.getByText('조합 저장·관리', { exact: true }).click();
    page.once('dialog', dialog => dialog.accept('삭제 검사 조합'));
    await page.getByRole('button', { name: '새 조합 저장', exact: true }).click();
    page.once('dialog', dialog => dialog.accept());
    await page.getByRole('button', { name: '항목 삭제', exact: true }).click();
    await expect(page.getByLabel('테스트 표기', { exact: true })).toHaveCount(0);
    await page.getByRole('button', { name: '저장', exact: true }).click();
    await expect.poll(async () => (await (await request.get('/api/config')).json()).model.prompt_composer.combinations.some(s => s.name === '삭제 검사 조합')).toBeTruthy();
    await page.reload(); await page.locator('.settings-button').click();
    await page.getByRole('tab', { name: '프로필', exact: true }).click();
    await expect(page.getByLabel('최종 프롬프트 미리보기')).not.toHaveValue(/테스트 조건/);
    await page.getByRole('combobox', { name: '프롬프트 작성 방식', exact: true }).selectOption('direct');
    await page.getByRole('combobox', { name: '프롬프트 내용', exact: true }).selectOption('custom');
    await expect(page.locator('textarea.system-prompt')).toBeEditable();
  } finally { await request.put('/api/config', { data: original }); }
});


test('distinguishes clearing a preset, fresh text, and editing a copy across save and reload', async ({ page, request }) => {
  const original = await (await request.get('/api/config')).json();
  try {
    const cfg = structuredClone(original);
    cfg.model.prompt_composer.enabled = false;
    cfg.model.system_prompt_presets = [{ name: '테스트 프리셋', prompt: '직전 프리셋의 지침입니다.' }];
    cfg.model.system_prompt_preset = '테스트 프리셋';
    cfg.model.system_prompt = '직전 프리셋의 지침입니다.';
    expect((await request.put('/api/config', { data: cfg })).ok()).toBeTruthy();
    await page.goto('/'); await page.locator('.settings-button').click();
    await page.getByRole('tab', { name: '프로필', exact: true }).click();
    const source = page.getByRole('combobox', { name: '프롬프트 내용', exact: true });
    await expect(source).toHaveValue('preset:테스트 프리셋');
    await source.selectOption('custom');
    await expect(page.locator('textarea.system-prompt')).toHaveValue('');
    await page.locator('textarea.system-prompt').fill('직접 쓴 내용');
    await source.selectOption('preset:테스트 프리셋');
    await page.getByRole('button', { name: '현재 내용으로 직접 편집', exact: true }).click();
    await expect(source).toHaveValue('custom');
    await expect(page.locator('textarea.system-prompt')).toHaveValue('직전 프리셋의 지침입니다.');
    await page.locator('textarea.system-prompt').fill('복사해서 수정한 내용');
    await page.getByRole('button', { name: '저장', exact: true }).click();
    await expect.poll(async () => (await (await request.get('/api/config')).json()).model.system_prompt).toBe('복사해서 수정한 내용');
    let saved = await (await request.get('/api/config')).json();
    expect(saved.model.system_prompt_preset).toBe('');
    expect(saved.model.system_prompt_presets[0].prompt).toBe('직전 프리셋의 지침입니다.');
    await source.selectOption('preset:테스트 프리셋');
    await source.selectOption('none');
    await expect(page.locator('textarea.system-prompt')).toHaveCount(0);
    await page.getByRole('button', { name: '저장', exact: true }).click();
    await expect.poll(async () => (await (await request.get('/api/config')).json()).model.system_prompt).toBe('');
    await page.reload(); await page.locator('.settings-button').click();
    await page.getByRole('tab', { name: '프로필', exact: true }).click();
    await expect(page.getByRole('combobox', { name: '프롬프트 내용', exact: true })).toHaveValue('none');
    saved = await (await request.get('/api/config')).json();
    expect(saved.model.system_prompt_preset).toBe('');
    expect(saved.model.system_prompt_presets[0].prompt).toBe('직전 프리셋의 지침입니다.');
  } finally { await request.put('/api/config', { data: original }); }
});
