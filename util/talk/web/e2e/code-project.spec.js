import { expect, test } from '@playwright/test';
import { createServer } from 'node:http';

test('one code pool survives partial edits, version restore, reload and session switches', async ({ page, request }) => {
  test.setTimeout(90_000);
  let round = 0, projectId;
  const model = createServer(async (req, res) => {
    if (req.method !== 'POST') { res.end(JSON.stringify({ data: [{ id: 'test-model' }] })); return; }
    let raw = ''; for await (const chunk of req) raw += chunk;
    const body = JSON.parse(raw);
    if (!body.stream) { res.end(JSON.stringify({ choices: [{ message: { content: '코드 프로젝트 검증' } }] })); return; }
    round++;
    res.setHeader('Content-Type', 'text/event-stream');
    let args;
    if (round === 1) args = { action: 'create', title: 'ONE POOL', files: [{ name: 'index.html', source: '<html><head><title>ONE POOL</title><link rel="stylesheet" href="./style.css"><script src="./script.js" defer></script></head><body><h1>Version One</h1></body></html>' }, { name: 'style.css', source: 'h1 { color: red; }' }, { name: 'script.js', source: 'document.body.dataset.executions = String(Number(document.body.dataset.executions || 0) + 1);' }] };
    if (round === 2) {
      const tool = [...body.messages].reverse().find(m => m.role === 'tool');
      projectId = JSON.parse(tool.content).id;
    }
    if (round === 3) args = { action: 'read', project_id: projectId };
    if (round === 4) {
      const current = JSON.parse([...body.messages].reverse().find(m => m.role === 'tool').content);
      args = { action: 'edit', project_id: projectId, base_version: current.version, summary: '제목과 색상 개선', edits: [{ name: 'index.html', operation: 'delete' }, { name: 'index.html', operation: 'create', source: current.files.find(f => f.name === 'index.html').source.replace('Version One', 'Version Two') }, { name: 'style.css', operation: 'replace', old: 'red', new: 'blue' }] };
    }
    const delta = args ? { tool_calls: [{ index: 0, id: `code-${round}`, type: 'function', function: { name: 'code_project', arguments: JSON.stringify(args) } }] } : { content: round === 2 ? '프로젝트 생성 완료' : '부분 수정 완료' };
    res.end('data: ' + JSON.stringify({ choices: [{ delta, finish_reason: args ? 'tool_calls' : 'stop' }] }) + '\n\ndata: [DONE]\n\n');
  });
  await new Promise(resolve => model.listen(0, '127.0.0.1', resolve));
  const original = await (await request.get('/api/config')).json();
  const sessions = [];
  try {
    const config = structuredClone(original);
    config.model.endpoint = `http://127.0.0.1:${model.address().port}`;
    config.tools.max_rounds = 6;
    expect((await request.put('/api/config', { data: config })).ok()).toBeTruthy();
    const session = await (await request.post('/api/sessions', { data: { title: '코드 프로젝트 검증' } })).json(); sessions.push(session.id);
    await page.goto('/');
    const composer = page.locator('.composer textarea');
    await composer.fill('웹 페이지 만들어'); await composer.press('Enter');
    await expect(page.locator('.messages')).toContainText('프로젝트 생성 완료');
    await expect(page.locator('.artifact-panel')).toBeVisible();
    let frame = page.frameLocator('.artifact-stage iframe');
    await expect(frame.locator('h1')).toHaveText('Version One');
    await expect(frame.locator('body')).toHaveAttribute('data-executions', '1');
    const first = await (await request.get(`/api/artifacts/${projectId}?session_id=${session.id}`)).json();
    await composer.fill('제목과 색상만 수정해'); await composer.press('Enter');
    await expect(page.locator('.messages')).toContainText('부분 수정 완료');
    await expect(frame.locator('h1')).toHaveText('Version Two');
    await expect(frame.locator('h1')).toHaveCSS('color', 'rgb(0, 0, 255)');
    await expect(frame.locator('body')).toHaveAttribute('data-executions', '1');
    const list = await (await request.get(`/api/artifacts?session_id=${session.id}`)).json();
    expect(list).toHaveLength(1); expect(list[0].id).toBe(projectId); expect(list[0].version).toBe(2);
    const second = await (await request.get(`/api/artifacts/${projectId}?session_id=${session.id}`)).json();
    expect(second.files.find(f => f.name === 'script.js')).toEqual(first.files.find(f => f.name === 'script.js'));
    await page.getByRole('button', { name: '버전 기록', exact: true }).click();
    await page.getByRole('button', { name: /v1 · 최초 저장/ }).click();
    await expect(frame.locator('h1')).toHaveText('Version One');
    await page.getByRole('button', { name: '이 버전 복원', exact: true }).click();
    await expect(page.locator('.artifact-header small')).toHaveText('버전 3 · 현재');
    const revisions = await (await request.get(`/api/artifacts/${projectId}/versions?session_id=${session.id}`)).json();
    expect(revisions.map(r => r.version)).toEqual([3, 2, 1]);
    await page.reload();
    await page.locator('.code-project-bar button').click();
    await expect(page.locator('.artifact-header small')).toHaveText('버전 3 · 현재');
    await expect(frame.locator('h1')).toHaveText('Version One');
    // Visiting an unrelated session must not retain the old pool.
    const other = await (await request.post('/api/sessions', { data: { title: '다른 코드 대화' } })).json(); sessions.push(other.id);
    await page.reload();
    await expect(page.locator('.code-project-bar')).toHaveCount(0);
    await expect(page.locator('.artifact-panel')).toHaveCount(0);
    await page.getByText('코드 프로젝트 검증', { exact: true }).first().click();
    await page.locator('.code-project-bar button').click();
    await expect(frame.locator('h1')).toHaveText('Version One');
    await page.setViewportSize({ width: 390, height: 700 });
    await expect(page.getByRole('button', { name: '버전 기록', exact: true })).toBeVisible();
    const box = await page.locator('.artifact-panel').boundingBox(); expect(box.width).toBe(390);
  } finally {
    await request.put('/api/config', { data: original });
    for (const id of sessions) await request.delete(`/api/sessions/${id}`);
    model.closeAllConnections(); await new Promise(resolve => model.close(resolve));
  }
});
