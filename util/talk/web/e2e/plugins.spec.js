import { test, expect } from '@playwright/test';

test('plugin foundation exposes an empty catalog, without business fixtures', async ({ page, request }) => {
  const response = await request.get('/api/plugins');
  expect(response.ok()).toBeTruthy();
  expect(await response.json()).toEqual([]);
  await page.goto('/');
  await page.getByRole('button', { name: '▤ 라이브러리', exact: true }).click();
  await page.getByRole('button', { name: '플러그인', exact: true }).click();
  await expect(page.getByText('등록된 플러그인이 없습니다.', { exact: true })).toBeVisible();
  const unknown = await request.post('/api/plugins/fixture/enable', { data: {} });
  expect(unknown.status()).toBe(404);
});

test('generic plugin controls save grants, enable, invoke panels and show run results', async ({ page }) => {
  // Browser-only fixture. The real server catalog stays empty.
  const item = {
    manifest: { id: 'fixture', name: '테스트 확장', version: '1.0.0', description: '공통 UI 검사', data_version: 1,
      permissions: ['storage', 'jobs'], operations: [{ name: 'echo', description: '입력 확인', background: true }],
      panels: [{ id: 'panel', title: '테스트 화면', description: '선언형 화면', operations: ['echo'] }] },
    settings: { enabled: false, config: {}, grants: [], data_version: 1 }, status: 'disabled', active_runs: 0,
  };
  let runs = [];
  await page.route('**/api/plugins', route => route.fulfill({ json: [item] }));
  await page.route('**/api/plugins/fixture/*', async route => {
    const action = new URL(route.request().url()).pathname.split('/').at(-1);
    if (action === 'runs') return route.fulfill({ json: runs });
    const data = route.request().postDataJSON();
    if (action === 'configure') { item.settings.config = data.config; item.settings.grants = data.grants; }
    if (action === 'enable') { item.status = 'active'; item.settings.enabled = true; }
    if (action === 'disable') { item.status = 'disabled'; item.settings.enabled = false; }
    if (action === 'submit') {
      item.active_runs = 1;
      runs = [{ id: 'one', operation: data.operation, status: 'running', created_at: new Date().toISOString() }];
      setTimeout(() => { item.active_runs = 0; runs[0].status = 'completed'; runs[0].result = data.request.input; }, 100);
      return route.fulfill({ json: runs[0], status: 202 });
    }
    return route.fulfill({ json: { ok: true } });
  });
  await page.goto('/');
  await page.getByRole('button', { name: '▤ 라이브러리', exact: true }).click();
  await page.getByRole('button', { name: '플러그인', exact: true }).click();
  await page.getByRole('button', { name: /테스트 확장/ }).click();
  await page.getByLabel('전용 저장 공간', { exact: true }).check();
  await page.getByLabel('백그라운드 작업', { exact: true }).check();
  await page.getByRole('button', { name: '설정·권한 저장', exact: true }).click();
  await page.getByRole('button', { name: '활성화', exact: true }).click();
  await expect(page.getByRole('heading', { name: '테스트 화면', exact: true })).toBeVisible();
  await page.getByLabel('작업 입력 (JSON)', { exact: true }).fill('{"message":"hello"}');
  await page.getByRole('button', { name: '입력 확인', exact: true }).click();
  await expect(page.getByText('결과', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: '입력 확인', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: '비활성화', exact: true }).click();
  await expect(page.getByRole('heading', { name: '테스트 화면', exact: true })).toHaveCount(0);
});

test('external package installs and runs through the real sandbox and web UI', async ({ page, request }) => {
  const { mkdtemp, readFile, rm } = await import('node:fs/promises');
  const { tmpdir, arch } = await import('node:os');
  const { join, resolve } = await import('node:path');
  const { execFileSync } = await import('node:child_process');
  const { zipSync, strToU8 } = await import('fflate');
  const dir = await mkdtemp(join(tmpdir(), 'talk-plugin-ui-'));
  try {
    execFileSync('go', ['build', '-o', join(dir, 'plugin'), './pluginsdk/example'], { cwd: resolve(import.meta.dirname, '../..'), env: { ...process.env, CGO_ENABLED: '0' } });
    const manifest = { id:'example', name:'외부 예제', version:'1.0.0', api_version:1, data_version:1, description:'실제 설치 테스트', permissions:[],
      executable:'plugin', platform:`linux/${arch() === 'arm64' ? 'arm64' : 'amd64'}`,
      operations:[{ name:'echo', description:'입력 반환', parameters:{type:'object'}, tool:false, background:false, timeout_seconds:2 }],
      panels:[{id:'main',title:'예제 화면',operations:['echo']}] };
    const archive = zipSync({'plugin.json':strToU8(JSON.stringify(manifest)),plugin:new Uint8Array(await readFile(join(dir,'plugin')))});
    await page.goto('/');
    await page.getByRole('button',{name:'▤ 라이브러리',exact:true}).click();
    await page.getByRole('button',{name:'플러그인',exact:true}).click();
    await page.getByLabel('플러그인 패키지 (.zip)',{exact:true}).setInputFiles({name:'example.zip',mimeType:'application/zip',buffer:Buffer.from(archive)});
    await page.getByRole('button',{name:'설치·업데이트',exact:true}).click();
    await page.getByRole('button',{name:/외부 예제/}).click();
    await page.getByRole('button',{name:'활성화',exact:true}).click();
    await expect(page.getByRole('heading',{name:'예제 화면',exact:true})).toBeVisible();
    await page.getByLabel('작업 입력 (JSON)',{exact:true}).fill('{"external":"verified"}');
    await page.getByRole('button',{name:'입력 반환',exact:true}).click();
    await expect(page.locator('pre').first()).toContainText('verified');
    await page.getByRole('button',{name:'비활성화',exact:true}).click();
    await page.getByLabel('제거할 때 데이터도 삭제',{exact:true}).check();
    page.once('dialog',dialog=>dialog.accept());
    await page.getByRole('button',{name:'제거',exact:true}).click();
    await expect(page.getByText('등록된 플러그인이 없습니다.',{exact:true})).toBeVisible();
  } finally {
    await request.post('/api/plugins/example/disable',{data:{}});
    await request.post('/api/plugins/example/remove',{data:{purge:true}});
    await rm(dir,{recursive:true,force:true});
  }
});
