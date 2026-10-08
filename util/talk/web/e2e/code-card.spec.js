import { expect, test } from '@playwright/test';

test('keeps long chat code compact until the user expands it', async ({ page }) => {
  await page.setViewportSize({ width: 1000, height: 800 });
  await page.goto('/');
  await page.locator('.messages').evaluate((messages) => {
    const card = document.createElement('div');
    card.className = 'prose';
    card.innerHTML = `<div class="code-card code-card-long" data-code-card>
      <div class="code-card-header"><span>javascript</span><div><button type="button" data-code-toggle aria-expanded="false">전체 보기</button></div></div>
      <pre><code>${Array.from({ length: 40 }, (_, index) => `const value${index} = ${index};`).join('\n')}</code></pre>
    </div>`;
    messages.append(card);
  });

  const card = page.locator('[data-code-card]');
  const code = card.locator('pre');
  expect((await code.boundingBox())?.height).toBeLessThanOrEqual(230);
  await card.getByRole('button', { name: '전체 보기' }).click();
  await expect(card).toHaveClass(/expanded/);
  expect((await code.boundingBox())?.height).toBeGreaterThan(500);
  await card.getByRole('button', { name: '접기', exact: true }).click();
  await expect(card).not.toHaveClass(/expanded/);
});

test('preserves expanded cards and code scrolling across streamed updates and completion', async ({ page, request }) => {
  const { createServer } = await import('node:http');
  let append, finish;
  const appendGate = new Promise(resolve => { append = resolve; });
  const finishGate = new Promise(resolve => { finish = resolve; });
  const code = Array.from({ length: 40 }, (_, i) => `const value${i} = "${'wide code '.repeat(25)}";`).join('\n');
  const backend = createServer(async (req, res) => {
    if (req.method !== 'POST') { res.end(JSON.stringify({ data: [{ id: 'test-model' }] })); return; }
    let raw = ''; for await (const part of req) raw += part;
    if (!JSON.parse(raw).stream) { res.end(JSON.stringify({ choices: [{ message: { content: 'Code scroll' }, finish_reason: 'stop' }] })); return; }
    res.setHeader('Content-Type', 'text/event-stream');
    const delta = content => res.write('data: ' + JSON.stringify({ choices: [{ delta: { content }, finish_reason: null }] }) + '\n\n');
    delta('```javascript\n' + code + '\n```\n\n```javascript\n' + code);
    await appendGate;
    delta('\nconst streamedTail = 41;\n```\n\n스트리밍 추가 완료');
    await finishGate;
    res.end('data: ' + JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }] }) + '\n\ndata: [DONE]\n\n');
  });
  await new Promise(resolve => backend.listen(0, '127.0.0.1', resolve));
  const original = await (await request.get('/api/config')).json();
  let session;
  try {
    const config = structuredClone(original);
    config.model.endpoint = `http://127.0.0.1:${backend.address().port}`;
    expect((await request.put('/api/config', { data: config })).ok()).toBeTruthy();
    session = await (await request.post('/api/sessions', { data: { title: 'Streaming code scroll' } })).json();
    await page.setViewportSize({ width: 1000, height: 800 });
    await page.goto('/');
    await page.locator('textarea').first().fill('코드 두 개를 작성해.');
    await page.locator('textarea').first().press('Enter');
    const cards = page.locator('.bubble [data-code-card]');
    await expect(cards).toHaveCount(2);
    await cards.first().getByRole('button', { name: '전체 보기' }).click();
    const pane = page.locator('.messages');
    const pre = cards.nth(1).locator('pre');
    await pre.evaluate(el => { el.scrollTop = 140; el.scrollLeft = 90; el.dataset.scrollIdentity = 'same-pre'; });
    await pane.evaluate(el => { el.style.scrollBehavior = 'auto'; el.scrollTop = 240; });
    const before = await pane.evaluate(el => el.scrollTop);
    const assertRetained = async () => {
      await expect(cards.first()).toHaveClass(/expanded/);
      await expect(cards.first().getByRole('button', { name: '접기', exact: true })).toHaveAttribute('aria-expanded', 'true');
      await expect(pre).toHaveAttribute('data-scroll-identity', 'same-pre');
      await expect.poll(() => pre.evaluate(el => [el.scrollTop, el.scrollLeft])).toEqual([140, 90]);
      await expect.poll(() => pane.evaluate((el, old) => Math.abs(el.scrollTop - old), before)).toBeLessThan(5);
    };
    append();
    await expect(pane).toContainText('스트리밍 추가 완료');
    await assertRetained();
    finish();
    await expect(pane.locator('article[data-message-id]:not([data-message-id=""])')).toHaveCount(2);
    await assertRetained();
    await expect(pre.locator('code')).toContainText('const streamedTail = 41;');
  } finally {
    append(); finish();
    if (session) await request.delete(`/api/sessions/${session.id}`);
    await request.put('/api/config', { data: original });
    backend.closeAllConnections(); await new Promise(resolve => backend.close(resolve));
  }
});

test('can repeatedly expand and collapse while chunks arrive between pointer down and up', async ({ page, request }) => {
  const { createServer } = await import('node:http');
  let send, finish;
  const finished = new Promise(resolve => finish = resolve);
  const code = Array.from({length: 45}, (_, i) => `const item${i} = ${i};`).join('\n');
  const backend = createServer(async (req, res) => {
    if (req.method !== 'POST') { res.end(JSON.stringify({data:[{id:'test-model'}]})); return; }
    let body='';for await (const p of req) body+=p;
    if (!JSON.parse(body).stream) {res.end(JSON.stringify({choices:[{message:{content:'Streaming toggle'},finish_reason:'stop'}]}));return;}
    res.setHeader('Content-Type','text/event-stream');
    send = content => res.write('data: '+JSON.stringify({choices:[{delta:{content},finish_reason:null}]})+'\n\n');
    send('```js\n'+code.split('\n').slice(0,8).join('\n'));
    await finished;
    res.end('data: [DONE]\n\n');
  });
  await new Promise(r=>backend.listen(0,'127.0.0.1',r));
  const original=await (await request.get('/api/config')).json();let session;
  try {
    const cfg=structuredClone(original);cfg.model.endpoint=`http://127.0.0.1:${backend.address().port}`;
    await request.put('/api/config',{data:cfg});
    session=await (await request.post('/api/sessions',{data:{title:'Toggle during stream'}})).json();
    await page.goto('/');await page.locator('textarea').first().fill('코드를 작성해.');await page.locator('textarea').first().press('Enter');
    const card=page.locator('.bubble [data-code-card]');await expect(card).toHaveCount(1);
    await expect(card.locator('[data-code-toggle]')).toHaveCount(0);
    send('\n'+code.split('\n').slice(8).join('\n'));
    await expect(card.locator('[data-code-toggle]')).toBeVisible();
    await expect(card.locator('[data-code-collapse]')).toBeHidden();
    for(let i=0;i<3;i++) {
      const toggle=card.locator('[data-code-toggle]');
      await toggle.scrollIntoViewIfNeeded();const box=await toggle.boundingBox();
      await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.mouse.down();
      send(`\nconst chunk${i} = ${i};`);
      await expect(card.locator('code')).toContainText(`const chunk${i}`);
      await page.mouse.up();await expect(toggle).toHaveAttribute('aria-expanded','true');
      const bottom=card.locator('[data-code-collapse]');await expect(bottom).toBeVisible();
      if(i%2===0){await bottom.click();await expect(toggle).toBeFocused();}else{await toggle.click();}
      await expect(toggle).toHaveAttribute('aria-expanded','false');await expect(bottom).toBeHidden();
      const view=await toggle.boundingBox(),pane=await page.locator('.messages').boundingBox();
      expect(view.y).toBeGreaterThanOrEqual(pane.y-1);expect(view.y+view.height).toBeLessThanOrEqual(pane.y+pane.height+1);
    }
  } finally {
    finish(); if(session)await request.delete(`/api/sessions/${session.id}`);
    await request.put('/api/config',{data:original});backend.closeAllConnections();await new Promise(r=>backend.close(r));
  }
});
