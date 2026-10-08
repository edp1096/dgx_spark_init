import test from 'node:test';
import assert from 'node:assert/strict';
import { artifactsFromMessage } from './artifacts.js';

test('combines html css and javascript fences into one sandbox document', () => {
  const [artifact] = artifactsFromMessage({
    id: 7,
    role: 'assistant',
    content: '```html\n<!doctype html><html><head><title>카운터</title></head><body><button id="go">0</button></body></html>\n```\n```css\nbutton { color: red; }\n```\n```js\ndocument.querySelector("#go").onclick = () => {};\n```',
  });
  assert.equal(artifact.title, '카운터');
  assert.match(artifact.document, /Content-Security-Policy/);
  assert.match(artifact.document, /button \{ color: red; \}/);
  assert.match(artifact.document, /querySelector/);
  assert.match(artifact.document, /connect-src 'none'/);
  assert.deepEqual(artifact.files.map((item) => item.name), ['index.html', 'style.css', 'script.js']);
  assert.match(artifact.files[0].source, /href="\.\/style\.css"/);
  assert.match(artifact.files[0].source, /src="\.\/script\.js"/);
  assert.equal(artifact.files[1].source, 'button { color: red; }');
});

test('ignores ordinary code blocks and user messages', () => {
  assert.deepEqual(artifactsFromMessage({ role: 'assistant', content: '```go\npackage main\n```' }), []);
  assert.deepEqual(artifactsFromMessage({ role: 'user', content: '```html\n<p>no</p>\n```' }), []);
});

test('persistent project preview uses its exact stored files and stable identity', async () => {
  const { artifactFromProject } = await import('./artifacts.js');
  const files = [{ name: 'index.html', source: '<html><head><link rel="stylesheet" href="./theme.css"></head><body><h1>v2</h1><script src="./game.js"></script></body></html>' }, { name: 'theme.css', source: 'h1{color:blue}' }, { name: 'game.js', source: 'window.count = (window.count || 0) + 1;' }];
  const a = artifactFromProject({ id: 'pool', version: 2, title: 'game', files });
  assert.equal(a.id, 'pool');
  assert.deepEqual(a.files.map(({name,source}) => ({name,source})), files);
  assert.equal(a.document.match(/window.count =/g).length, 1);
  assert.equal(a.document.match(/h1\{color:blue\}/g).length, 1);
  assert.doesNotMatch(a.document, /src="\.\/game.js"|href="\.\/theme.css"/);
  assert.match(a.document, /connect-src 'none'/);
});

test('deferred project scripts run after body parsing in file order', async () => {
  const { artifactFromProject } = await import('./artifacts.js');
  const artifact = artifactFromProject({ id: 'defer', files: [
    { name: 'index.html', source: '<html><head><script src="first.js" defer></script><script defer src="second.js"></script></head><body><button id="go">Go</button></body></html>' },
    { name: 'first.js', source: 'window.go = document.getElementById("go");' },
    { name: 'second.js', source: 'window.go.textContent = "Ready";' },
  ] });
  assert.ok(artifact.document.indexOf('<button') < artifact.document.indexOf('window.go ='));
  assert.ok(artifact.document.indexOf('window.go =') < artifact.document.indexOf('window.go.textContent'));
  assert.equal(artifact.document.match(/window.go =/g).length, 1);
});

test('empty web entry reports a saved-file error instead of hiding it as source-only', async () => {
  const { artifactFromProject } = await import('./artifacts.js');
  const a = artifactFromProject({id:'broken',files:[{name:'index.html',source:''}]});
  assert.match(a.document, /저장된 웹 파일이 비어 있습니다/);
  assert.doesNotMatch(a.document, /소스 보기에서 확인할 수 있습니다/);
});
